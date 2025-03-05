#pragma once

#include <cuda_texture_types.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "ray.h"
#include "material.h"
#include "Sampling.h"
#include "KullaConty.h"


// MIS balance heuristic
__device__ float powerHeuristic(float f, float g, int nf=1, int ng=1) {
    f = nf * f;
    g = ng * g;
    return (f * f) / (g * g + f * f);
}

__device__ 
glm::vec3 sample_albedo(const glm::vec3 diffuse, const cudaTextureObject_t* texObjs, int texture_id, const glm::vec2 uv) {
    glm::vec3 albedo = diffuse;
    if (texture_id != -1)
    {
        float4 texColor = tex2D<float4>(texObjs[texture_id], uv.x, uv.y);
        albedo *= glm::vec3(texColor.x, texColor.y, texColor.z);
    }
    return albedo;
}
__device__ 
bool dielectric_sample(const MaterialDielectric& material, 
                        const bool entering_material,
                        const glm::vec3& omega_i, 
                        glm::vec3& omega_o,
                        float& pdf, 
                        glm::vec3& throughput,
    thrust::default_random_engine& rng) {
    float eta = entering_material ? material.ior : 1.f / material.ior;
    float alpha_x = roughness_to_alpha(material.linear_roughness);
    float alpha_y = roughness_to_alpha(material.linear_roughness);

    float cos_theta_i = omega_i.z;

    float E_i = dielectric_directional_albedo(material.ior, material.linear_roughness, cos_theta_i, entering_material);

    float F_avg = average_fresnel(material.ior);
    if (!entering_material) {
        F_avg = 1.f - (1.f - F_avg) / square(material.ior);
    }

    float E_avg_enter = dielectric_albedo(material.ior, material.linear_roughness, true);
    float E_avg_leave = dielectric_albedo(material.ior, material.linear_roughness, false);

    float x = kulla_conty_dielectric_reciprocity_factor(E_avg_enter, E_avg_leave);
    float ratio = (entering_material ? x : (1.0f - x)) * (1.0f - F_avg);

    float F;
    bool reflected;

    glm::vec3 omega_m;
    //glm::vec3 omega_o;

	//thrust::pair<float2, float2> rand_bsdf);
	thrust::random::uniform_real_distribution<float> u01(0, 1);
	float2 rand_bsdf_0 = make_float2(u01(rng), u01(rng));
	float2 rand_bsdf_1 = make_float2(u01(rng), u01(rng));

    if (rand_bsdf_0.x < E_i) {
        // Sample single scatter component
		omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_bsdf_1.x, rand_bsdf_1.x);

		F = fresnel_dielectric(glm::dot(omega_i, omega_m), eta);
        reflected = rand_bsdf_0.y < F;
		if (reflected) {
			omega_o = reflect_direction(omega_i, omega_m);
		}
		else {
			omega_o = refract_direction(omega_i, omega_m, eta);
		}
    }
    else {
        // Sample multiple scatter component
        omega_o = sample_cosine_weighted_direction(rand_bsdf_1.x, rand_bsdf_1.y);

		reflected = rand_bsdf_0.y > ratio;
        if (reflected) {
			omega_m = glm::normalize(omega_i + omega_o);
        }
        else {
			omega_o = -omega_o;
			omega_m = glm::normalize(eta*omega_i + omega_o);
        }
        omega_m *= glm::sign(omega_m.z);
		F = fresnel_dielectric(glm::dot(omega_i, omega_m), eta);
    }

    if (reflected ^ (omega_o.z >= 0.f)) return false;

	const float3 omega_i3 = make_float3(omega_i.x, omega_i.y, omega_i.z);
	const float3 omega_o3 = make_float3(omega_o.x, omega_o.y, omega_o.z);
	const float3 omega_m3 = make_float3(omega_m.x, omega_m.y, omega_m.z);

    float D = ggx_D(omega_m3, alpha_x, alpha_y);
    float G1 = ggx_G1(omega_i3, alpha_x, alpha_y);
    float G2 = ggx_G2(omega_o3, omega_i3, omega_m3, alpha_x, alpha_y);

    float i_dot_m = abs_dot(omega_i3, omega_m3);
    float o_dot_m = abs_dot(omega_o3, omega_m3);

    float bsdf_single;
    float bsdf_multi;

    float pdf_single;
    float pdf_multi;

    if (reflected) {
        bsdf_single = F * G2 * D / (4.0f * omega_i.z); // BRDF times cos(theta_o)
        pdf_single = F * G1 * D / (4.0f * omega_i.z);

        float E_o = dielectric_directional_albedo(material.ior, material.linear_roughness, omega_o.z, entering_material);
        float E_avg = entering_material ? E_avg_enter : E_avg_leave;

        bsdf_multi = (1.0f - ratio) * fabsf(omega_o.z) * kulla_conty_multiscatter_lobe(E_i, E_o, E_avg);
        pdf_multi = (1.0f - ratio) * fabsf(omega_o.z) * ONE_OVER_PI;
    } else {
        bsdf_single = (1.0f - F) * G2 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m) * square(eta)); // BRDF times cos(theta_o)
        pdf_single = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

        float E_o = dielectric_directional_albedo(material.ior, material.linear_roughness, omega_o.z, !entering_material);
        float E_avg = entering_material ? E_avg_leave : E_avg_enter; // NOTE: inverted!

        bsdf_multi = ratio * fabsf(omega_o.z) * kulla_conty_multiscatter_lobe(E_i, E_o, E_avg);
        pdf_multi = ratio * fabsf(omega_o.z) * ONE_OVER_PI;

        // Update the Medium based on whether we are transmitting into or out of the Material
        //if (entering_material) {
        //    medium_id = material.medium_id;
        //}
        //else {
        //    medium_id = INVALID;
        //}
    }

    pdf = lerp(pdf_multi, pdf_single, E_i);
    throughput *= (bsdf_single + bsdf_multi) / pdf;

    //direction_out = local_to_world(omega_o, tangent, bitangent, normal);
	return pdf > 0.f;
}


__device__ bool dielectric_eval(const MaterialDielectric& material,
                    const bool entering_material,
                    const glm::vec3& omega_i,
                    const glm::vec3& omega_o, 
                    float cos_theta_o, 
                    glm::vec3& bsdf,
                    float& pdf) {
    bool reflected = omega_o.z >= 0.0f;
    float eta = entering_material ? material.ior : 1.f / material.ior;
	glm::vec3 omega_m;
	if (reflected) {
		omega_m = glm::normalize(omega_i + omega_o);
	}
	else {
		omega_m = glm::normalize(eta * omega_i + omega_o);
	}
	omega_m *= glm::sign(omega_m.z);

	const float3 omega_i3 = make_float3(omega_i.x, omega_i.y, omega_i.z);
	const float3 omega_o3 = make_float3(omega_o.x, omega_o.y, omega_o.z);
	const float3 omega_m3 = make_float3(omega_m.x, omega_m.y, omega_m.z);

	float i_dot_m = abs_dot(omega_i3, omega_m3);
	float o_dot_m = abs_dot(omega_o3, omega_m3);

    float alpha_x = roughness_to_alpha(material.linear_roughness);
    float alpha_y = roughness_to_alpha(material.linear_roughness);

    float F = fresnel_dielectric(i_dot_m, eta);
    float D = ggx_D(omega_m3, alpha_x, alpha_y);
    float G1 = ggx_G1(omega_i3, alpha_x, alpha_y);
    float G2 = ggx_G2(omega_o3, omega_i3, omega_m3, alpha_x, alpha_y);
    
    float F_avg = average_fresnel(material.ior);
    if (!entering_material) {
        F_avg = 1.0f - (1.0f - F_avg) / square(material.ior);
    }

    float E_avg_enter = dielectric_albedo(material.ior, material.linear_roughness, true);
    float E_avg_leave = dielectric_albedo(material.ior, material.linear_roughness, false);

    float x = kulla_conty_dielectric_reciprocity_factor(E_avg_enter, E_avg_leave);
    float ratio = (entering_material ? x : (1.0f - x)) * (1.0f - F_avg);

    float E_i = dielectric_directional_albedo(material.ior, material.linear_roughness, omega_i.z, entering_material);

    float bsdf_single;
    float bsdf_multi;

    float pdf_single;
    float pdf_multi;

    if (reflected) {
        bsdf_single = F * G2 * D / (4.0f * omega_i.z); // BRDF times cos(theta_o)
        pdf_single = F * G1 * D / (4.0f * omega_i.z);

        float E_o = dielectric_directional_albedo(material.ior, material.linear_roughness, omega_o.z, entering_material);
        float E_avg = entering_material ? E_avg_enter : E_avg_leave;

        bsdf_multi = (1.0f - ratio) * fabsf(omega_o.z) * kulla_conty_multiscatter_lobe(E_i, E_o, E_avg);
        pdf_multi = (1.0f - ratio) * fabsf(omega_o.z) * ONE_OVER_PI;
    }
    else {
        bsdf_single = (1.0f - F) * G2 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m) * square(eta)); // BRDF times cos(theta_o)
        pdf_single = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

        float E_o = dielectric_directional_albedo(material.ior, material.linear_roughness, omega_o.z, !entering_material);
        float E_avg = entering_material ? E_avg_leave : E_avg_enter; // NOTE: inverted!

        bsdf_multi = ratio * fabsf(omega_o.z) * kulla_conty_multiscatter_lobe(E_i, E_o, E_avg);
        pdf_multi = ratio * fabsf(omega_o.z) * ONE_OVER_PI;
    }

    bsdf = glm::vec3(bsdf_single + bsdf_multi);

    pdf = lerp(pdf_multi, pdf_single, E_i);
	return pdf > 0.f;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 * 
 * This method returns a Sample struct with the scattered ray direction, the bsdf and the pdf.
 * You may need to change the parameter list for your purposes!
 */
__device__
void scatterRay(
    Integrator::CUDATracer& path,
    Integrator::CUDAShadowRay& shadow_ray,
        const Integrator::CUDARecord & record,
        const Integrator::CUDAMaterial *material,
	    const cudaTextureObject_t* texObjs,
        const Light* lights,
	    const int& num_lights,
        const float& total_lights_weight,
        thrust::default_random_engine& rng) {
    // ! Scatter the ray according to the type of material
    thrust::uniform_real_distribution<float> u01(0, 1);
	Ray ray_in = path.ray;
    glm::vec3 throughput_in = path.throughput;
    glm::vec3 BSDF;
    glm::vec3 direct_out;
	glm::vec3 abedo = glm::vec3(0.f);

    glm::vec3 normal = record.outside ? record.surfaceNormal : -record.surfaceNormal;
	glm::vec3 tangent, bitangent;
	orthonormal_basis(normal, tangent, bitangent);

	glm::vec3 omega_i = world_to_local(-ray_in.direction, tangent, bitangent, normal);
    if (omega_i.z <= 0.0f) return;


    if (material->type == MaterialType::SPECULAR){ // Perfect Reflection
        direct_out = reflect_direction(-ray_in.direction, record.surfaceNormal);
        float cosTheta = glm::dot(direct_out, record.surfaceNormal);
        abedo = sample_albedo(material->diffuse.diffuse, texObjs, material->diffuse.texture_id, record.uv);
        BSDF = abedo / cosTheta;
        path.last_pdf = 1.f;
		path.from_specular = true;
    } 
    else if (material->type == MaterialType::DIFFUSE){ // Lambertian
		// Sample
        direct_out = calculateRandomDirectionOnHemisphere(record.surfaceNormal, rng);
		abedo = sample_albedo(material->specular.diffuse, texObjs, material->specular.texture_id, record.uv);
        BSDF = abedo * ONE_OVER_PI;
		// Evaluate
		path.last_pdf = glm::dot(direct_out, record.surfaceNormal) * ONE_OVER_PI;
		path.from_specular = false;
		path.throughput *= abedo;
	}
	else if (material->type == MaterialType::DIELECTRIC) { // Refraction
        MaterialDielectric dielectric;
		dielectric.ior = material->dielectric.ior;
		dielectric.linear_roughness = material->dielectric.linear_roughness;
		glm::vec3 omega_o;
		float pdf;
        if (!dielectric_sample(dielectric, record.outside, omega_i, omega_o, pdf, path.throughput, rng))
            return;
		direct_out = local_to_world(omega_o, tangent, bitangent, normal);
        path.last_pdf = pdf;
        path.from_specular = dielectric.linear_roughness < ROUGHNESS_CUTOFF;
	}
	else if (material->type == MaterialType::LIGHT) {// No scattering
		assert(false);
    }
	path.ray.origin = getPointOnRay(path.ray, record.t);
    path.ray.direction = direct_out;
    
	// Shadow ray: Next Event Estimation
	if (!path.from_specular) {
		float rand_lgts = u01(rng) * total_lights_weight;
		int light_idx = 0;
        for (int i = 0; i < num_lights; i++) {
            rand_lgts -= lights[i].area * lights[i].scale * lights[i].emittance;
            if (rand_lgts <= 0) {
                light_idx = i;
                break;
            }
        }
        const glm::vec3& hit_point = path.ray.origin;
		glm::vec3 light_pos, light_normal;
		const Light& light = lights[light_idx];
        light.sample(light_pos, light_normal, rng);
		glm::vec3 dir_to_light = light_pos - hit_point;
		float distance = glm::length(dir_to_light);
		dir_to_light = dir_to_light / distance;

        shadow_ray.ray.origin = hit_point;
		shadow_ray.ray.direction = dir_to_light;
		shadow_ray.t_max = distance - EPSILON;

		if (material->type == MaterialType::DIFFUSE) {
            float cosTheta = glm::dot(dir_to_light, record.surfaceNormal);
			BSDF = abedo / PI;
			float cosTheta_light = glm::dot(-dir_to_light, light_normal);
			if (cosTheta > 0 && cosTheta_light > 0) {
                float bsdf_pdf = cosTheta / PI;
				float light_pdf = light.emittance * distance * distance / (cosTheta_light * total_lights_weight);
				float mis_weight = powerHeuristic(light_pdf, bsdf_pdf);
				shadow_ray.radiance_direct = throughput_in * BSDF * light.emittance * mis_weight / light_pdf;
			}
		}
        else if (material->type == MaterialType::DIELECTRIC) {
			MaterialDielectric dielectric;
			dielectric.ior = material->dielectric.ior;
			dielectric.linear_roughness = material->dielectric.linear_roughness;
			glm::vec3 omega_o = world_to_local(dir_to_light, tangent, bitangent, normal);
			float pdf;
			glm::vec3 bsdf;
			float cosTheta = glm::dot(dir_to_light, record.surfaceNormal);
			float cosTheta_light = glm::dot(-dir_to_light, light_normal);
            if (dielectric_eval(dielectric, record.outside, omega_i, omega_o, cosTheta, bsdf, pdf)) {
				float light_pdf = light.emittance * distance * distance / (cosTheta_light * total_lights_weight);
				float mis_weight = powerHeuristic(light_pdf, pdf);
				shadow_ray.radiance_direct = throughput_in * bsdf * light.emittance * mis_weight / light_pdf;
            }
        
        }

    }
    else {
		shadow_ray.radiance_direct = glm::vec3(0.f);
        shadow_ray.t_max = -1.0f;
    }
}