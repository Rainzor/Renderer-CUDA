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
	Ray ray_o = path.ray;
    glm::vec3 BSDF;
    glm::vec3 direct_i = glm::vec3(0.f);
	glm::vec3 direct_o = ray_o.direction;
	glm::vec3 abedo = glm::vec3(0.f);
    if (material->type == MaterialType::SPECULAR){ // Perfect Reflection
        direct_i = glm::reflect(direct_o, record.surfaceNormal);
        float cosTheta = glm::dot(direct_i, record.surfaceNormal);
        abedo = sample_albedo(material->diffuse.diffuse, texObjs, material->diffuse.texture_id, record.uv);
        BSDF = abedo / cosTheta;
        path.last_pdf = 1.f;
		path.from_specular = true;
    } else if (material->type == MaterialType::DIFFUSE){ // Lambertian
		// Sample
        direct_i = calculateRandomDirectionOnHemisphere(record.surfaceNormal, rng);
		abedo = sample_albedo(material->specular.diffuse, texObjs, material->specular.texture_id, record.uv);
        BSDF = abedo / PI;
		// Evaluate
        path.last_pdf = glm::dot(direct_i, record.surfaceNormal) / PI;
		path.from_specular = false;
	}
	else if (material->type == MaterialType::DIELECTRIC) { // Refraction

		float eta = record.outside ? material->dielectric.ior : 1.f / material->dielectric.ior;
		glm::vec3 normal = record.outside ? record.surfaceNormal : -record.surfaceNormal;
        //float E_i = dielectric_directional_albedo(material.ior, material.linear_roughness, omega_i.z, entering_material);
	}
	else if (material->type == MaterialType::LIGHT) {// No scattering
		assert(false);
    }
    float cosTheta = glm::dot(direct_i, record.surfaceNormal);
	path.ray.origin = getPointOnRay(path.ray, record.t);
    path.ray.direction = direct_i;
	glm::vec3 throughput_in = path.throughput;
	path.throughput *= BSDF * cosTheta / path.last_pdf;
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
			cosTheta = glm::dot(dir_to_light, record.surfaceNormal);
			BSDF = abedo / PI;
			float cosTheta_light = glm::dot(-dir_to_light, light_normal);
			if (cosTheta > 0 && cosTheta_light > 0) {
                float bsdf_pdf = cosTheta / PI;
				float light_pdf = light.emittance * distance * distance / (cosTheta_light * total_lights_weight);
				float mis_weight = powerHeuristic(light_pdf, bsdf_pdf);
				shadow_ray.radiance_direct = throughput_in * BSDF * light.emittance * mis_weight / light_pdf;
			}
		}
    }
    else {
		shadow_ray.radiance_direct = glm::vec3(0.f);
        shadow_ray.t_max = -1.0f;
    }
}