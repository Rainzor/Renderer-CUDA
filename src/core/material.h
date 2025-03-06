#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utils/utilities.h"

#define ROUGHNESS_CUTOFF (0.05f)
enum MaterialType: unsigned char {
    LIGHT,
    DIFFUSE,
    SPECULAR,
	DIELECTRIC,
    CONDUCTOR,
    PLASTIC,
};

struct Material {
	enum MaterialType type = DIFFUSE;
	int texture_id = -1;
	float ior = 1.0f;
	float emittance = 0.0f;
	float roughness = 0.5f;
	glm::vec3 diffuse = glm::vec3(1.0f);
	glm::vec3 eta = glm::vec3(0.0f);
	glm::vec3 k = glm::vec3(0.0f);
};

struct MaterialLight {
	glm::vec3 emission;
};

struct MaterialDiffuse {
	glm::vec3 diffuse;
	int    texture_id;
};

struct MaterialPlastic {
	glm::vec3 diffuse;
	int    texture_id;
	float  linear_roughness;
};

struct MaterialDielectric {
	int   medium_id;
	float ior;
	float linear_roughness;
};

struct MaterialConductor {
	glm::vec3 eta;
	float  linear_roughness;
	glm::vec3 k;
};


__device__ inline float roughness_to_alpha(float linear_roughness) {
	return fmaxf(1e-6f, (linear_roughness* linear_roughness));
}

__device__ inline float fresnel_dielectric(float cos_theta_i, float eta) {
	float sin_theta_o2 = eta * eta * (1.0f - square(cos_theta_i));
	if (sin_theta_o2 >= 1.0f) {
		return 1.0f; // Total internal reflection (TIR)
	}

	float cos_theta_o = safe_sqrt(1.0f - sin_theta_o2);

	float p = divide_difference_by_sum(eta * cos_theta_i, cos_theta_o);
	float s = divide_difference_by_sum(cos_theta_i, eta * cos_theta_o);

	return 0.5f * (p * p + s * s);
}


// Formula from Shirley - Physically Based Lighting Calculations for Computer Graphics
__device__ inline float3 fresnel_conductor(float cos_theta_i, float3 eta, float3 k) {
	float cos_theta_i2 = square(cos_theta_i);
	float sin_theta_i2 = 1.0f - cos_theta_i2;

	float3 inner = eta * eta - k * k - sin_theta_i2;
	float3 a2_plus_b2 = safe_sqrt(inner * inner + 4.0f * k * k * eta * eta);
	float3 a = safe_sqrt(0.5f * (a2_plus_b2 + inner));

	float3 s2 = divide_difference_by_sum(a2_plus_b2 + cos_theta_i2, 2.0f * a * cos_theta_i);
	float3 p2 = divide_difference_by_sum(a2_plus_b2 * cos_theta_i2 + square(sin_theta_i2), 2.0f * a * cos_theta_i * sin_theta_i2) * s2;

	return 0.5f * (p2 + s2);
}

__device__ inline float average_fresnel(float ior) {
	// Approximation by Kully-Conta 2017
	return (ior - 1.0f) / (4.08567f + 1.00071f * ior);
}

__device__ inline float3 average_fresnel(float3 eta, float3 k) {
	// Approximation by d'Eon (Hitchikers Guide to Multiple Scattering)
	float3 numerator = eta * (133.736f - 98.9833f * eta) + k * (eta * (59.5617f - 3.98288f * eta) - 182.37f) + ((0.30818f * eta - 13.1093f) * eta - 62.5919f) * k * k - 8.21474f;
	float3 denominator = k * (eta * (94.6517f - 15.8558f * eta) - 187.166f) + (-78.476 * eta - 395.268f) * eta + (eta * (eta - 15.4387f) - 62.0752f) * k * k;
	return numerator / denominator;
}

// Distribution of Normals term D for the GGX microfacet model
__device__ inline float ggx_D(float3 micro_normal, float alpha_x, float alpha_y) {
	if (micro_normal.z < 1e-6f) {
		return 0.0f;
	}

	float sx = -micro_normal.x / (micro_normal.z * alpha_x);
	float sy = -micro_normal.y / (micro_normal.z * alpha_y);

	float sl = 1.0f + sx * sx + sy * sy;

	float cos_theta_2 = micro_normal.z * micro_normal.z;
	float cos_theta_4 = cos_theta_2 * cos_theta_2;

	return 1.0f / (sl * sl * PI * alpha_x * alpha_y * cos_theta_4);
}

__device__ inline float ggx_lambda(float3 omega, float alpha_x, float alpha_y) {
	return 0.5f * (sqrtf(1.0f + (square(alpha_x * omega.x) + square(alpha_y * omega.y)) / square(omega.z)) - 1.0f);
}

// Monodirectional Smith shadowing/masking term
__device__ inline float ggx_G1(float3 omega, float alpha_x, float alpha_y) {
	return 1.0f / (1.0f + ggx_lambda(omega, alpha_x, alpha_y));
}

// Height correlated shadowing and masking term
__device__ inline float ggx_G2(float3 omega_o, float3 omega_i, float3 omega_m, float alpha_x, float alpha_y) {
	bool omega_i_backfacing = dot(omega_i, omega_m) * omega_i.z <= 0.0f;
	bool omega_o_backfacing = dot(omega_o, omega_m) * omega_o.z <= 0.0f;

	if (omega_i_backfacing || omega_o_backfacing) {
		return 0.0f;
	}
	else {
		return 1.0f / (1.0f + ggx_lambda(omega_o, alpha_x, alpha_y) + ggx_lambda(omega_i, alpha_x, alpha_y));
	}
}