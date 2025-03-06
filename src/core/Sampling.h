#pragma once

#include <cuda_runtime.h>
#include <thrust/random.h>
#include "utils/utilities.h"
__device__ __constant__ float2* pmj_samples;
__device__ __constant__ uchar2* blue_noise_textures;
enum struct SampleDimension:int {
    FILTER,
    APERTURE,

    RUSSIAN_ROULETTE,
    NEE_LIGHT,
    NEE_TRIANGLE,
    BSDF_0,
    BSDF_1,

    NUM_DIMENSIONS,
    NUM_BOUNCE = 5 // Last 5 dimensions are reused every bounce
};

 /**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__device__ inline
unsigned int utilhash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
}

__device__ inline
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

/**
 * Computes a cosine-weighted random direction on a hemisphere surface.
 * Used for diffuse lighting.
 */
__device__ inline
glm::vec3 calculateRandomDirectionOnHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or 
    // whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


// Based on: http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
__device__ float2 sample_disk(float u1, float u2) {
    float a = 2.0f * u1 - 1.0f;
    float b = 2.0f * u2 - 1.0f;

    float phi, r;
    if (a * a > b * b) {
        r = a;
        phi = 0.25f * PI * (b / a);
    }
    else {
        r = b;
        phi = 0.5f * PI - 0.25f * PI * (a / b);
    }

    return r * sincos(phi);
}

// Based on: Heitz - Sampling the GGX Distribution of Visible Normals
__device__ glm::vec3 sample_visible_normals_ggx(glm::vec3 omega, float alpha_x, float alpha_y, float u1, float u2) {
    // Transform the view direction to the hemisphere configuration
    glm::vec3 v = glm::normalize(glm::vec3(alpha_x * omega.x, alpha_y * omega.y, omega.z));

    // Orthonormal basis (with special case if cross product is zero)
    float length_squared = v.x * v.x + v.y * v.y;
    glm::vec3 axis_1 = length_squared > 0.0f ? glm::vec3(-v.y, v.x, 0.0f) / sqrtf(length_squared) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 axis_2 = glm::cross(v, axis_1);

    // Parameterization of the projected area
    float2 d = sample_disk(u1, u2);
    float t1 = d.x;
    float t2 = lerp(safe_sqrt(1.0f - t1 * t1), d.y, 0.5f + 0.5f * v.z);

    // Reproject onto hemisphere
    glm::vec3 n_h = t1 * axis_1 + t2 * axis_2 + safe_sqrt(1.0f - t1 * t1 - t2 * t2) * v;

    // Transform the normal back to the ellipsoid configuration
    return glm::normalize(glm::vec3(alpha_x * n_h.x, alpha_y * n_h.y, n_h.z));
}

// Based on: Heitz - Sampling the GGX Distribution of Visible Normals
__device__ float3 sample_visible_normals_ggx(float3 omega, float alpha_x, float alpha_y, float u1, float u2) {
    // Transform the view direction to the hemisphere configuration
    float3 v = normalize(make_float3(alpha_x * omega.x, alpha_y * omega.y, omega.z));

    // Orthonormal basis (with special case if cross product is zero)
    float length_squared = v.x * v.x + v.y * v.y;
    float3 axis_1 = length_squared > 0.0f ? make_float3(-v.y, v.x, 0.0f) / sqrtf(length_squared) : make_float3(1.0f, 0.0f, 0.0f);
    float3 axis_2 = cross(v, axis_1);

    // Parameterization of the projected area
    float2 d = sample_disk(u1, u2);
    float t1 = d.x;
    float t2 = lerp(safe_sqrt(1.0f - t1 * t1), d.y, 0.5f + 0.5f * v.z);

    // Reproject onto hemisphere
    float3 n_h = t1 * axis_1 + t2 * axis_2 + safe_sqrt(1.0f - t1 * t1 - t2 * t2) * v;

    // Transform the normal back to the ellipsoid configuration
    return normalize(make_float3(alpha_x * n_h.x, alpha_y * n_h.y, n_h.z));
}

__device__ glm::vec3 sample_cosine_weighted_direction(float u1, float u2) {
    float2 d = sample_disk(u1, u2);
    return glm::vec3(d.x, d.y, safe_sqrt(1.0f - dot(d, d)));
}
