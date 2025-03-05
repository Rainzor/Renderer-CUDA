#pragma once

#include <cuda_runtime.h>
#include <utils/utilities.h>
#include "glm/glm.hpp"

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__device__ inline glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - 0.001f) * glm::normalize(r.direction);
}

__device__ inline glm::vec3 reflect_direction(glm::vec3 direction, glm::vec3 normal) {
	return 2.0f * glm::dot(direction, normal) * normal - direction;
}

__device__ inline glm::vec3 refract_direction(glm::vec3 direction, glm::vec3 normal, float eta) {
	float cos_theta = glm::dot(direction, normal);
	float k = 1.0f - eta * eta * (1.0f - square(cos_theta));
	return (eta * cos_theta - safe_sqrt(k)) * normal - eta * direction;
}

__device__ inline float3 reflect_direction(float3 direction, float3 normal) {
	return 2.0f * dot(direction, normal) * normal - direction;
}

__device__ inline float3 refract_direction(float3 direction, float3 normal, float eta) {
	float cos_theta = dot(direction, normal);
	float k = 1.0f - eta * eta * (1.0f - square(cos_theta));
	return (eta * cos_theta - safe_sqrt(k)) * normal - eta * direction;
}