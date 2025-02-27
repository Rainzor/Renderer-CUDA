#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - 0.001f) * glm::normalize(r.direction);
}