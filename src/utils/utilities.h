#pragma once

#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "cuda_math.h"

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};




__device__ inline float sign(float x) {
    return copysignf(1.0f, x);
}

__device__ inline constexpr bool is_power_of_two(unsigned x) {
    return x != 0 && (x & (x - 1)) == 0;
}

__device__ inline float square(float x) {
    return x * x;
}

__device__ inline float cube(float x) {
    return x * x * x;
}

__device__ inline float remap(float value, float old_min, float old_max, float new_min, float new_max) {
    return new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min);
}

__device__ inline float safe_sqrt(float x) {
    return sqrtf(fmaxf(0.0f, x));
}

__device__ inline float2 safe_sqrt(float2 v) { return make_float2(safe_sqrt(v.x), safe_sqrt(v.y)); }
__device__ inline float3 safe_sqrt(float3 v) { return make_float3(safe_sqrt(v.x), safe_sqrt(v.y), safe_sqrt(v.z)); }
__device__ inline float4 safe_sqrt(float4 v) { return make_float4(safe_sqrt(v.x), safe_sqrt(v.y), safe_sqrt(v.z), safe_sqrt(v.w)); }

__device__ inline float abs_dot(float3 a, float3 b) {
    return fabsf(dot(a, b));
}

__device__ inline float2 sincos(float x) {
    return make_float2(sinf(x), cosf(x));
}

template<typename T>
__device__ inline T divide_difference_by_sum(const T& a, const T& b) {
    return (a - b) / (a + b);
};


__device__ inline float online_average(float avg, float sample, int n) {
    if (n == 0) {
        return sample;
    }
    else {
        return avg + (sample - avg) / float(n);
    }
}

template<typename T>
__device__ T lerp(T const& a, T const& b, float t) {
    return (1.0f - t) * a + t * b;
}


/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__device__ __host__ inline glm::vec3 tone_mapping(glm::vec3 color, float adapted_lum = 1.0f) {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;

    color *= adapted_lum;
    return (color * (A * color + B)) / (color * (C * color + D) + E);
}

__device__ __host__ inline float gamma_to_linear(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }
    else if (x >= 1.0f) {
        return 1.0f;
    }
    else if (x < 0.04045f) {
        return x / 12.92f;
    }
    else {
        return powf((x + 0.055f) / 1.055f, 2.4f);
    }
}

__device__ __host__ inline float linear_to_gamma(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }
    else if (x >= 1.0f) {
        return 1.0f;
    }
    else if (x < 0.0031308f) {
        return 12.92f * x;
    }
    else {
        return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
    }
}



extern glm::vec3 clampRGB(glm::vec3 color);
extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
extern std::string convertIntToString(int number);