#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

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
 
namespace utilityCore {
     /**
     * Handy-dandy hash function that provides seeds for random number generation.
     */
    __host__ __device__ inline unsigned int utilhash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }
    __device__ __host__ inline
        float clamp(float f, float min, float max) {
        if (f < min) {
            return min;
        }
        else if (f > max) {
            return max;
        }
        else {
            return f;
        }
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
}