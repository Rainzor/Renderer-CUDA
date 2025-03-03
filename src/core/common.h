#pragma once

#include "glm/glm.hpp"
#include "material.h"
#include "shape.h"
#include "bvh.h"
#include "light.h"
#include "ray.h"
#include "camera.h"
#include "texture.h"



#define BLOCK_SIZE1D 64
#define BLOCK_SIZE2D (dim3(8, 8))

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define USE_NEE 1

// BSDF LUTS
#define LUT_DIELECTRIC_DIM_IOR 16
#define LUT_DIELECTRIC_DIM_ROUGHNESS 16
#define LUT_DIELECTRIC_DIM_COS_THETA 16

#define LUT_DIELECTRIC_MIN_IOR 1.0001f
#define LUT_DIELECTRIC_MAX_IOR 2.5f

#define LUT_CONDUCTOR_DIM_ROUGHNESS 32
#define LUT_CONDUCTOR_DIM_COS_THETA 32

