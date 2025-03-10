#pragma once
#include <cuda_runtime.h>
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

#define ROUGHNESS_CUTOFF (0.05f)

#define PLASTIC_IOR 1.5f
#define PLASTIC_ETA 0.6666666666666667f 


// Raytracing
#define MAX_BOUNCES 128

// RNG
#define PMJ_NUM_SEQUENCES 64
#define PMJ_NUM_SAMPLES_PER_SEQUENCE 4096

#define BLUE_NOISE_NUM_TEXTURES 16
#define BLUE_NOISE_TEXTURE_DIM 128
