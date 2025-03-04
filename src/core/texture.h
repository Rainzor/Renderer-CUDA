#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <string>
#include "stb_image.h"
#include "utils/utilities.h"

struct Texture {
    public:
        int width = 0;
        int height = 0;

		glm::u8vec4* pixels = nullptr;
		glm::vec4* pixelsf = nullptr;

        bool load(const std::string& file_name, bool gamma = true);
        bool loadf(const std::string & file_name);
    };

template<typename T>
struct CUDATexture {
	cudaTextureObject_t texture;

	__device__ inline T get(float s) const {
		return tex1D<T>(texture, s);
	}

    __device__ inline T get(float s, float t) const{
        return tex2D<T>(texture, s, t);
    }

    __device__ inline T get(float s, float t, float r) const {
		return tex3D<T>(texture, s, t, r);
	}

	__device__ inline T get_lod(float s, float t, float lod) const {
		return tex2DLod<T>(texture, s, t, lod);
	}

	__device__ inline T get_grad(float s, float t, float2 dx, float2 dy) const {
		return tex2DGrad<T>(texture, s, t, dx, dy);
	}
};

// Type safe wrapper around Surface Object
template<typename T>
struct CUDASurface {
	cudaSurfaceObject_t surface;

	void init(cudaArray_t data) {
		cudaResourceDesc resDesc = {};
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = data;
		cudaSurfaceObject_t surface;
		cudaCreateSurfaceObject(&surface, &resDesc);
		this->surface = surface;
	}

	void free() {
		cudaDestroySurfaceObject(surface);
	}	

	__device__ inline T get(int x, int y) const {
		T value;
		surf2Dread<T>(&value, surface, x * sizeof(T), y, cudaBoundaryModeClamp);
		return value;
	}

	__device__ inline T get(int x, int y, int z) const {
		T value;
		surf3Dread<T>(&value, surface, x * sizeof(T), y, z, cudaBoundaryModeClamp);
		return value;
	}

	__device__ inline void set(int x, int y, const T & value) {
		surf2Dwrite<T>(value, surface, x * sizeof(T), y);
	}

	__device__ inline void set(int x, int y, int z, const T & value) {
		surf3Dwrite<T>(value, surface, x * sizeof(T), y, z);
	}
};