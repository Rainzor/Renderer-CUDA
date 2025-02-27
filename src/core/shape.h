#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

enum Primitive: unsigned char {
    SPHERE,
    CUBE,
    TRIANGLE,
};

struct Transform {
	glm::vec3 translation = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;
};

struct Triangle{
	glm::vec3 v0, v1, v2;
	glm::vec3 n0, n1, n2;
	glm::vec2 uv0, uv1, uv2;

	__host__ __device__ glm::vec3 getNormal(float u, float v) const {
		return glm::normalize(n0 * (1 - u - v) + n1 * u + n2 * v);
	}

	__host__ __device__ glm::vec2 getUV(float u, float v) const {
		return uv0 * (1 - u - v) + uv1 * u + uv2 * v;
	}

	__host__ __device__ glm::vec3 getPosition(float u, float v) const {
		return v0 * (1 - u - v) + v1 * u + v2 * v;
	}

	__host__ __device__ float getArea() const {
		glm::vec3 e1 = v1 - v0;
		glm::vec3 e2 = v2 - v0;
		glm::vec3 normal = glm::cross(e1, e2);
		return 0.5f * glm::length(normal);
	}

};

struct TriangleMesh {
	Triangle* triangles;
	size_t num;
};

struct Geom {
    enum Primitive type = TRIANGLE;
    int trimeshId = -1;
	int lightId = -1;
    size_t materialId = 0;
    Transform transform;
};