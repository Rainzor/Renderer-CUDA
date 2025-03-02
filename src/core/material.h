#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

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
	float indexOfRefraction = 1.0f;
	float emittance = 0.0f;
	float roughness = 0.0f;
	glm::vec3 diffuse = glm::vec3(1.0f);
	glm::vec3 eta = glm::vec3(0.0f);
	glm::vec3 k = glm::vec3(0.0f);
};