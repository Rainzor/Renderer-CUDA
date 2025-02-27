#pragma once

#include "glm/glm.hpp"

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;

    // Depth of field
    float aperture=0.0f;
    float focalLength=1.0f;
    
    // near and far plane
    float farClip = 1000.f;
    float nearClip = 0.001f;
};
