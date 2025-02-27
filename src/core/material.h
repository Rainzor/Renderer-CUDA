#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

enum MaterialType: unsigned char {
    LIGHT,
    DIFFUSE,
    SPECULAR,
    DIELECTRIC,
};

enum TextureType {
    RGB,
    BITMAP,
};

struct Bitmap {
    public:
        int width;
        int height;
        glm::u8vec4* pixels;
        __host__ __device__ inline glm::vec3 getPixel(glm::vec2 uv) {
            int i = uv.x * width;
            int j = (1 - uv.y) * height;// flip y
    
            if (i >= width) i = width - 1;
            if (j >= height) j = height - 1;
    
            int index = (i + j * width);
            float r = pixels[index].r / 255.f;
            float g = pixels[index].g / 255.f;
            float b = pixels[index].b / 255.f;
            float a = pixels[index].a / 255.f;
            return glm::vec3(r, g, b) * a;
        }
    };

struct Texture {
    TextureType type = RGB;
    glm::vec3 color = glm::vec3(0.0f);
    size_t bitmapId = 0; 
};

struct Material {
    enum MaterialType type;
    Texture texture;
    float indexOfRefraction;
    float emittance; 
};