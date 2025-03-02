#include "texture.h"
#include <iostream>

using namespace utilityCore;
bool Texture::load(const std::string & file_name, bool gamma) {

	int w, h, n;
	unsigned char* data = stbi_load(file_name.c_str(), &w, &h, &n, 0);
	if (data == nullptr) {
		std::cout << "Error loading texture!" << std::endl;
		return false;
	}
    if(pixels != nullptr) {
        delete[] pixels;
    }
	if (gamma) {
		glm::vec4* rgba = new glm::vec4[w * h];
		for (int i = 0; i < w * h; i++) {
			float r = gamma_to_linear(data[i * n] / 255.0f);
			float g = gamma_to_linear(data[i * n + 1] / 255.0f);
			float b = gamma_to_linear(data[i * n + 2] / 255.0f);
			float a = n == 4 ? data[i * n + 3] / 255.0f : 1.0f;
			a = gamma_to_linear(a);
			rgba[i] = glm::vec4(r, g, b, a);
		}

		pixels = new glm::u8vec4[w * h];
		for (int i = 0; i < w * h; i++) {
			unsigned char r = (unsigned char)(clamp(rgba[i].r * 255.f, 0, 255.f));
			unsigned char g = (unsigned char)(clamp(rgba[i].g * 255.f, 0, 255.f));
			unsigned char b = (unsigned char)(clamp(rgba[i].b * 255.f, 0, 255.f));
			unsigned char a = (unsigned char)(clamp(rgba[i].a * 255.f, 0, 255.f));
			pixels[i] = glm::u8vec4(r, g, b, a);
		}
		delete[] rgba;
	}
	else {
		pixels = new glm::u8vec4[w * h];
		for (int i = 0; i < w * h; i++) {
			unsigned char r = data[i * n];
			unsigned char g = data[i * n + 1];
			unsigned char b = data[i * n + 2];
			unsigned char a = n == 4 ? data[i * n + 3] : 255;
			pixels[i] = glm::u8vec4(r, g, b, a);
		}
	}
	width = w;
    height = h;
	stbi_image_free(data);

	return true;
}

bool Texture::loadf(const std::string & file_name) {

	int w, h, n;
	float* data = stbi_loadf(file_name.c_str(), &w, &h, &n, STBI_rgb);
	if (data == nullptr) {
		std::cout << "Error loading texture!" << std::endl;
		return false;
	}
    if(pixelsf != nullptr) {
        delete[] pixelsf;
    }
	pixelsf = new glm::vec4[w * h];
	for (int i = 0; i < w * h; i++) {
		pixelsf[i] = glm::vec4(data[i * n], 
                        data[i * n + 1], 
                        data[i * n + 2], 
                        1.0f);
	}
	width = w;
    height = h;
	stbi_image_free(data);
	return true;
}