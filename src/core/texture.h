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
