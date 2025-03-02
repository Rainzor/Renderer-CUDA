#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "glm/glm.hpp"

#include "common.h"
#include "utils/utilities.h"
#include "utils/tiny_obj_loader.h"

using json = nlohmann::json;
using namespace std;
struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};
class Scene {
private:
	string workdir;
    std::ifstream fp_in;
    int loadMaterial(const json& materialData);
    int loadGeom(const json& geomData);
    int loadObj(const string& obj_file,const Transform& transform, bool usemtl = false);
    int loadCamera(const json& cameraData);
    void Scene::buildBVH();

	float getAera(const Geom& geom) {
		glm::vec3 scale = geom.transform.scale;
		if (geom.type == Primitive::SPHERE) {
			if (scale.x == scale.y && scale.y == scale.z)
				return 4 * PI * powf(scale.x, 2);
			float p = 1.6075f;
			float ap = powf(scale.x, p);
			float bp = powf(scale.y, p);
			float cp = powf(scale.z, p);
			return 4 * PI * powf((ap * bp + ap * cp + bp * cp) / 3, 1 / p);
		}
		else if (geom.type == Primitive::CUBE) {
			return 6 * powf(1.0f, 2) * scale.x * scale.y * scale.z;
		}
		else if (geom.type == Primitive::TRIANGLE) {
			TriangleMesh& trimesh = trimeshes[geom.trimeshId];
			float area = 0;
			for (int i = 0; i < trimesh.num; i++) {
				Triangle& tri = trimesh.triangles[i];
				glm::vec3 e1 = tri.v1 - tri.v0;
				glm::vec3 e2 = tri.v2 - tri.v0;
				glm::vec3 normal = glm::cross(e1, e2);
				area += 0.5f * glm::length(normal);
			}
			return area * scale.x * scale.y * scale.z;
		}
	}

public:
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
	BVH scene_bvh;
    std::vector<Material> materials;
    std::vector<Texture> textures;
	std::vector<TriangleMesh> trimeshes;
    std::vector<BVH> tri_bvhs;
	std::vector<Light> lights;
	Texture envmap;
	float lights_total_weight;
    RenderState state;
};
