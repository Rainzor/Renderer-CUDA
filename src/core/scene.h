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
	std::map<std::string, int> material_map;
};
