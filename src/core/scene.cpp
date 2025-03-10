#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "scene.h"
#include <filesystem>

namespace fs = std::filesystem;

Scene::Scene(std::string filename) {
	this->lights_total_weight = 0;
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
	this->workdir = fs::path(filename).parent_path().string();
	std::cout << "Working directory: " << this->workdir << std::endl;

    std::string type = filename.substr(filename.find_last_of(".") + 1);

    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }

	if (strcmp(type.c_str(), "json") == 0) {
        json sceneData;
        fp_in >> sceneData;
        this->state.imageName = sceneData["name"];
        std::cout << "Scene Name: " << this->state.imageName << std::endl;
        this->state.traceDepth = sceneData["integrator"]["maxdepth"];
        if (loadCamera(sceneData["sensor"]) == -1) {
            std::cout << "Error loading camera!" << std::endl;
			throw;
        }

		if (sceneData.contains("emitter")) {
			auto envmap_path = fs::path(workdir) / string(sceneData["emitter"]["filename"]);
			if (envmap.loadf(envmap_path.string())) {
				std::cout << "Loaded environment map from " << envmap_path << std::endl;
				std::cout << "Width: " << envmap.width << " Height: " << envmap.height << std::endl;

			}
			else {
				std::cout << "Error loading environment map!" << std::endl;
				throw;
			}
		}

		Material default_m;
		default_m.type = MaterialType::DIFFUSE;
		default_m.diffuse = glm::vec3(1.0f);
		default_m.texture_id = -1;
		default_m.emittance = 0.0f;
		default_m.ior = 1.0f;
		materials.push_back(default_m);
		material_map["default"] = 0;

        for (const auto& material : sceneData["bsdf"]) {
			if (loadMaterial(material) == -1) {
				std::cout << "Error loading material!" << std::endl;
				throw;
			}
        }
        for (const auto& object : sceneData["shape"]){
			if (loadGeom(object) == -1) {
				std::cout << "Error loading object!" << std::endl;
				throw;
			}
        }

		// Build BVH
		buildBVH();
    }
    fp_in.close();
	std::cout << "Scene loaded successfully!" << std::endl;
	std::cout << "Starting rendering..." << std::endl;
}

Scene::~Scene() {
	std::cout << "Cleaning up scene..." << std::endl;
	for (int i = 0; i < textures.size(); i++) {
		delete[] textures[i].pixels;
	}
	for (int i = 0; i < trimeshes.size(); i++) {
		delete[] trimeshes[i].triangles;
	}
	if (!envmap.pixelsf) {
		delete[] envmap.pixelsf;
	}
}


int Scene::loadCamera(const json& cameraData) {
    std::cout << std::endl << "Loading Camera ..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;

    const json filmData = cameraData["film"];
    camera.resolution = glm::ivec2(filmData["resolution"][0], filmData["resolution"][1]);

    state.iterations = filmData["spp"];

    camera.position = glm::vec3(cameraData["eye"][0], cameraData["eye"][1], cameraData["eye"][2]);
	//std::cout << camera.position.x << " " << camera.position.y << " " << camera.position.z << endl;
    camera.lookAt = glm::vec3(cameraData["lookat"][0], cameraData["lookat"][1], cameraData["lookat"][2]);
    camera.up = glm::vec3(cameraData["up"][0], cameraData["up"][1], cameraData["up"][2]);

    camera.focalLength = cameraData["focal"];
    camera.aperture = cameraData["aperture"];
    
    // Calculate fov on resolution
    float fovy = cameraData["fovy"];
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);


    // Set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    return 1;
}

int Scene::loadMaterial(const json& materialData) {
std::cout << std::endl << "Creating new material " << materials.size() << "..." << std::endl;
    Material newMaterial;
    std::string type = materialData["type"];

	if (type == "light") {
		newMaterial.type = MaterialType::LIGHT;
	}
	else if (type == "diffuse") {
		newMaterial.type = MaterialType::DIFFUSE;
	}
	else if (type == "specular") {
		newMaterial.type = MaterialType::SPECULAR;
	}
	else if (type == "plastic") {
		newMaterial.type = MaterialType::PLASTIC;
	}
	else if (type == "dielectric") {
		newMaterial.type = MaterialType::DIELECTRIC;
	}
	else if (type == "conductor") {
		newMaterial.type = MaterialType::CONDUCTOR;
	}

    // Load color and other properties
    if (materialData.contains("rgb")) {
        newMaterial.diffuse = glm::vec3(materialData["rgb"][0], materialData["rgb"][1], materialData["rgb"][2]);
	}
	else {
		newMaterial.diffuse = glm::vec3(1.0f);
	}

    if (materialData.contains("bitmap")) {
        auto bitmap_path = fs::path(workdir) / string(materialData["bitmap"]);
		Texture newTexture;
		if (newTexture.load(bitmap_path.string())) {
			textures.push_back(newTexture);
			newMaterial.texture_id = textures.size() - 1;
		}
        else {
            return -1;
        }
	} else {
		newMaterial.texture_id = -1;
	}

    if (materialData.contains("emission")) {
        newMaterial.emittance = materialData["emission"];
    }
    if (materialData.contains("ior")) {
        newMaterial.ior = materialData["ior"];
    }
	if (materialData.contains("alpha")) {
		newMaterial.roughness = sqrtf(materialData["alpha"]);
	}
	if (materialData.contains("eta")) {
		newMaterial.eta = glm::vec3(materialData["eta"][0], materialData["eta"][1], materialData["eta"][2]);
	}
	if (materialData.contains("k")) {
		newMaterial.k = glm::vec3(materialData["k"][0], materialData["k"][1], materialData["k"][2]);
	}

	if (materialData.contains("id")) {
		string id = materialData["id"];
		material_map[id] = materials.size();
	}
	else {
		string id = std::to_string(materials.size());
		material_map[id] = materials.size();
	}
    materials.push_back(newMaterial);
    return 1;
}

int Scene::loadGeom(const json& shapeData) {
	Geom newGeom;
	// Load transformations
	newGeom.transform.translation = glm::vec3(0.0f);
	newGeom.transform.rotation = glm::vec3(0.0f);
	newGeom.transform.scale = glm::vec3(1.0f);
	if (shapeData.contains("transform")) {
		auto transform = shapeData["transform"];
		if (transform.contains("translate"))
			newGeom.transform.translation = glm::vec3(transform["translate"][0], transform["translate"][1], transform["translate"][2]);
		
		if (transform.contains("rotate"))
			newGeom.transform.rotation = glm::vec3(transform["rotate"][0], transform["rotate"][1], transform["rotate"][2]);

		if (transform.contains("scale"))
			newGeom.transform.scale = glm::vec3(transform["scale"][0], transform["scale"][1], transform["scale"][2]);
	}


	newGeom.transform.transform = buildTransformationMatrix(newGeom.transform.translation, newGeom.transform.rotation, newGeom.transform.scale);
	newGeom.transform.inverseTransform = glm::inverse(newGeom.transform.transform);
	newGeom.transform.invTranspose = glm::inverseTranspose(newGeom.transform.transform);

	// Link material (bsdf)


	string mat_id = shapeData.contains("bsdf") ? shapeData["bsdf"] : "default";
	newGeom.materialId = material_map[mat_id];

	auto material_type = materials[newGeom.materialId].type;

	std::string shape_type = shapeData["type"];
	if (shape_type == "sphere") {
		std::cout << std::endl << "Creating new sphere..." << std::endl;
		newGeom.type = Primitive::SPHERE;
	}
	else if (shape_type == "cube") {
		std::cout << std::endl <<"Creating new cube..." << std::endl;
		newGeom.type = Primitive::CUBE;
	}
	else if (shape_type == "obj") {
		std::cout << std::endl<< "Creating new Triangle Mesh..." << std::endl;
		bool usemtl = false;
		if (shapeData.contains("usemtl")) {
			usemtl = shapeData["usemtl"] && material_type != MaterialType::LIGHT;
		}
		newGeom.type = Primitive::TRIANGLE;
		if (shapeData.contains("filename")) {
			//string obj_file = workdir + string(shapeData["filename"]);
			fs::path obj_path= fs::path(workdir) / string(shapeData["filename"]);
			std::cout << std::endl << "Loading obj file from " << obj_path << "..." << std::endl;
			if (loadObj(obj_path.string(), newGeom.transform, usemtl) == -1) {
				return -1; 
			}
			if (usemtl) { 
				return 1; 
			}
			newGeom.trimeshId = trimeshes.size() - 1;

			if(material_type == MaterialType::LIGHT){
				// Create light
				std::cout << std::endl << "Creating new area light..." << std::endl;
				Light newLight;
				newLight.type = LightType::Area;
				newLight.triangles = trimeshes.back().triangles;
				newLight.num = trimeshes.back().num;
				newLight.area = 0;
				for(int i = 0; i < newLight.num; i++){
					newLight.area += newLight.triangles[i].getArea();
				}
				newLight.scale = newGeom.transform.scale.x * newGeom.transform.scale.y * newGeom.transform.scale.z;
				newLight.emittance = materials[newGeom.materialId].emittance;
				lights.push_back(newLight);
				lights_total_weight += newLight.area * materials[newGeom.materialId].emittance * newLight.scale;
				newGeom.lightId = lights.size() - 1;
			}
		}
		else {
			std::cout << std::endl << "No filename provided for obj shape!" << std::endl;
			return -1;
		}
	}
	else {
		std::cout << std::endl << "Unknown shape type: " << shape_type << std::endl;
		return -1;
	}

	std::cout << "Connecting Geom to Material " << mat_id << "..." << std::endl;
	geoms.push_back(newGeom);
	return 1;
}

int Scene::loadObj(const string& obj_path,const Transform& trans, bool usemtl) {
    TriangleMesh trimesh;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> to_materials;
    std::string warn, err;
	//std::string base_dir = obj_file.substr(0, obj_file.find_last_of("/"));
	std::string base_dir = fs::path(obj_path).parent_path().string();

	if (!tinyobj::LoadObj(&attrib, &shapes, &to_materials, &warn, &err, obj_path.c_str(), base_dir.c_str())) {
		std::cerr << warn << err << std::endl;
		return -1;
	}

	size_t numTriangles = 0;
	for (size_t s = 0; s < shapes.size(); s++) {
		numTriangles += shapes[s].mesh.num_face_vertices.size();
	}
	if (numTriangles == 0) {
		std::cout << std::endl << "No triangles found in obj file!" << std::endl;
		return -1;
	}

	std::cout <<"Obj file has materials size: " << to_materials.size() << std::endl;
	int materialId_offset = this->materials.size();
    for (size_t i = 0; i < to_materials.size() && usemtl; i++) {
		std::cout << std::endl << "Creating new material " << "..." << std::endl;
		Material newMaterial;
		tinyobj::material_t &mat = to_materials[i];
		newMaterial.type = MaterialType::DIFFUSE;
		newMaterial.ior = mat.ior;
		newMaterial.diffuse = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        if (mat.diffuse_texname != "") {
			fs::path bitmap_path = fs::path(base_dir) / mat.diffuse_texname;
			Texture newTexture;
			if (newTexture.load(bitmap_path.string())) {
				cout << "Loaded texture from " << bitmap_path << endl;
				cout << "Width: " << newTexture.width << " Height: " << newTexture.height << endl;
				textures.push_back(newTexture);
				newMaterial.texture_id = textures.size() - 1;
			}
			else {
				return -1;
			}
		}
		else {
			newMaterial.texture_id = -1;
		}
		material_map[mat.name] = this->materials.size();
		this->materials.push_back(newMaterial);
    }

    trimesh.num = numTriangles;
    trimesh.triangles = new Triangle[numTriangles];
	size_t triangleIndex = 0;
    for (const auto& shape : shapes) {
		size_t index_offset = 0;
		size_t tri_index_start = triangleIndex;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
			size_t fv = shape.mesh.num_face_vertices[f];
            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];
            index_offset += 3;

			Triangle& tri = trimesh.triangles[triangleIndex++];
			// Vertex
			tri.v0 = glm::vec3(attrib.vertices[3 * idx0.vertex_index + 0], attrib.vertices[3 * idx0.vertex_index + 1], attrib.vertices[3 * idx0.vertex_index + 2]);
			tri.v1 = glm::vec3(attrib.vertices[3 * idx1.vertex_index + 0], attrib.vertices[3 * idx1.vertex_index + 1], attrib.vertices[3 * idx1.vertex_index + 2]);
			tri.v2 = glm::vec3(attrib.vertices[3 * idx2.vertex_index + 0], attrib.vertices[3 * idx2.vertex_index + 1], attrib.vertices[3 * idx2.vertex_index + 2]);

			// Normals
			bool has_normals = !attrib.normals.empty();
			has_normals &= (idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0);

            if (has_normals) {
				tri.n0 = glm::vec3(attrib.normals[3 * idx0.normal_index + 0], attrib.normals[3 * idx0.normal_index + 1], attrib.normals[3 * idx0.normal_index + 2]);
				tri.n1 = glm::vec3(attrib.normals[3 * idx1.normal_index + 0], attrib.normals[3 * idx1.normal_index + 1], attrib.normals[3 * idx1.normal_index + 2]);
				tri.n2 = glm::vec3(attrib.normals[3 * idx2.normal_index + 0], attrib.normals[3 * idx2.normal_index + 1], attrib.normals[3 * idx2.normal_index + 2]);
            }
            else {
				glm::vec3 normal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
				tri.n0 = tri.n1 = tri.n2 = normal;
            }

			// UVs
			bool has_uvs = !attrib.texcoords.empty();
			has_uvs &= (idx0.texcoord_index >= 0 && idx1.texcoord_index >= 0 && idx2.texcoord_index >= 0);
            if (has_uvs) {
                tri.uv0 = glm::vec2(attrib.texcoords[2 * idx0.texcoord_index + 0], attrib.texcoords[2 * idx0.texcoord_index + 1]);
                tri.uv1 = glm::vec2(attrib.texcoords[2 * idx1.texcoord_index + 0], attrib.texcoords[2 * idx1.texcoord_index + 1]);
                tri.uv2 = glm::vec2(attrib.texcoords[2 * idx2.texcoord_index + 0], attrib.texcoords[2 * idx2.texcoord_index + 1]);
			}
            else {
                tri.uv0 = tri.uv1 = tri.uv2 = glm::vec2(0.0f);
            }
        }
    }

	if (usemtl) {
		size_t tri_index_offset = 0;
		for (const auto& shape : shapes) {

			std::cout << std::endl << "Creating new trimesh..." << std::endl;
			size_t face_size = shape.mesh.num_face_vertices.size();
			TriangleMesh new_trimesh;
			new_trimesh.num = face_size;
			new_trimesh.triangles = new Triangle[face_size];
			memcpy(new_trimesh.triangles, trimesh.triangles + tri_index_offset, face_size * sizeof(Triangle));

			trimeshes.push_back(new_trimesh);
			const auto& to_mat = to_materials[shape.mesh.material_ids[0]];
			Geom newGeom;
			newGeom.type = Primitive::TRIANGLE;
			newGeom.trimeshId = trimeshes.size() - 1;
			if (shape.mesh.material_ids[0] == -1) {
				newGeom.materialId = 0;
				std::cout << "Connecting Geom to Material default..." << std::endl;
			}
			else {
				newGeom.materialId = material_map[to_mat.name];
				std::cout << "Connecting Geom to Material " << to_mat.name << "..." << std::endl;
			}
			newGeom.transform = trans;
			geoms.push_back(newGeom);
			tri_index_offset += face_size;
		}
		delete[] trimesh.triangles;
		return 1;
	} else {
		trimeshes.push_back(trimesh);
		return 1;
	}
}

void Scene::buildBVH() {
	std::cout << "Building BVH for all geometries..." << std::endl;
	for (int i = 0; i < trimeshes.size(); i++) {
		std::cout << "Building BVH for trimesh " << i << "..." << std::endl;
		BVH bvh(trimeshes[i]);
		tri_bvhs.push_back(bvh);
	}
	std::vector<BBox> geo_bboxs;
	for (int i = 0; i < geoms.size(); i++) {
		BBox bb;
		if (geoms[i].type == Primitive::TRIANGLE) {
			bb = tri_bvhs[geoms[i].trimeshId].bvh_nodes[0].bbox;
		}
		else {
			bb = BBox(geoms[i].type);
		}
		bb.transform(geoms[i].transform.transform);
		geo_bboxs.push_back(bb);
	}
	scene_bvh = BVH(geo_bboxs);
}