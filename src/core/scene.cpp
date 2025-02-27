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
		Material default_m;
		default_m.type = MaterialType::DIFFUSE;
		default_m.diffuse = glm::vec3(1.0f);
		default_m.texture_id = -1;
		default_m.emittance = 0.0f;
		default_m.indexOfRefraction = 1.0f;
		materials.push_back(default_m);

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
}

Scene::~Scene() {
	std::cout << "Cleaning up scene..." << std::endl;
	for (int i = 0; i < bitmaps.size(); i++) {
		delete[] bitmaps[i].pixels;
	}
	for (int i = 0; i < trimeshes.size(); i++) {
		delete[] trimeshes[i].triangles;
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

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // Set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    return 1;
}

int Scene::loadBitmap(const string& bitmapPath){
	std::cout << "Loading bitmap texture from " << bitmapPath << "..." << std::endl;
	int w, h, n;
	unsigned char* data = stbi_load(bitmapPath.c_str(), &w, &h, &n, 0);
	if (data == nullptr) {
		std::cout << "Error loading bitmap texture!" << std::endl;
		return -1;
	}
	glm::u8vec4* dataCopy = new glm::u8vec4[w * h];
	std::cout << "Width: " << w << " Height: " << h << " Channels: " << n << std::endl;
	for (int i = 0; i < w * h; i++) {
		unsigned char r = data[i * n];
		unsigned char g = data[i * n + 1];
		unsigned char b = data[i * n + 2];
		unsigned char a = n == 4 ? data[i * n + 3] : 255;
		dataCopy[i] = glm::u8vec4(r, g, b, a);
	}
	Bitmap newBitmap;
	newBitmap.width = w;
	newBitmap.height = h;
	newBitmap.pixels = dataCopy;
	bitmaps.push_back(newBitmap);
	stbi_image_free(data);
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
    else if (type == "dielectric") {
        newMaterial.type = MaterialType::DIELECTRIC;
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
		if (loadBitmap(bitmap_path.string()) == 1)
            newMaterial.texture_id = bitmaps.size() - 1;
        else {
            return -1;
        }
	}
	else {
		newMaterial.texture_id = -1;
	}

    if (materialData.contains("emission")) {
        newMaterial.emittance = materialData["emission"];
    }
    if (materialData.contains("ior")) {
        newMaterial.indexOfRefraction = materialData["indexOfRefraction"];
    }

    materials.push_back(newMaterial);
    return 1;
}

int Scene::loadGeom(const json& shapeData) {
	Geom newGeom;
	// Load transformations
	if (shapeData.contains("transform")) {
		auto transform = shapeData["transform"];
		newGeom.transform.translation = glm::vec3(transform["translate"][0], transform["translate"][1], transform["translate"][2]);
		newGeom.transform.rotation = glm::vec3(transform["rotate"][0], transform["rotate"][1], transform["rotate"][2]);
		newGeom.transform.scale = glm::vec3(transform["scale"][0], transform["scale"][1], transform["scale"][2]);
	}
	else {
		// Default transformations
		newGeom.transform.translation = glm::vec3(0.0f);
		newGeom.transform.rotation = glm::vec3(0.0f);
		newGeom.transform.scale = glm::vec3(1.0f);
	}


	newGeom.transform.transform = utilityCore::buildTransformationMatrix(newGeom.transform.translation, newGeom.transform.rotation, newGeom.transform.scale);
	newGeom.transform.inverseTransform = glm::inverse(newGeom.transform.transform);
	newGeom.transform.invTranspose = glm::inverseTranspose(newGeom.transform.transform);

	// Link material (bsdf)
	if (shapeData.contains("bsdf"))
		newGeom.materialId = shapeData["bsdf"] + 1; // Offset by 1 to account for default material
	else {
		// Default material
		newGeom.materialId = 0;
	}

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

	std::cout << "Connecting Geom to Material " << newGeom.materialId << "..." << std::endl;
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

	std::cout << std::endl << "Obj file has materials size: " << to_materials.size() << std::endl;
	int materialId_offset = this->materials.size();
    for (size_t i = 0; i < to_materials.size() && usemtl; i++) {
		std::cout << std::endl << "Creating new material " << this->materials.size() << "..." << std::endl;
		Material newMaterial;
		tinyobj::material_t &mat = to_materials[i];
		newMaterial.type = MaterialType::DIFFUSE;
		newMaterial.indexOfRefraction = mat.ior;
		newMaterial.diffuse = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        if (mat.diffuse_texname != "") {
			fs::path bitmap_path = fs::path(base_dir) / mat.diffuse_texname;
			if (loadBitmap(bitmap_path.string()) == 1)
				newMaterial.texture_id = bitmaps.size() - 1;
            else {
                return -1;
            }
		}
		else {
			newMaterial.texture_id = -1;
		}
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

			Geom newGeom;
			newGeom.type = Primitive::TRIANGLE;
			newGeom.trimeshId = trimeshes.size() - 1;
			if (shape.mesh.material_ids[0] == -1)
				newGeom.materialId = 0;
			else
				newGeom.materialId = shape.mesh.material_ids[0] + materialId_offset;

			std::cout << "Connecting Geom to Material " << newGeom.materialId << "..." << std::endl;
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