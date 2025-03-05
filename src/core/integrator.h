#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#include "common.h"
#include "scene.h"
#include "utils/utilities.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
inline void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

struct Integrator {
public:
	struct CUDATracer {
		Ray ray;
		glm::vec3 color;
		glm::vec3 throughput;
		int pixel_id;
		int remainingBounces;
		float last_pdf;
		bool from_specular;
	};

	struct CUDAShadowRay {
		Ray ray;
		float t_max;
		glm::vec3 radiance_direct;
	};

	struct CUDARecord {
		float t;
		glm::vec3 surfaceNormal;
		glm::vec2 uv;
		size_t material_id;
		bool outside;
	};

	union alignas(float4) CUDAMaterial {
		struct {
			float emission;
		} light;
		struct {
			glm::vec3 diffuse;
			int     texture_id;
		} diffuse;
		struct {
			glm::vec3 diffuse;
			int    texture_id;
		} specular;
		struct {
			glm::vec3 diffuse;
			int     texture_id;
			float   linear_roughness;
		} plastic;
		struct {
			int   medium_id;
			float ior;
			float linear_roughness;
		} dielectric;
		struct {
			glm::vec3 eta;
			float   linear_roughness;
			glm::vec3 k;
		} conductor;
		enum MaterialType type;

		CUDAMaterial() {};

		CUDAMaterial(const CUDAMaterial& other) {
			type = other.type;
			switch (type) {
			case MaterialType::LIGHT:
				light = other.light;
				break;
			case MaterialType::DIFFUSE:
				diffuse = other.diffuse;
				break;
			case MaterialType::SPECULAR:
				specular = other.specular;
				break;
			case MaterialType::PLASTIC:
				plastic = other.plastic;
				break;
			case MaterialType::DIELECTRIC:
				dielectric = other.dielectric;
				break;
			case MaterialType::CONDUCTOR:
				conductor = other.conductor;
				break;
			}
		}
	};

	struct CUDAGeom {
		enum Primitive type;
		Triangle* dev_triangles;
		BVHNode* dev_bvh_nodes;
		int light_id;
		size_t material_id;
		Transform transform;
	};

	struct LUTTexture {
		cudaArray_t array;
		cudaTextureObject_t texture;

		void init(cudaArray_t data) {
			array = data;
			//CU_TR_FILTER_MODE_LINEAR, CU_TR_ADDRESS_MODE_CLAMP
			cudaResourceDesc resDesc = {};
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = array;
			
			cudaTextureDesc texDesc = {};
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeClamp;
			texDesc.addressMode[1] = cudaAddressModeClamp;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL);
		}

		void free() {
			cudaDestroyTextureObject(texture);
			cudaFreeArray(array);
		}
	};


private:

	Scene* hst_scene;
	int screen_width;
	int screen_height;
	int max_depth;

	glm::vec3* dev_image = NULL;
	CUDATracer* dev_paths = NULL;
	CUDAShadowRay* dev_shadows = NULL;
	CUDARecord* dev_records = NULL;

	CUDAMaterial* dev_materials = NULL;
	CUDAGeom* dev_geoms = NULL;
	std::vector<CUDAGeom> hst_geoms;

	BVHNode* dev_world_bvh = NULL;
	Light* dev_lights = NULL;
	
	//CUDATexture* dev_textures = NULL;
    cudaTextureObject_t* dev_texs = NULL;

	cudaTextureObject_t dev_envmap;

	LUTTexture lut_dielectric_directional_albedo_enter;
	LUTTexture lut_dielectric_directional_albedo_leave;
	LUTTexture lut_dielectric_albedo_enter;
	LUTTexture lut_dielectric_albedo_leave;
	LUTTexture lut_conductor_directional_albedo;
	LUTTexture lut_conductor_albedo;

public:
	void lutDielectricTexInit();
	void lutDielectricTexFree();
	void lutConductorTetInit();
	void lutConductorTexFree();

	void pathtraceInit(Scene* scene) {
		if(scene == NULL){
			return;
		}
		hst_scene = scene;
		screen_width = scene->state.camera.resolution.x;
		screen_height = scene->state.camera.resolution.y;
		max_depth = scene->state.traceDepth;

		const Camera& cam = hst_scene->state.camera;
		const int pixelcount = cam.resolution.x * cam.resolution.y;

		cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
		cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

		cudaMalloc(&dev_paths, pixelcount * sizeof(CUDATracer));
		cudaMalloc(&dev_shadows, pixelcount * sizeof(CUDAShadowRay));

		cudaMalloc(&dev_records, pixelcount * sizeof(CUDARecord));
		cudaMemset(dev_records, 0, pixelcount * sizeof(CUDARecord));

		checkCUDAError("pathtraceInit");
	}

	void resourceInit(Scene *scene) {
		if(scene == NULL){
			return;
		}
		if(!scene->materials.empty()){
			std::vector<CUDAMaterial> hst_materials;
			for (int i = 0; i < scene->materials.size(); i++) {
				CUDAMaterial newMaterial;
				memset(&newMaterial, 0, sizeof(CUDAMaterial));
				if(scene->materials[i].type == MaterialType::LIGHT){
					newMaterial.light.emission = scene->materials[i].emittance;
					newMaterial.type = MaterialType::LIGHT;
				}
				else if(scene->materials[i].type == MaterialType::DIFFUSE){
					newMaterial.diffuse.diffuse = scene->materials[i].diffuse;
					newMaterial.diffuse.texture_id = scene->materials[i].texture_id;
					newMaterial.type = MaterialType::DIFFUSE;
				}
				else if(scene->materials[i].type == MaterialType::SPECULAR){
					newMaterial.specular.diffuse = scene->materials[i].diffuse;
					newMaterial.specular.texture_id = -1; 
					newMaterial.type = MaterialType::SPECULAR;
				}
				else if(scene->materials[i].type == MaterialType::PLASTIC){
					newMaterial.plastic.diffuse = scene->materials[i].diffuse;
					newMaterial.plastic.texture_id = scene->materials[i].texture_id;
					newMaterial.plastic.linear_roughness = scene->materials[i].roughness;
					newMaterial.type = MaterialType::PLASTIC;
				}
				else if(scene->materials[i].type == MaterialType::DIELECTRIC){
					newMaterial.dielectric.ior = scene->materials[i].ior;
					newMaterial.type = MaterialType::DIELECTRIC;
				}
				else if(scene->materials[i].type == MaterialType::CONDUCTOR){
					newMaterial.conductor.eta = scene->materials[i].eta;
					newMaterial.conductor.k = scene->materials[i].k;
					newMaterial.conductor.linear_roughness = scene->materials[i].roughness;
					newMaterial.type = MaterialType::CONDUCTOR;
				}
				else {
					cerr << "Unknown material type!" << endl;
					throw;
				}
				hst_materials.push_back(newMaterial);
			}

			cudaMalloc(&dev_materials, scene->materials.size() * sizeof(CUDAMaterial));
			cudaMemcpy(dev_materials, hst_materials.data(), scene->materials.size() * sizeof(CUDAMaterial), cudaMemcpyHostToDevice);
			hst_materials.clear();
		}

		if(!scene->geoms.empty()){
			cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(CUDAGeom));
			if (!scene->lights.empty())
				cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
			else {
				dev_lights = NULL;
			}

			for (int i = 0; i < scene->geoms.size(); i++) {
				CUDAGeom newGeom;
				newGeom.type = scene->geoms[i].type;
				newGeom.material_id = scene->geoms[i].materialId;
				newGeom.light_id = scene->geoms[i].lightId;
				newGeom.transform = scene->geoms[i].transform;
				int trimesh_id = scene->geoms[i].trimeshId;
				int light_id = scene->geoms[i].lightId;
				if(!scene->trimeshes.empty() && trimesh_id != -1){
					int numTriangles = scene->trimeshes[trimesh_id].num;
					cudaMalloc(&newGeom.dev_triangles, numTriangles * sizeof(Triangle));
					cudaMemcpy(newGeom.dev_triangles, scene->trimeshes[trimesh_id].triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

					int numNodes = scene->tri_bvhs[trimesh_id].bvh_nodes.size();
					cudaMalloc(&newGeom.dev_bvh_nodes, numNodes * sizeof(BVHNode));
					cudaMemcpy(newGeom.dev_bvh_nodes, scene->tri_bvhs[trimesh_id].bvh_nodes.data(), numNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
				}else{
					newGeom.dev_triangles = NULL;
					newGeom.dev_bvh_nodes = NULL;
				}

				if(light_id != -1){
					cudaMemcpy(&dev_lights[light_id], &scene->lights[light_id], sizeof(Light), cudaMemcpyHostToDevice);
					cudaMemcpy(&dev_lights[light_id].triangles, &newGeom.dev_triangles, sizeof(Triangle*), cudaMemcpyHostToDevice);
				}
				hst_geoms.push_back(newGeom);
			}
			cudaMemcpy(dev_geoms, hst_geoms.data(), scene->geoms.size() * sizeof(CUDAGeom), cudaMemcpyHostToDevice);
		}else{
			dev_geoms = NULL;
		}

		int numNodeds = scene->scene_bvh.bvh_nodes.size();
		if (numNodeds > 0) {
			cudaMalloc(&dev_world_bvh, sizeof(BVHNode) * numNodeds);
			cudaMemcpy(dev_world_bvh, scene->scene_bvh.bvh_nodes.data(), sizeof(BVHNode) * numNodeds, cudaMemcpyHostToDevice);
		}
		else {
			dev_world_bvh = NULL;
		}
		Texture &envmap = scene->envmap;
		if (envmap.pixelsf && envmap.width > 0 && envmap.height > 0) {
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
			cudaArray_t envmapArray;
			cudaMallocArray(&envmapArray, &channelDesc, envmap.width, envmap.height);
			cudaMemcpy2DToArray(envmapArray, 0, 0,
				envmap.pixelsf, envmap.width * sizeof(glm::vec4),
				envmap.width * sizeof(glm::vec4), envmap.height, cudaMemcpyHostToDevice);
			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = envmapArray;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&dev_envmap, &resDesc, &texDesc, NULL);
		}
		else {
			dev_envmap = NULL;
		}

		if (!scene->textures.empty()) {
			cudaTextureObject_t* hst_texs = new cudaTextureObject_t[scene->textures.size()];
			for (int i = 0; i < scene->textures.size(); i++)
			{
				cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
				cudaArray_t curArray;
				cudaMallocArray(&curArray, &channelDesc, scene->textures[i].width, scene->textures[i].height);
				cudaMemcpy2DToArray(curArray, 0, 0,
					scene->textures[i].pixels, scene->textures[i].width * sizeof(uchar4),
					scene->textures[i].width * sizeof(uchar4), scene->textures[i].height, cudaMemcpyHostToDevice);

				cudaResourceDesc resDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = curArray;

				cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = cudaAddressModeWrap;
				texDesc.addressMode[1] = cudaAddressModeWrap;
				texDesc.filterMode = cudaFilterModeLinear;
				texDesc.readMode = cudaReadModeNormalizedFloat;
				texDesc.normalizedCoords = 1;

				cudaCreateTextureObject(&hst_texs[i], &resDesc, &texDesc, NULL);
			}

			cudaMalloc(&dev_texs, scene->textures.size() * sizeof(cudaTextureObject_t));
			cudaMemcpy(dev_texs, hst_texs, scene->textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
			delete[] hst_texs;
		}
		else {
			dev_texs = NULL;
		}
		lutDielectricTexInit();
		lutConductorTetInit();
		checkCUDAError("resourceInit");
	}


	void pathtraceFree() {
		cudaFree(dev_image);  // no-op if dev_image is null
		cudaFree(dev_paths);
		cudaFree(dev_records);
		checkCUDAError("pathtraceFree");
	}

	void resourceFree() {
		if (hst_scene == NULL)
		return;

		lutDielectricTexFree();
		lutConductorTexFree();

		int num_texs = hst_scene->textures.size();

		if (dev_texs != NULL) {
			for (int i = 0; i < num_texs; i++)
			{
				cudaTextureObject_t texObj;
				cudaMemcpy(&texObj, &dev_texs[i], sizeof(cudaTextureObject_t), cudaMemcpyDeviceToHost);
				cudaResourceDesc resDesc;
				cudaGetTextureObjectResourceDesc(&resDesc, texObj);
				cudaArray_t array = resDesc.res.array.array;
				cudaFreeArray(array);
				cudaDestroyTextureObject(texObj);
			}
			cudaFree(dev_texs);
		}

		if (dev_envmap != NULL) {
			cudaDestroyTextureObject(dev_envmap);
		}
		if (dev_lights != NULL) {
			cudaFree(dev_lights);
		}
		cudaFree(dev_materials);

		if (dev_geoms != NULL) {
			for(int i = 0; i < hst_scene->geoms.size(); i++){
				if(hst_scene->geoms[i].type == Primitive::TRIANGLE){
					cudaFree(hst_geoms[i].dev_triangles);
					cudaFree(hst_geoms[i].dev_bvh_nodes);
				}
			}
			cudaFree(dev_geoms);
		}

		if (dev_world_bvh != NULL) {
			cudaFree(dev_world_bvh);
		}
		checkCUDAError("resourceFree");
	}

    void render(uchar4* pbo, int frame, int iter, GuiDataContainer* guiData);
};
