#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include "utils/common.h"
#include "scene.h"
#include "material.h"
#include "shape.h"
#include "bvh.h"
#include "light.h"
#include "ray.h"

struct Integrator {
	struct CUDATracer {
		Ray path_ray;
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

    struct CUDAIntersection {
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


	Scene* hst_scene;
	int screen_width;
	int screen_height;
	int max_depth;

	glm::vec3* dev_image = NULL;
	CUDATracer* dev_paths = NULL;
	CUDAShadowRay* dev_shadows = NULL;
	CUDAIntersection* dev_intersections = NULL;

	CUDAMaterial* dev_materials = NULL;
	CUDAGeom* dev_geoms = NULL;
	BVHNode* dev_world_bvh = NULL;
	Light* dev_lights = NULL;
    cudaTextureObject_t* hst_texs = NULL;
    cudaTextureObject_t* dev_texs = NULL;

    void pathtraceInit(Scene* scene);
    void pathtraceFree();

    void resourceInit(Scene* scene);
    void resourceFree();

    void render(uchar4* pbo, int frame, int iter, GuiDataContainer* guiData);
};
