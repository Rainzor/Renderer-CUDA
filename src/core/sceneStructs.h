#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include "glm/glm.hpp"
#include "ray.h"
#include "material.h"
#include "camera.h"
#include "shape.h"
#include "bvh.h"
#include "light.h"


#define BACKGROUND_COLOR (glm::vec3(0.2f))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

struct GeomGPU {
    enum Primitive type;
    Triangle* dev_triangles;
    BVHNode* dev_bvh_nodes;
	int light_id;
    size_t material_id;
    Transform transform;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray path_ray;
    glm::vec3 color;
	glm::vec3 throughput;
    int pixel_id;
    int remainingBounces;
    float last_pdf;
	bool from_specular;
};

struct ShadowRay {
	Ray ray;
	float t_max;
	glm::vec3 radiance_direct;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct Intersection {
    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
	//Material* material;
	size_t material_id;
    bool outside;
};

//struct Sample {
//    float pdf;
//    glm::vec3 BSDF;
//    Ray ray;
//};