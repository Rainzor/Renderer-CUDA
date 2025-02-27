#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utils/utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param Intersection       Output the record of the intersection.
 * @return                   Whether the intersection test was successful.
 */
__device__ bool boxIntersectionTest(Ray r, float tmax,
        Intersection & intersection) {
	intersection.t = -1;
	glm::vec3 intersectionPoint;
    Ray &q = r;
	bool outside;

    float tmin = -1e38f;
    //float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }

		intersection.t = tmin;
		intersection.surfaceNormal = tmin_n;
		intersection.outside = outside;
		return true;
    }
    return false;
}

/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 1 and is centered at the origin.
 *
 * @param  r				 The ray to test.
 * @param  tmax			     The maximum distance along the ray to test.
 * @param  intersection      Output the record of the intersection.
 * @param  outside           Whether the ray came from outside the sphere.
 * @return                   Whether the intersection test was successful.
 */
__device__ bool sphereIntersectionTest(Ray r, float tmax,
        Intersection& intersection) {
	glm::vec3 intersectionPoint;
    glm::vec3 normal;
	intersection.t = -1;
    float radius = 1.f;
	bool outside;

	Ray &rt = r;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return false;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return false;
    } else if (t1 > 0 && t2 > 0) {
        t = MIN(t1, t2);
        outside = true;
    } else {
        t = MAX(t1, t2);
        outside = false;
    }

	if (t > tmax) {
		return false;
	}

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
	glm::vec3 outward_normal = glm::normalize(objspaceIntersection);
	glm::vec2 uv = glm::vec2(0.f);

	float theta = glm::acos(outward_normal.y);
	float phi = glm::atan(outward_normal.z, outward_normal.x) + PI;
	uv.x = phi / (2 * PI);
	uv.y = theta / PI;

	normal = outward_normal;
    if (!outside) {
        normal = -normal;
    }

	intersection.surfaceNormal = normal;
	intersection.uv = uv;
	intersection.t = t;
	intersection.outside = outside;
	return true;
}


/**
* Test intersection between a ray and a transformed triangle mesh.
* 
* @param  r				    The ray to test.
* @param  tmax			    The maximum distance along the ray to test.
* @param  intersection      Output the record of the intersection.
* @return                   Whether the intersection test was successful.
*/


__device__ bool trimeshIntersectionTest(Ray r, float tmax,
	Intersection& intersection, Triangle* triangles, BVHNode* bvh_nodes) {
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	intersection.t = -1;

    Ray &q = r;
	float tmin = 0;
	float t = tmax;
	glm::vec3 weight;

	BVHNode* stack[STACK_SIZE];
	BVHNode** stackPtr = stack;

	int stack_size = 0;
	float t_root_max = tmax;
	float t_root_min = tmin;
    if (!bvh_nodes[0].bbox.intersect(q, t_root_min, t_root_max)) {
		return false;
    }
	stack_size ++;
	*(++stackPtr) = &bvh_nodes[0];
	int triangle_id = -1;
	while(stack_size > 0 && stack_size < STACK_SIZE) {
		BVHNode* node = *(stackPtr--); // pop
		stack_size--;
		if(node == NULL) break;
        // Bounding box intersection check
        else {
            if (node->isLeaf()) {
                Triangle& tri = triangles[node->primId];
                glm::vec3 baryPos;
				float distance;
                // Triangle-ray intersection check
				//if (glm::intersectRayTriangle(q.origin, q.direction, tri.v0, tri.v1, tri.v2, baryPos, distance)) {
				if (glm::intersectRayTriangle(q.origin, q.direction, tri.v0, tri.v1, tri.v2, baryPos)) {
					float t_temp = baryPos.z;
                    if (t_temp < t && t_temp > 0) {
                        t = t_temp;
						triangle_id = node->primId;
                        weight = glm::vec3(1 - baryPos.x - baryPos.y, baryPos.x, baryPos.y);
                    }
                }
            }
            else {
				float tl_min = tmin;
				float tl_max = t;
				float tr_min = tmin;
				float tr_max = t;

				bool hit_left =false, hit_right =  false;
				if (node->leftId != -1)
                    hit_left = bvh_nodes[node->leftId].bbox.intersect(q, tl_min, tl_max);
				if (node->rightId != -1)
                    hit_right = bvh_nodes[node->rightId].bbox.intersect(q, tr_min, tr_max);

				if (hit_left && hit_right) {
					if (tl_min < tr_min) {
						stack_size += 2;
						*(++stackPtr) = &bvh_nodes[node->rightId];
						*(++stackPtr) = &bvh_nodes[node->leftId];
					}
					else {
						stack_size += 2;
						*(++stackPtr) = &bvh_nodes[node->leftId];
						*(++stackPtr) = &bvh_nodes[node->rightId];
					}
				}
				else if (hit_left) {
					stack_size++;
					*(++stackPtr) = &bvh_nodes[node->leftId];
				}
				else if (hit_right) {
					stack_size++;
					*(++stackPtr) = &bvh_nodes[node->rightId];
				}
            }
        }
	}

	if (t < tmax&& triangle_id >=0) {
		normal = weight.x * triangles[triangle_id].n0 +
                 weight.y * triangles[triangle_id].n1 +
                 weight.z * triangles[triangle_id].n2;
		intersection.surfaceNormal = normal;
		intersection.t = t;

		intersection.uv = weight.x * triangles[triangle_id].uv0 +
			              weight.y * triangles[triangle_id].uv1 +
			              weight.z * triangles[triangle_id].uv2;
		intersection.uv.y = 1 - intersection.uv.y;
		intersection.outside = glm::dot(q.direction, normal) < 0;
		return true;
	}
	return false;
}



__device__ bool worldIntersectionTest(
    Ray ray, float tmax,
    Intersection& intersection,
	GeomGPU* geoms,
    BVHNode* geomBVHs
    ) {
	bool outside;
    float final_t = tmax;
	Intersection test_intersection;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    bool is_intersect = false;

    // naive parse through global geoms
    bool anyhit = false;

    BVHNode* stack[STACK_SIZE];
    BVHNode** stackPtr = stack;
    *stackPtr = NULL;

    int stack_size = 0;
    float t_root_max = FLT_MAX;
    float t_root_min = 0;
    if (!geomBVHs[0].bbox.intersect(ray, t_root_min, t_root_max)) {
        intersection.t = -1.0f;
        return false;
    }

    stack_size++;
    *(++stackPtr) = &geomBVHs[0];

	while (stack_size > 0 && stack_size < STACK_SIZE) {
		BVHNode* node = *(stackPtr--);
		stack_size--;
		if (node == NULL)
			break;
		else {

			if (node->isLeaf()) {
				Ray local_ray = ray;
				GeomGPU& geom = geoms[node->primId];
				local_ray.origin = multiplyMV(geom.transform.inverseTransform, glm::vec4(local_ray.origin, 1.0f));
				local_ray.direction = glm::normalize(multiplyMV(geom.transform.inverseTransform, glm::vec4(local_ray.direction, 0.0f)));

				test_intersection.t = -1.0f;
				is_intersect = false;

				if (geom.type == Primitive::CUBE)
				{
					is_intersect = boxIntersectionTest(local_ray, final_t, test_intersection);
				}
				else if (geom.type == Primitive::SPHERE)
				{
					is_intersect = sphereIntersectionTest(local_ray, final_t, test_intersection);
				}
				else if (geom.type == Primitive::TRIANGLE) {
					is_intersect = trimeshIntersectionTest(local_ray, final_t, test_intersection, geom.dev_triangles, geom.dev_bvh_nodes);
				}

				if (is_intersect) {
					intersect_point = multiplyMV(geom.transform.transform, glm::vec4(getPointOnRay(local_ray, test_intersection.t), 1.0f));
					normal = glm::normalize(multiplyMV(geom.transform.invTranspose, glm::vec4(test_intersection.surfaceNormal, 0.0f)));
					test_intersection.t = glm::length(intersect_point - ray.origin);
					test_intersection.surfaceNormal = normal;
					test_intersection.material_id = geom.material_id;
					// Compute the minimum t from the intersection tests to determine 
					// what scene geometry object was hit first.
					if (test_intersection.t < final_t) {
						final_t = test_intersection.t;
						intersection = test_intersection;
						anyhit = true;
					}
				}

			}
			else {
				float tl_min = 0;
				float tl_max = final_t;
				float tr_min = 0;
				float tr_max = final_t;

				bool hit_left = false, hit_right = false;
				if (node->leftId != -1)
					hit_left = geomBVHs[node->leftId].bbox.intersect(ray, tl_min, tl_max);
				if (node->rightId != -1)
					hit_right = geomBVHs[node->rightId].bbox.intersect(ray, tr_min, tr_max);
				if (hit_left && hit_right) {
					if (tl_min < tr_min) {
						*(++stackPtr) = &geomBVHs[node->rightId];
						stack_size++;
						*(++stackPtr) = &geomBVHs[node->leftId];
						stack_size++;
					}
					else {
						*(++stackPtr) = &geomBVHs[node->leftId];
						stack_size++;
						*(++stackPtr) = &geomBVHs[node->rightId];
						stack_size++;
					}
				}
				else if (hit_left) {
					*(++stackPtr) = &geomBVHs[node->leftId];
					stack_size++;
				}
				else if (hit_right) {
					*(++stackPtr) = &geomBVHs[node->rightId];
					stack_size++;
				}
			}
		}
	}

	return anyhit;
}