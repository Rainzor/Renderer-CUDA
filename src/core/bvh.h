#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "ray.h"
#include "shape.h"

#define STACK_SIZE 256

struct BBox {
	glm::vec3 max;
	glm::vec3 min;

	BBox() {
		max = glm::vec3(-FLT_MAX);
		min = glm::vec3(FLT_MAX);
	}

	BBox(glm::vec3 max, glm::vec3 min) {
		this->max = max;
		this->min = min;
	}

	BBox(glm::vec3 p) {
		max = p;
		min = p;
	}

	BBox(const Triangle& tri) {
		max = glm::vec3(fmaxf(fmaxf(tri.v0.x, tri.v1.x), tri.v2.x),
			fmaxf(fmaxf(tri.v0.y, tri.v1.y), tri.v2.y),
			fmaxf(fmaxf(tri.v0.z, tri.v1.z), tri.v2.z));
		min = glm::vec3(fminf(fminf(tri.v0.x, tri.v1.x), tri.v2.x),
			fminf(fminf(tri.v0.y, tri.v1.y), tri.v2.y),
			fminf(fminf(tri.v0.z, tri.v1.z), tri.v2.z));
	}

	BBox(Primitive prim) {
		if (prim == Primitive::SPHERE) {
			max = glm::vec3(1.0f);
			min = glm::vec3(-1.0f);
		}
		else if (prim == Primitive::CUBE) {
			max = glm::vec3(0.5f);
			min = glm::vec3(-0.5f);
		}
	}

	__host__ __device__
		void expand(BBox bbox) {
		max.x = fmaxf(max.x, bbox.max.x);
		max.y = fmaxf(max.y, bbox.max.y);
		max.z = fmaxf(max.z, bbox.max.z);

		min.x = fminf(min.x, bbox.min.x);
		min.y = fminf(min.y, bbox.min.y);
		min.z = fminf(min.z, bbox.min.z);
	}

	__host__ __device__
		void expand(glm::vec3 p) {
		max.x = fmaxf(max.x, p.x);
		max.y = fmaxf(max.y, p.y);
		max.z = fmaxf(max.z, p.z);
		min.x = fminf(min.x, p.x);
		min.y = fminf(min.y, p.y);
		min.z = fminf(min.z, p.z);
	}

	__host__ __device__
		glm::vec3 centroid() {
		return (max + min) * 0.5f;
	}

	__host__ __device__
		glm::vec3 offsetUnit(glm::vec3 p) {
		glm::vec3 o = p - min;
		glm::vec3 d = max - min;
		float x = d.x == 0 ? 0 : o.x / d.x;
		float y = d.y == 0 ? 0 : o.y / d.y;
		float z = d.z == 0 ? 0 : o.z / d.z;
		return glm::vec3(x, y, z);
	}

	__host__ __device__
		bool intersect(const Ray& ray, float& tmin, float& tmax) const{

		glm::vec3 inv_direction = 1.0f / ray.direction;

		float t1 = (min.x - ray.origin.x) * inv_direction.x;
		float t2 = (max.x - ray.origin.x) * inv_direction.x;
		tmin = fminf(t1, t2);
		tmax = fmaxf(t1, t2);
		float t3 = (min.y - ray.origin.y) * inv_direction.y;
		float t4 = (max.y - ray.origin.y) * inv_direction.y;
		tmin = fmaxf(tmin, fminf(t3, t4));
		tmax = fminf(tmax, fmaxf(t3, t4));
		float t5 = (min.z - ray.origin.z) * inv_direction.z;
		float t6 = (max.z - ray.origin.z) * inv_direction.z;
		tmin = fmaxf(tmin, fminf(t5, t6));
		tmax = fminf(tmax, fmaxf(t5, t6));
		return tmax >= tmin;
	}

	__host__ __device__
		void transform(glm::mat4 transform) {
		// Initialize transformed bounding box
		glm::vec3 newMax = glm::vec3(-FLT_MAX);
		glm::vec3 newMin = glm::vec3(FLT_MAX);

		// Define the corners of the bounding box
		glm::vec3 corners[8] = {
			glm::vec3(min.x, min.y, min.z),
			glm::vec3(min.x, min.y, max.z),
			glm::vec3(min.x, max.y, min.z),
			glm::vec3(min.x, max.y, max.z),
			glm::vec3(max.x, min.y, min.z),
			glm::vec3(max.x, min.y, max.z),
			glm::vec3(max.x, max.y, min.z),
			glm::vec3(max.x, max.y, max.z)
		};

		// Apply the transformation to each corner
		for (int i = 0; i < 8; ++i) {
			glm::vec3 transformed = glm::vec3(transform * glm::vec4(corners[i], 1.0f));

			// Update the transformed bounding box
			newMax.x = fmaxf(newMax.x, transformed.x);
			newMax.y = fmaxf(newMax.y, transformed.y);
			newMax.z = fmaxf(newMax.z, transformed.z);

			newMin.x = fminf(newMin.x, transformed.x);
			newMin.y = fminf(newMin.y, transformed.y);
			newMin.z = fminf(newMin.z, transformed.z);
		}

		// Set the transformed bounding box
		max = newMax;
		min = newMin;
	}

};


struct BVHNode {
	BBox bbox;
	int leftId;
	int rightId;
	int primId;

	BVHNode() {
		leftId = -1;
		rightId = -1;
		primId = -1;
	}

	BVHNode(const BBox& bbox) {
		this->bbox = bbox;
		this->leftId = -1;
		this->rightId = -1;
		this->primId = -1;
	}

	__host__ __device__
	inline bool isLeaf() const {
		return leftId == -1 && rightId == -1;
	}
};


struct BRTreeNode {
public:
	BRTreeNode() :childA(0), childB(0), parent(0), idx(0), counter(0) {}

	__host__ __device__
		inline void setChildA(int _childA, bool is_leaf)
	{
		if (is_leaf) { childA = -_childA - 1; }
		else { childA = _childA + 1; }
	}

	__host__ __device__
		inline void setChildB(int _childB, bool is_leaf)
	{
		if (is_leaf) { childB = -_childB - 1; }
		else { childB = _childB + 1; }
	}

	__host__ __device__
		inline void setParent(int _parent)
	{
		parent = _parent + 1;
	}

	__host__ __device__
		inline void setIdx(int _idx)
	{
		idx = _idx;
	}

	__host__ __device__
		inline int getChildA(bool& is_leaf, bool& is_null)
	{
		if (childA == 0) { is_null = true; return -1; } is_null = false; is_leaf = childA < 0; if (is_leaf) return -(childA + 1); else return childA - 1;
	}

	__host__ __device__
		inline int getChildB(bool& is_leaf, bool& is_null)
	{
		if (childB == 0) { is_null = true; return -1; } is_null = false; is_leaf = childB < 0; if (is_leaf) return -(childB + 1); else return childB - 1;
	}

	__host__ __device__
		inline int getParent(bool& is_null)
	{
		if (parent == 0) { is_null = true; return -1; } is_null = false; return parent - 1;
	}

	__host__ __device__
		inline int getIdx() { return idx; }
public:
	unsigned int counter;
	BBox bbox;

private:
	int childA;
	int childB;
	int parent;
	int idx;
};

struct BVH {
public:
	std::vector<BVHNode> bvh_nodes;
	std::vector<BBox> bboxes;
	std::vector<size_t> prim_indices;
	std::vector<unsigned int> morton_codes;
	BVH() { }
	BVH(const TriangleMesh& );
	BVH(const std::vector<BBox>& );
	~BVH();
	void buildBVH(BVHNode& node, int start, int end);
	int findSplit(int start, int end);
private:
	BRTreeNode* leaf_br_nodes;
	BRTreeNode* internal_br_nodes;
	int brtree_size;

	unsigned int expandBits(unsigned int v) {
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;
		return v;
	}
	unsigned int morton3D(float x, float y, float z) {
		x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
		y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
		z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
		unsigned int xx = expandBits((unsigned int)x);
		unsigned int yy = expandBits((unsigned int)y);
		unsigned int zz = expandBits((unsigned int)z);
		return xx * 4 + yy * 2 + zz;
	}
	unsigned int morton3D(glm::vec3 v) {
		return morton3D(v.x, v.y, v.z);
	}
};

struct BVHGPU {
	BVHNode* bvh_nodes;
	size_t num;
};