#include "bvh.h"

#include <algorithm>

BVH::BVH(const TriangleMesh& _trimesh){
	if (_trimesh.num == 0) {
		return;
	}
	bvh_nodes.reserve(2 * _trimesh.num + 1);
	prim_indices.reserve(_trimesh.num);
	morton_codes.reserve(_trimesh.num);
	leaf_br_nodes = nullptr;
	internal_br_nodes = nullptr;
	brtree_size = _trimesh.num;

	// Calculate bounding boxes for all triangles
	BBox bb;
	for (size_t i = 0; i < _trimesh.num; i++) {
		BBox tri_bb(_trimesh.triangles[i]);
		bboxes.push_back(tri_bb);
		bb.expand(tri_bb);
		prim_indices.push_back(i);
	}
	BVHNode root(bb);
	// Calculate morton codes for all triangles
	for (BBox& bbox : bboxes) {
		glm::vec3 point = bb.offsetUnit(bbox.centroid());
		morton_codes.push_back(morton3D(point));
	}

	std::sort(prim_indices.begin(), prim_indices.end(), [&](size_t a, size_t b) {
		return morton_codes[a] < morton_codes[b];
		});

	// Build BVH
	bvh_nodes.push_back(root);
	buildBVH(bvh_nodes[0], 0, _trimesh.num);
}

BVH::BVH(const std::vector<BBox>& _geo_bboxs) {
	if (_geo_bboxs.size() == 0) {
		return;
	}
	bvh_nodes.reserve(2 * _geo_bboxs.size() + 1);
	prim_indices.reserve(_geo_bboxs.size());
	morton_codes.reserve(_geo_bboxs.size());
	leaf_br_nodes = nullptr;
	internal_br_nodes = nullptr;
	brtree_size = _geo_bboxs.size();

	this->bboxes = _geo_bboxs;
	// Calculate bounding boxes for all triangles
	BBox bb;
	for (size_t i = 0; i < bboxes.size(); i++) {
		bb.expand(bboxes[i]);
		prim_indices.push_back(i);
	}
	BVHNode root(bb);
	// Calculate morton codes for all triangles
	for (BBox& bbox : bboxes) {
		glm::vec3 point = bb.offsetUnit(bbox.centroid());
		morton_codes.push_back(morton3D(point));
	}
	std::sort(prim_indices.begin(), prim_indices.end(), [&](size_t a, size_t b) {
		return morton_codes[a] < morton_codes[b];
		});
	// Build BVH
	bvh_nodes.push_back(root);
	buildBVH(bvh_nodes[0], 0, _geo_bboxs.size());
}

BVH::~BVH() {
	if (leaf_br_nodes) {
		delete[] leaf_br_nodes;
	}
	if (internal_br_nodes) {
		delete[] internal_br_nodes;
	}
}

int count_leading_zero(unsigned int val)
{
	int count = 0;
	while (val){
		val >>= 1;
		count++;
	}
	return 32 - count;
}

int is_diff_at_bit(unsigned int val1, unsigned int val2, int n)
{
	return val1 >> (31 - n) != val2 >> (31 - n);
}

int BVH::findSplit(int start, int end) {
	unsigned int first_code = morton_codes[prim_indices[start]];
	unsigned int last_code = morton_codes[prim_indices[end - 1]];
	int common_prefix = count_leading_zero(first_code ^ last_code);
	if (common_prefix == 32) { // all the same
		return (start + end) >> 1;
	}
	int split = start;
	int step = end - start;
	do {
		step = (step + 1) >> 1;
		int new_split = split + step;
		if (new_split < end) {
			unsigned int split_code = morton_codes[prim_indices[new_split]];
			bool is_diff = is_diff_at_bit(first_code, split_code, common_prefix);
			if (!is_diff) { split = new_split; }
		}
	} while (step > 1);
	return split;
}

void BVH::buildBVH(BVHNode& node, int start, int end) {

	if (end - start == 0) {
		std::cerr << "Error: Building BVH with no primitive" << std::endl;
		throw;
	}

	if (end - start == 1) {
		// Leaf node
		node.leftId = -1;
		node.rightId = -1;
		node.primId = prim_indices[start];
		return;
	}

	if (end - start == 2) {
		// Leaf node
		BBox left_bb = bboxes[prim_indices[start]];
		BBox right_bb = bboxes[prim_indices[start + 1]];
		
		node.leftId = bvh_nodes.size();
		BVHNode left = BVHNode(left_bb);
		left.primId = prim_indices[start];
		bvh_nodes.push_back(left);

		node.rightId = bvh_nodes.size();
		BVHNode right = BVHNode(right_bb);
		right.primId = prim_indices[start + 1];
		bvh_nodes.push_back(right);
		
		return;
	}
	
	{	// Internal node
		int split = findSplit(start, end);

		BBox left_bb, right_bb;
		BVHNode left, right;
	
		for (int i = start; i <= split; i++) {
			left_bb.expand(bboxes[prim_indices[i]]);
		}
		left = BVHNode(left_bb);
		bvh_nodes.push_back(left);
		node.leftId = bvh_nodes.size() - 1;

		for (int i = split+1; i < end; i++) {
			right_bb.expand(bboxes[prim_indices[i]]);
		}
		right = BVHNode(right_bb);
		bvh_nodes.push_back(right);
		node.rightId = bvh_nodes.size() - 1;
		buildBVH(bvh_nodes[node.leftId], start, split+1);
		buildBVH(bvh_nodes[node.rightId], split+1, end);
	}
}