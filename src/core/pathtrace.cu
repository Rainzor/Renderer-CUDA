#include <cstdio>
#include <cuda.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "pathtrace.h"
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "intersections.h"
#include "bsdf.h"


#define ERRORCHECK 1
#define USE_NEE 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
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

// Random number generator
// Iter and index are used to generate a seed for the random number generator
// To make sure that each pixel has a different seed, we use the pixel index
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

// Stream Compaction Valid Path
struct is_valid{
	__host__ __device__
		bool operator()(const PathSegment& path) {
		return path.remainingBounces > 0;
	}
};

static Scene* hst_scene = NULL;
// static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static GeomGPU* dev_geoms = NULL;
static BVHNode* dev_bvh = NULL;
static Triangle** dev_trimesh_ptr = NULL;
static BVHNode** dev_tribvh_ptr = NULL;
static Light* dev_lights = NULL;

static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadowRay* dev_shadows = NULL;
static Intersection* dev_intersections = NULL;

static cudaTextureObject_t* hst_texs = NULL;
static cudaTextureObject_t* dev_texs = NULL;

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_shadows, pixelcount * sizeof(ShadowRay));

	cudaMalloc(&dev_intersections, pixelcount * sizeof(Intersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(Intersection));


	checkCUDAError("pathtraceInit");
}

void resourceInit(Scene *scene) {

	//cudaMemcpy(&lights_total_weight, &scene->lights_total_weight, sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	if (!scene->trimeshes.empty()) {
		cudaMalloc((void**)&dev_trimesh_ptr, scene->trimeshes.size() * sizeof(Triangle*));
		for (int i = 0; i < scene->trimeshes.size(); i++)
		{
			int numTriangles = scene->trimeshes[i].num;
			Triangle* host_trimesh_ptr = NULL;
			cudaMalloc(&(host_trimesh_ptr), numTriangles * sizeof(Triangle));
			cudaMemcpy(host_trimesh_ptr, scene->trimeshes[i].triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

			cudaMemcpy((void**)&(dev_trimesh_ptr[i]), &host_trimesh_ptr, sizeof(Triangle*), cudaMemcpyHostToDevice);
		}
	}
	else {
		dev_trimesh_ptr = NULL;
	}

	if (!scene->tri_bvhs.empty()){
		cudaMalloc((void**)&dev_tribvh_ptr, scene->tri_bvhs.size() * sizeof(BVHNode*));

		for (int i = 0; i < scene->tri_bvhs.size(); i++)
		{
			int numNodes = scene->tri_bvhs[i].bvh_nodes.size();
			BVHNode* host_tribvh_ptr = NULL;
			cudaMalloc((void**)&host_tribvh_ptr, numNodes * sizeof(BVHNode));
			cudaMemcpy(host_tribvh_ptr, scene->tri_bvhs[i].bvh_nodes.data(), numNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

			cudaMemcpy((void**)&(dev_tribvh_ptr[i]), &host_tribvh_ptr, sizeof(BVHNode*), cudaMemcpyHostToDevice);
		}
	} else {
		dev_tribvh_ptr = NULL;
	}
	int numNodeds = scene->scene_bvh.bvh_nodes.size();
	if (numNodeds > 0) {
		cudaMalloc(&dev_bvh, sizeof(BVHNode) * numNodeds);
		cudaMemcpy(dev_bvh, scene->scene_bvh.bvh_nodes.data(), sizeof(BVHNode) * numNodeds, cudaMemcpyHostToDevice);
	}
	else {
		dev_bvh = NULL;
	}


	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(GeomGPU));
	for (int i = 0; i < scene->geoms.size(); i++) {
		cudaMemcpy(&dev_geoms[i].type, &scene->geoms[i].type, sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(&dev_geoms[i].transform, &scene->geoms[i].transform, sizeof(Transform), cudaMemcpyHostToDevice);
		//cudaMemcpy((void**)&dev_geoms[i].dev_material, &dev_materials[scene->geoms[i].materialId], sizeof(Material*), cudaMemcpyHostToDevice);
		cudaMemcpy(&dev_geoms[i].material_id, &scene->geoms[i].materialId, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&dev_geoms[i].light_id, &scene->geoms[i].lightId, sizeof(int), cudaMemcpyHostToDevice);
		if (scene->geoms[i].type == TRIANGLE) {
			cudaMemcpy(&dev_geoms[i].dev_triangles, &dev_trimesh_ptr[scene->geoms[i].trimeshId], sizeof(Triangle*), cudaMemcpyDeviceToDevice);
			cudaMemcpy(&dev_geoms[i].dev_bvh_nodes, &dev_tribvh_ptr[scene->geoms[i].trimeshId], sizeof(BVHNode*), cudaMemcpyDeviceToDevice);
		}
		int light_id = scene->geoms[i].lightId;
		if (light_id != -1) {
			cudaMemcpy(&dev_lights[light_id], &scene->lights[light_id], sizeof(Light), cudaMemcpyHostToDevice);
			cudaMemcpy(&dev_lights[light_id].triangles, &dev_geoms[i].dev_triangles, sizeof(Triangle*), cudaMemcpyDeviceToDevice);
		}
	}

	if (!scene->bitmaps.empty()) {
		hst_texs = new cudaTextureObject_t[scene->bitmaps.size()];
		for (int i = 0; i < scene->bitmaps.size(); i++)
		{
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
			cudaArray_t curArray;
			cudaMallocArray(&curArray, &channelDesc, scene->bitmaps[i].width, scene->bitmaps[i].height);
			cudaMemcpy2DToArray(curArray, 0, 0,
				scene->bitmaps[i].pixels, scene->bitmaps[i].width * sizeof(uchar4),
				scene->bitmaps[i].width * sizeof(uchar4), scene->bitmaps[i].height, cudaMemcpyHostToDevice);

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


		cudaMalloc(&dev_texs, scene->bitmaps.size() * sizeof(cudaTextureObject_t));
		cudaMemcpy(dev_texs, hst_texs, scene->bitmaps.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
	}
	else {
		dev_texs = NULL;
		hst_texs = NULL;
	}
	// TODO: initialize any extra device memeory you need
	checkCUDAError("resourceInit");
}

void resourceFree() {

	// TODO: clean up any extra device memory you created
	int numBitmaps = hst_scene->bitmaps.size();

	if (dev_texs != NULL){
		cudaFree(dev_texs);
	}
	if (hst_texs != NULL){
		for (int i = 0; i < numBitmaps; i++)
		{
			cudaTextureObject_t texObj = hst_texs[i];

			// Get the cudaArray from the texture object
			cudaResourceDesc resDesc;
			cudaGetTextureObjectResourceDesc(&resDesc, texObj);
			cudaArray_t array = resDesc.res.array.array;

			// Destroy the texture object and free the cudaArray
			cudaDestroyTextureObject(texObj);
			cudaFreeArray(array);
		}
		delete[] hst_texs;
	}


	cudaFree(dev_geoms);

	cudaFree(dev_materials);

	if (hst_scene == NULL)
		return;

	if (dev_trimesh_ptr != NULL) {
		for (int i = 0; i < hst_scene->trimeshes.size(); i++) {
			Triangle* host_trimesh_ptr = NULL;
			cudaMemcpy(&host_trimesh_ptr, &(dev_trimesh_ptr[i]), sizeof(Triangle*), cudaMemcpyDeviceToHost);
			cudaFree(host_trimesh_ptr);
		}
		cudaFree(dev_trimesh_ptr);
	}

	if (dev_tribvh_ptr != NULL) {
		for (int i = 0; i < hst_scene->tri_bvhs.size(); i++) {
			BVHNode* host_tribvh_ptr = NULL;
			cudaMemcpy(&host_tribvh_ptr, &(dev_tribvh_ptr[i]), sizeof(BVHNode*), cudaMemcpyDeviceToHost);
			cudaFree(host_tribvh_ptr);
		}
		cudaFree(dev_tribvh_ptr);
	}

	if (dev_bvh != NULL) {
		cudaFree(dev_bvh);
	}

	checkCUDAError("resourceFree");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_intersections);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& new_path = pathSegments[index];

		new_path.path_ray.origin = cam.position;
		new_path.color = glm::vec3(0.0f, 0.0f, 0.0f);
		new_path.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

		// ! implement antialiasing by jittering the ray

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 bias = glm::vec2(u01(rng)-0.5f, u01(rng)-0.5f);
		//glm::vec2 bias = glm::vec2(0,0);
		new_path.path_ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + bias.x)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + bias.y)
		);

		// ! Physics-based depth of field
		if (cam.aperture > 0.0f)
		{
			// Generate a random point on the lens
			glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
			glm::vec2 lensPoint = cam.aperture * sample;

			// Compute the point on the focal plane
			float focalDistance = glm::abs(cam.focalLength / new_path.path_ray.direction.z);
			glm::vec3 focalPoint = new_path.path_ray.origin + focalDistance * new_path.path_ray.direction;

			// Update the ray origin
			new_path.path_ray.origin += cam.right * lensPoint.x + cam.up * lensPoint.y;
			new_path.path_ray.direction = glm::normalize(focalPoint - new_path.path_ray.origin);
		}
		new_path.from_specular = false;
		new_path.pixel_id = index;
		new_path.remainingBounces = traceDepth;
	}
}


// computeIntersections handles generating ray intersections ONLY.
__global__ void computeIntersections(
	int num_paths,
	PathSegment* pathSegments,
	GeomGPU* geoms,
	BVHNode* geomBVHs,
	Intersection* intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths)
		return;

	Intersection test_intersection;
	bool outside;
	bool anyhit = worldIntersectionTest(
		pathSegments[path_index].path_ray,
		FLT_MAX,
		test_intersection,
		geoms,
		geomBVHs);

	if (!anyhit) {
		intersections[path_index].t = -1.0f;
	}
	else {
		intersections[path_index] = test_intersection;
	}
	pathSegments[path_index].remainingBounces--;
}


/**
* Compute the color of the ray after intersection with the scene.
* It is like the "shader" function in OpenGL
*/
__global__ void shadeMaterialMIS(
	int depth,
	int iter,
	int num_paths,
	int num_lights,
	float total_lights_weight,
	Intersection* shadeableIntersections,
	PathSegment* pathSegments,
	ShadowRay* shadowRays,
	Material* materials,
	Light* lights,
	cudaTextureObject_t* texObjs
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths && pathSegments[idx].remainingBounces < 0)
		return;

	Intersection intersection = shadeableIntersections[idx];
	PathSegment& cur_path = pathSegments[idx];
	ShadowRay& shadow_ray = shadowRays[idx];
	shadow_ray.radiance_direct = glm::vec3(0);
	shadow_ray.t_max = -1.0f;

	if (intersection.t <= 0.0f) {
		cur_path.color += BACKGROUND_COLOR * cur_path.throughput;
		cur_path.remainingBounces = 0;
		return;
	}

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);

	Material* material = &materials[intersection.material_id];
	glm::vec3 material_color = material->diffuse;;

	//Texture& texture = material->texture;

	if (material->texture_id != -1)
	{
		float4 texColor = tex2D<float4>(texObjs[material->texture_id], intersection.uv.x, intersection.uv.y);
		material_color *= glm::vec3(texColor.x, texColor.y, texColor.z);
	}

	// If the material indicates that the object was a light, "light" the ray
	if (material->type == MaterialType::LIGHT) {
#if USE_NEE
		if (depth == 0 || cur_path.from_specular) {// mis weight is 1
			cur_path.color += cur_path.throughput * material->emittance;
			cur_path.remainingBounces = 0;
			shadow_ray.radiance_direct = glm::vec3(0);
			shadow_ray.t_max = -1.0f;
			return;
		}

		glm::vec3 light_point = getPointOnRay(cur_path.path_ray, intersection.t);
		glm::vec3 light_normal = intersection.surfaceNormal;
		float cosine_term = glm::abs(glm::dot(light_normal, -cur_path.path_ray.direction));
		float distance_to_light_squared = intersection.t * intersection.t;
		float light_pdf = distance_to_light_squared * material->emittance / (cosine_term * total_lights_weight);
		if (light_pdf <= 0)
			return;
		float mis_weight = powerHeuristic(cur_path.last_pdf, light_pdf);
		cur_path.color += cur_path.throughput * material->emittance * mis_weight;
		cur_path.remainingBounces = 0;
		shadow_ray.radiance_direct = glm::vec3(0);
		shadow_ray.t_max = -1.0f;
		return;
#else 
		cur_path.color += cur_path.throughput * material->emittance;
		cur_path.remainingBounces = 0;// terminate the path
#endif
	}
	
	// Russian Roulette
	if (cur_path.remainingBounces > 1 && depth > 8) {
		glm::vec3 throughput_albedo = cur_path.throughput * material_color;
		float survival_probability = MAX(throughput_albedo.x, MAX(throughput_albedo.y, throughput_albedo.z));
		if (u01(rng) > survival_probability) {
			cur_path.remainingBounces = 0;
			return;
		}
		cur_path.throughput /= survival_probability;
	}

	// At least one bounce remaining
	if(pathSegments[idx].remainingBounces > 0){
		scatterRay(cur_path, shadow_ray, intersection, material, material_color, lights, num_lights, total_lights_weight, rng);
	}
}

// ! Shadow ray intersection
__global__ void shadowIntersection(
	int num_paths,
	PathSegment* pathSegments,
	ShadowRay* shadowRays,
	GeomGPU* geoms,
	BVHNode* geomBVHs
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
		return;

	ShadowRay& shadow_ray = shadowRays[idx];
	PathSegment& cur_path = pathSegments[idx];
	if (shadow_ray.t_max > 0.0f) {
		Intersection shadow_intersection;
		bool outside;
		bool anyhit = worldIntersectionTest(
			shadow_ray.ray,
			shadow_ray.t_max,
			shadow_intersection,
			geoms,
			geomBVHs);
		if (!anyhit) {// no blocking object
			cur_path.color += shadow_ray.radiance_direct;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixel_id] += iterationPath.color;
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

void pathtrace(uchar4* pbo, int frame, int iter, GuiDataContainer* guiData) {
	/*
	* 1.  Ray Generation
	* 2.  Intersection with Scene
	* 3.  Sample and Shading (BSDF Evaluation)
	* 4.  Stream Compaction
	* ->  Go to 2 until max depth
	* 5.  Gather results
	*/

	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	//-------------------------------------------------------------------------

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * Stream compact away all of the terminated paths.
	//   * Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     Recommend just updating the ray's PathSegment in place.
	// 	 * Finally, add this iteration's results to the image. 

	// --- 1. Generating Camera Rays ---
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks: intersections info
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(Intersection));
		cudaMemset(dev_shadows, 0, pixelcount * sizeof(ShadowRay));
		// --- 2. PathSegment Intersection Stage ---
		// path tracing to get the intersections with the scene
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths,
			dev_paths,
			dev_geoms,
			dev_bvh,
			dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		// --- 3. Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.

		 //thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareIntersection());
		shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			iter,
			num_paths,
			hst_scene->lights.size(),
			hst_scene->lights_total_weight,
			dev_intersections,
			dev_paths,
			dev_shadows,
			dev_materials,
			dev_lights,
			dev_texs
			);
		
		// --- 4. Shadow Ray Intersection Stage ---
		// path tracing to get the intersections with the scene
#if USE_NEE	
		shadowIntersection << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths,
			dev_paths,
			dev_shadows,
			dev_geoms,
			dev_bvh
			);
#endif

		// --- 5. Stream Compaction Stage ---
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, is_valid());
		num_paths = dev_path_end - dev_paths;
		depth++;
		iterationComplete = num_paths == 0 || depth >= traceDepth;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// --- 5. PathSegment Final Gather Stage ---
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	//-------------------------------------------------------------------------
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
