#include <cstdio>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/partition.h>

#include "utils/common.h"
#include "integrator.h"
#include "intersections.cuh"
#include "bsdf.cuh"

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

// Stream Compaction Valid Path
struct is_valid{
	__host__ __device__
		bool operator()(const CUDATracer& path) {
		return path.remainingBounces > 0;
	}
};

/**
* Generate PathSegments with rays from the camera through the screen into the scene, 

* Antialiasing - add rays for sub-pixel sampling
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, CUDATracer* trace_buffer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x > cam.resolution.x || y > cam.resolution.y) return;

	int index = x + (y * cam.resolution.x);
	CUDATracer& new_trace = trace_buffer[index];

	new_trace.ray.origin = cam.position;
	new_trace.color = glm::vec3(0.0f, 0.0f, 0.0f);
	new_trace.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

	// ! implement antialiasing by jittering the ray

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec2 bias = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
	//glm::vec2 bias = glm::vec2(0,0);
	new_trace.ray.direction = glm::normalize(cam.view
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
		float focalDistance = glm::abs(cam.focalLength / new_trace.ray.direction.z);
		glm::vec3 focalPoint = new_trace.ray.origin + focalDistance * new_trace.ray.direction;

		// Update the ray origin
		new_trace.ray.origin += cam.right * lensPoint.x + cam.up * lensPoint.y;
		new_trace.ray.direction = glm::normalize(focalPoint - new_trace.ray.origin);
	}
	new_trace.from_specular = false;
	new_trace.pixel_id = index;
	new_trace.remainingBounces = traceDepth;
}


// Ray Tracing with BVH
__global__ void computeIntersections(
	int num_paths,
	CUDATracer* trace_buffer,
	CUDAGeom* geoms,
	BVHNode* geomBVHs,
	CUDARecord* records)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths)
		return;

	CUDARecord test_records;
	bool outside;
	bool anyhit = worldIntersectionTest(
		trace_buffer[path_index].ray,
		FLT_MAX,
		test_records,
		geoms,
		geomBVHs);

	if (!anyhit) {
		records[path_index].t = -1.0f;
	}
	else {
		records[path_index] = test_records;
	}
	trace_buffer[path_index].remainingBounces--;
}


/**
* Compute the color of the ray after record with the scene.
* It is like the "shader" function in OpenGL
*/
__global__ void shadeMaterialMIS(
	int depth,
	int iter,
	int num_paths,
	int num_lights,
	float total_lights_weight,
	CUDARecord* shadeableIntersections,
	CUDATracer* trace_buffer,
	CUDAShadowRay* shadowRays,
	CUDAMaterial* materials,
	Light* lights,
	cudaTextureObject_t* texObjs
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths && trace_buffer[idx].remainingBounces < 0)
		return;

	CUDARecord record = shadeableIntersections[idx];
	CUDATracer& cur_trace = trace_buffer[idx];
	CUDAShadowRay& shadow_ray = shadowRays[idx];
	shadow_ray.radiance_direct = glm::vec3(0);
	shadow_ray.t_max = -1.0f;

	if (record.t <= 0.0f) {
		cur_trace.color += BACKGROUND_COLOR * cur_trace.throughput;
		cur_trace.remainingBounces = 0;
		return;
	}

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);

	CUDAMaterial* material = &materials[record.material_id];
	// If the material indicates that the object was a light, "light" the ray
	if (material->type == MaterialType::LIGHT) {
#if USE_NEE
		if (depth == 0 || cur_trace.from_specular) {// mis weight is 1
			cur_trace.color += cur_trace.throughput * material->light.emission;
			cur_trace.remainingBounces = 0;
			shadow_ray.radiance_direct = glm::vec3(0);
			shadow_ray.t_max = -1.0f;
			return;
		}

		glm::vec3 light_point = getPointOnRay(cur_trace.ray, record.t);
		glm::vec3 light_normal = record.surfaceNormal;
		float cosine_term = glm::abs(glm::dot(light_normal, -cur_trace.ray.direction));
		float distance_to_light_squared = record.t * record.t;
		float light_pdf = distance_to_light_squared * material->light.emission / (cosine_term * total_lights_weight);
		if (light_pdf <= 0)
			return;
		float mis_weight = powerHeuristic(cur_trace.last_pdf, light_pdf);
		cur_trace.color += cur_trace.throughput * material->light.emission * mis_weight;
		cur_trace.remainingBounces = 0;
		shadow_ray.radiance_direct = glm::vec3(0);
		shadow_ray.t_max = -1.0f;
		return;
#else 
		cur_trace.color += cur_trace.throughput * material->light.emission;
		cur_trace.remainingBounces = 0;// terminate the path
#endif
	}

	 //Russian Roulette
	if (cur_trace.remainingBounces > 1 && depth > 8) {
		glm::vec3 throughput_albedo = cur_trace.throughput;
		float survival_probability = MAX(throughput_albedo.x, MAX(throughput_albedo.y, throughput_albedo.z));
		if (u01(rng) > survival_probability) {
			cur_trace.remainingBounces = 0;
			return;
		}
		cur_trace.throughput /= survival_probability;
	}

	// At least one bounce remaining
	if (trace_buffer[idx].remainingBounces > 0) {
		scatterRay(cur_trace, shadow_ray, record, material, texObjs, lights, num_lights, total_lights_weight, rng);
	}
}


// ! Shadow ray intersection
__global__ void shadowIntersection(
	int num_paths,
	CUDATracer* trace_buffer,
	CUDAShadowRay* shadowRays,
	CUDAGeom* geoms,
	BVHNode* geomBVHs
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
		return;

	CUDAShadowRay& shadow_ray = shadowRays[idx];
	CUDATracer& cur_path = trace_buffer[idx];
	if (shadow_ray.t_max > 0.0f) {
		CUDARecord shadow_record;
		bool outside;
		bool anyhit = worldIntersectionTest(
			shadow_ray.ray,
			shadow_ray.t_max,
			shadow_record,
			geoms,
			geomBVHs);
		if (!anyhit) {// no blocking object
			cur_path.color += shadow_ray.radiance_direct;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, CUDATracer* iteration_trace_buffer)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		CUDATracer iter_trace = iteration_trace_buffer[index];
		image[iter_trace.pixel_id] += iter_trace.color;
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

void Integrator::render(uchar4* pbo, int frame, int iter, GuiDataContainer* guiData) {
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

	// --- 1. Generating Camera Rays ---
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);

	int depth = 0;
	CUDATracer* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks: intersections info
		cudaMemset(dev_records, 0, pixelcount * sizeof(Intersection));
		cudaMemset(dev_shadows, 0, pixelcount * sizeof(ShadowRay));
		// --- 2. PathSegment Intersection Stage ---
		// path tracing to get the intersections with the scene
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths,
			dev_paths,
			dev_geoms,
			dev_world_bvh,
			dev_records
			);
		//checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		// --- 3. Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.

		 //thrust::sort_by_key(thrust::device, dev_records, dev_records + num_paths, dev_paths, compareIntersection());
		shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			iter,
			num_paths,
			hst_scene->lights.size(),
			hst_scene->lights_total_weight,
			dev_records,
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
			dev_world_bvh
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

	//checkCUDAError("pathtrace");
}