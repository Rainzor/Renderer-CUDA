#include "integrator.h"

void Integrator::pathtraceInit(Scene* scene) {
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

	cudaMalloc(&dev_intersections, pixelcount * sizeof(CUDAIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(CUDAIntersection));

	//checkCUDAError("pathtraceInit");
}


void Integrator::resourceInit(Scene *scene) {
    if(scene == NULL){
        return;
    }
    if(!scene->materials.empty()){
        std::vector<CUDAMaterial> hst_materials;
        for (int i = 0; i < scene->materials.size(); i++) {
            CUDAMaterial newMaterial;
            memset(&newMaterial, 0, sizeof(CUDAMaterial)); // ³õÊ¼»¯ÇåÁã
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
                newMaterial.type = MaterialType::SPECULAR;
            }
            else if(scene->materials[i].type == MaterialType::PLASTIC){
                newMaterial.plastic.diffuse = scene->materials[i].diffuse;
				newMaterial.plastic.texture_id = scene->materials[i].texture_id;
                newMaterial.plastic.linear_roughness = scene->materials[i].roughness;
                newMaterial.type = MaterialType::PLASTIC;
            }
            else if(scene->materials[i].type == MaterialType::DIELECTRIC){
                newMaterial.dielectric.ior = scene->materials[i].indexOfRefraction;
                newMaterial.type = MaterialType::DIELECTRIC;
            }
            else if(scene->materials[i].type == MaterialType::CONDUCTOR){
                newMaterial.conductor.eta = scene->materials[i].eta;
                newMaterial.conductor.k = scene->materials[i].k;
                newMaterial.type = MaterialType::CONDUCTOR;
            }
            hst_materials.push_back(newMaterial);
        }

        cudaMalloc(&dev_materials, scene->materials.size() * sizeof(CUDAMaterial));
        cudaMemcpy(dev_materials, hst_materials.data(), scene->materials.size() * sizeof(CUDAMaterial), cudaMemcpyHostToDevice);
        hst_materials.clear();
    }

    if(!scene->geoms.empty()){
        cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(CUDAGeom));
        std::vector<CUDAGeom> hst_geoms;
        cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));

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
        hst_geoms.clear();
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
	//checkCUDAError("resourceInit");
}


void Integrator::pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_intersections);
	//checkCUDAError("pathtraceFree");
}

void Integrator::resourceFree() {
	if (hst_scene == NULL)
    return;

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

    cudaFree(dev_lights);
	cudaFree(dev_materials);

    if (dev_geoms != NULL) {
        for(int i = 0; i < hst_scene->geoms.size(); i++){
            if(hst_scene->geoms[i].type == Primitive::TRIANGLE){
				cudaFree(dev_geoms[i].dev_triangles);
				cudaFree(dev_geoms[i].dev_bvh_nodes);
            }
        }
        cudaFree(dev_geoms);
    }

	if (dev_world_bvh != NULL) {
		cudaFree(dev_world_bvh);
	}
	//checkCUDAError("resourceFree");
}