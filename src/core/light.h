#pragma once

#include <cuda_runtime.h>
#include <thrust/random.h>

#include "shape.h"

enum LightType: unsigned char{
    Area,
};

struct Light{
    LightType type;
    Triangle* triangles;
    size_t num = 0;
    float area = 0;
	float scale = 1;
    float emittance = 1;
    
    __host__ __device__ 
    void sample(glm::vec3& pos, glm::vec3& normal, thrust::default_random_engine& rng) const{
        thrust::uniform_real_distribution<float> u01(0, 1);
        float u = u01(rng);
        float v = u01(rng);
        
        float pdf_area = u01(rng)*area;

        for(int i = 0; i < num; i++){// O(n) search
            pdf_area -= triangles[i].getArea();
            if(pdf_area <= 0){
                pos = triangles[i].getPosition(u, v);
                normal = triangles[i].getNormal(u, v);
                return;
            }
        }
    }
};
