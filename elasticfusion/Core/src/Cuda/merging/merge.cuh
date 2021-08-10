#ifndef CUDA_MERGE_CUH_
#define CUDA_MERGE_CUH_

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#include "../containers/device_array.hpp"
#include "../types.cuh"

void mergePointClouds(const DeviceArray<float> & src_cloud_1,    
                      const mat33& R_1,
                      const float3& t_1,
                      const int & cloud_1_size,
                      const DeviceArray<float> & src_cloud_2,
                      const mat33& R_2,
                      const float3& t_2,
                      const int & cloud_2_size,
                      DeviceArray<float> & dst,
                      const int & vertex_size);

#endif //CUDA_MERGE_CUH_