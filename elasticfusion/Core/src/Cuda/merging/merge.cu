#include "merge.cuh"

#include "../convenience.cuh"
#include "../operators.cuh"

__global__ void mergeCopy(PtrSz<float> dst, const PtrSz<float> src, int dst_offset, int src_offset, int n, int vertex_size)
{
    for(int i = threadIdx.x; i < n; i += blockDim.x)
    {
        memcpy(&(dst[dst_offset + (i * vertex_size)]), &(src[src_offset + (i * vertex_size)]), vertex_size * sizeof(float) * 1);
    }
}
__global__ void mergePointCloudsKernel(const PtrSz<float> src_cloud_1,    
                                       const mat33 R_1,
                                       const float3 t_1,
                                       const int  cloud_1_size,
                                       const PtrSz<float> src_cloud_2,
                                       const mat33 R_2,
                                       const float3 t_2,
                                       const int cloud_2_size,
                                       PtrSz<float> dst,
                                       int vertex_size)
{
    int m = cloud_1_size, n = cloud_2_size;
    int current = 0;
    int total = n + m;
    int i = 0, j = 0, k = 0;
    int per_map_cache_size = 272;
    int i_cached = 0, j_cached = 272;

    __shared__ float cached_map[12000];
    memcpy(cached_map, src_cloud_1, vertex_size * sizeof(float) * ( cloud_1_size > 272 ? 272 : cloud_1_size));
    memcpy(&(cached_map[per_map_cache_size * vertex_size]), src_cloud_2, vertex_size * sizeof(float) * ( cloud_2_size > 272 ? 272 : cloud_2_size));

    if(threadIdx.x != 0)return;

    while(current < total)
    {
        k = i;
        while(i < m && i_cached <= per_map_cache_size && ( j >= n  || cached_map[i_cached * vertex_size + 4 + 2] <= cached_map[j_cached * vertex_size + 4 + 2]))
        {
            i++;
            i_cached++;
        }
        
        int src_vertex_offset = k * vertex_size;
        int dst_vertex_offset = current * vertex_size;
        memcpy(&dst[dst_vertex_offset], &src_cloud_1[src_vertex_offset], vertex_size * sizeof(float) * (i - k));
        //mergeCopy<<<1, 256>>>(dst, src_cloud_1, dst_vertex_offset, src_vertex_offset, i - k, vertex_size);
        //cudaDeviceSynchronize();
        current += i - k;

        if(i_cached > per_map_cache_size)
        {
            i_cached = 0;
            memcpy(cached_map, &(src_cloud_1[i * vertex_size]), vertex_size * sizeof(float) * ( cloud_1_size - i > 272 ? 272 : cloud_1_size - i));
            continue;
        }

        k = j;
        while(j < n && j_cached <= (2 * per_map_cache_size) && (i >= m || cached_map[j_cached * vertex_size + 4 + 2] <= cached_map[i_cached * vertex_size + 4 + 2]))
        {
            j++;
            j_cached++;
        }

        src_vertex_offset = k * vertex_size;
        dst_vertex_offset = current * vertex_size;
        memcpy(&dst[dst_vertex_offset], &src_cloud_2[src_vertex_offset], vertex_size * sizeof(float) * (j - k));
        //mergeCopy<<<1, 256>>>(dst, src_cloud_2, dst_vertex_offset, src_vertex_offset, j - k, vertex_size);
        //cudaDeviceSynchronize();
        current+= j - k;

        if(j_cached > 2 * per_map_cache_size)
        {
            j_cached = per_map_cache_size;
            memcpy(&(cached_map[per_map_cache_size * vertex_size]), &(src_cloud_2[j* vertex_size]), vertex_size * sizeof(float) * ( cloud_2_size - j > 272 ? 272 : cloud_2_size - j));            
        }
    }
}

__global__ void transformPointCloudKernel(const PtrSz<float> cloud,
                                          const mat33 R,
                                          const float3 t,
                                          const int size,
                                          PtrSz<float> dst,
                                          int vertex_size)
{
    int num_sensors = vertex_size - 12;

    for(int i = (blockIdx.x * blockDim.x + threadIdx.x); i < size; i += blockDim.x * gridDim.x)
    {
        int index = i * vertex_size;
        memcpy(&dst[index], &cloud[index], vertex_size * sizeof(float));
        float3 newPos = {cloud[index + 0 + 0], 
                        cloud[index + 0 + 1],
                        cloud[index + 0 + 2]};
        float3 newNorm = {cloud[index + 8 + num_sensors + 0], 
                          cloud[index + 8 + num_sensors + 1],
                          cloud[index + 8 + num_sensors + 2]};
        
        newPos = R * newPos + t;
        newNorm = R * newNorm;
        
        dst[index + 0] = newPos.x;
        dst[index + 1] = newPos.y;
        dst[index + 2] = newPos.z;
        
        dst[index + 8 + num_sensors + 0] = newNorm.x;
        dst[index + 8 + num_sensors + 1] = newNorm.y;
        dst[index + 8 + num_sensors + 2] = newNorm.z;        
    }
}

void mergePointClouds(const DeviceArray<float> & src_cloud_1,    
                      const mat33& R_1,
                      const float3& t_1,
                      const int & cloud_1_size,
                      const DeviceArray<float> & src_cloud_2,
                      const mat33& R_2,
                      const float3& t_2,
                      const int & cloud_2_size,
                      DeviceArray<float> & dst,
                      const int & vertex_size)
{
    int threads = 256;
    int blocks = 112;
    transformPointCloudKernel<<<threads, blocks>>>(src_cloud_2, R_2, t_2, cloud_2_size, src_cloud_2, vertex_size);
    cudaSafeCall(cudaDeviceSynchronize());
    mergePointCloudsKernel<<<1, 1>>>(src_cloud_1, R_1, t_1, cloud_1_size, src_cloud_2, R_2, t_2, cloud_2_size, dst, vertex_size);

    cudaSafeCall ( cudaGetLastError () );
}