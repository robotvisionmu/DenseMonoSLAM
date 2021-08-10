/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at
 * <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "convenience.cuh"
#include "cudafuncs.cuh"
#include "operators.cuh"

__global__ void pyrDownGaussKernel(const PtrStepSz<unsigned short> src,
                                   PtrStepSz<unsigned short> dst,
                                   float sigma_color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows) return;

  const int D = 5;

  int center = src.ptr(2 * y)[2 * x];

  int x_mi = max(0, 2 * x - D / 2) - 2 * x;
  int y_mi = max(0, 2 * y - D / 2) - 2 * y;

  int x_ma = min(src.cols, 2 * x - D / 2 + D) - 2 * x;
  int y_ma = min(src.rows, 2 * y - D / 2 + D) - 2 * y;

  float sum = 0;
  float wall = 0;

  float weights[] = {0.375f, 0.25f, 0.0625f};

  for (int yi = y_mi; yi < y_ma; ++yi)
    for (int xi = x_mi; xi < x_ma; ++xi) {
      int val = src.ptr(2 * y + yi)[2 * x + xi];

      if (abs(val - center) < 3 * sigma_color) {
        sum += val * weights[abs(xi)] * weights[abs(yi)];
        wall += weights[abs(xi)] * weights[abs(yi)];
      }
    }

  dst.ptr(y)[x] = static_cast<int>(sum / wall);
}

void pyrDown(const DeviceArray2D<unsigned short>& src,
             DeviceArray2D<unsigned short>& dst, cudaStream_t stream) {
  dst.create(src.rows() / 2, src.cols() / 2);

  dim3 block(32, 8);
  dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

  const float sigma_color = 30;

  pyrDownGaussKernel<<<grid, block, 0, stream>>>(src, dst, sigma_color);
  cudaSafeCall(cudaGetLastError());
};

__global__ void computeVmapKernel(const PtrStepSz<unsigned short> depth,
                                  PtrStep<float> vmap, float fx_inv,
                                  float fy_inv, float cx, float cy,
                                  float depthCutoff) {
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;

  if (u < depth.cols && v < depth.rows) {
    float z = depth.ptr(v)[u] / 1000.f;  // load and convert: mm -> meters

    if (z != 0 && z < depthCutoff) {
      float vx = z * (u - cx) * fx_inv;
      float vy = z * (v - cy) * fy_inv;
      float vz = z;

      vmap.ptr(v)[u] = vx;
      vmap.ptr(v + depth.rows)[u] = vy;
      vmap.ptr(v + depth.rows * 2)[u] = vz;
    } else {
      vmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
    }
  }
}

void createVMap(const CameraModel& intr,
                const DeviceArray2D<unsigned short>& depth,
                DeviceArray2D<float>& vmap, const float depthCutoff,
                cudaStream_t stream) {
  vmap.create(depth.rows() * 3, depth.cols());

  dim3 block(32, 8);
  dim3 grid(1, 1, 1);
  grid.x = getGridDim(depth.cols(), block.x);
  grid.y = getGridDim(depth.rows(), block.y);

  float fx = intr.fx, cx = intr.cx;
  float fy = intr.fy, cy = intr.cy;

  computeVmapKernel<<<grid, block, 0, stream>>>(depth, vmap, 1.f / fx, 1.f / fy,
                                                cx, cy, depthCutoff);
  cudaSafeCall(cudaGetLastError());
}

__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap,
                                  PtrStep<float> nmap) {
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;

  if (u >= cols || v >= rows) return;

  if (u == cols - 1 || v == rows - 1) {
    nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
    return;
  }

  float3 v00, v01, v10;
  v00.x = vmap.ptr(v)[u];
  v01.x = vmap.ptr(v)[u + 1];
  v10.x = vmap.ptr(v + 1)[u];

  if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x)) {
    v00.y = vmap.ptr(v + rows)[u];
    v01.y = vmap.ptr(v + rows)[u + 1];
    v10.y = vmap.ptr(v + 1 + rows)[u];

    v00.z = vmap.ptr(v + 2 * rows)[u];
    v01.z = vmap.ptr(v + 2 * rows)[u + 1];
    v10.z = vmap.ptr(v + 1 + 2 * rows)[u];

    float3 r = normalized(cross(v01 - v00, v10 - v00));

    nmap.ptr(v)[u] = r.x;
    nmap.ptr(v + rows)[u] = r.y;
    nmap.ptr(v + 2 * rows)[u] = r.z;
  } else
    nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap,
                cudaStream_t stream) {
  nmap.create(vmap.rows(), vmap.cols());

  int rows = vmap.rows() / 3;
  int cols = vmap.cols();

  dim3 block(32, 8);
  dim3 grid(1, 1, 1);
  grid.x = getGridDim(cols, block.x);
  grid.y = getGridDim(rows, block.y);

  computeNmapKernel<<<grid, block, 0, stream>>>(rows, cols, vmap, nmap);
  cudaSafeCall(cudaGetLastError());
}

__global__ void tranformMapsKernel(int rows, int cols,
                                   const PtrStep<float> vmap_src,
                                   const PtrStep<float> nmap_src,
                                   const mat33 Rmat, const float3 tvec,
                                   PtrStepSz<float> vmap_dst,
                                   PtrStep<float> nmap_dst) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < cols && y < rows) {
    // vertexes
    float3 vsrc, vdst = make_float3(__int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff));
    vsrc.x = vmap_src.ptr(y)[x];

    if (!isnan(vsrc.x)) {
      vsrc.y = vmap_src.ptr(y + rows)[x];
      vsrc.z = vmap_src.ptr(y + 2 * rows)[x];

      vdst = Rmat * vsrc + tvec;

      vmap_dst.ptr(y + rows)[x] = vdst.y;
      vmap_dst.ptr(y + 2 * rows)[x] = vdst.z;
    }

    vmap_dst.ptr(y)[x] = vdst.x;

    // normals
    float3 nsrc, ndst = make_float3(__int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff));
    nsrc.x = nmap_src.ptr(y)[x];

    if (!isnan(nsrc.x)) {
      nsrc.y = nmap_src.ptr(y + rows)[x];
      nsrc.z = nmap_src.ptr(y + 2 * rows)[x];

      ndst = Rmat * nsrc;

      nmap_dst.ptr(y + rows)[x] = ndst.y;
      nmap_dst.ptr(y + 2 * rows)[x] = ndst.z;
    }

    nmap_dst.ptr(y)[x] = ndst.x;
  }
}

__global__ void tranformMapsKernel(int rows, int cols,
                                   const PtrStep<float> vmap_src,
                                   const mat33 Rmat, const float3 tvec,
                                   PtrStepSz<float> vmap_dst) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < cols && y < rows) {
    // vertexes
    float3 vsrc, vdst = make_float3(__int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff));
    vsrc.x = vmap_src.ptr(y)[x];

    if (!isnan(vsrc.x)) {
      vsrc.y = vmap_src.ptr(y + rows)[x];
      vsrc.z = vmap_src.ptr(y + 2 * rows)[x];

      vdst = Rmat * vsrc + tvec;

      vmap_dst.ptr(y + rows)[x] = vdst.y;
      vmap_dst.ptr(y + 2 * rows)[x] = vdst.z;
    }

    vmap_dst.ptr(y)[x] = vdst.x;
  }
}

void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src, const mat33& Rmat,
                  const float3& tvec, DeviceArray2D<float>& vmap_dst,
                  DeviceArray2D<float>& nmap_dst) {
  int cols = vmap_src.cols();
  int rows = vmap_src.rows() / 3;

  vmap_dst.create(rows * 3, cols);
  nmap_dst.create(rows * 3, cols);

  dim3 block(32, 8);
  dim3 grid(1, 1, 1);
  grid.x = getGridDim(cols, block.x);
  grid.y = getGridDim(rows, block.y);

  tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat,
                                      tvec, vmap_dst, nmap_dst);
  cudaSafeCall(cudaGetLastError());
}

void tranformMaps(const DeviceArray2D<float>& vmap_src, const mat33& Rmat,
                  const float3& tvec, DeviceArray2D<float>& vmap_dst) {
  int cols = vmap_src.cols();
  int rows = vmap_src.rows() / 3;

  vmap_dst.create(rows * 3, cols);

  dim3 block(32, 8);
  dim3 grid(1, 1, 1);
  grid.x = getGridDim(cols, block.x);
  grid.y = getGridDim(rows, block.y);

  tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, Rmat, tvec,
                                      vmap_dst);
  cudaSafeCall(cudaGetLastError());
}

__global__ void copyMapsKernel(int rows, int cols, const float* vmap_src,
                               const float* nmap_src, PtrStepSz<float> vmap_dst,
                               PtrStep<float> nmap_dst) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < cols && y < rows) {
    // vertexes
    float3 vsrc, vdst = make_float3(__int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff));

    vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];
    vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];
    vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];

    if (!(vsrc.z == 0)) {
      vdst = vsrc;
    }

    vmap_dst.ptr(y)[x] = vdst.x;
    vmap_dst.ptr(y + rows)[x] = vdst.y;
    vmap_dst.ptr(y + 2 * rows)[x] = vdst.z;

    // normals
    float3 nsrc, ndst = make_float3(__int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff));

    nsrc.x = nmap_src[y * cols * 4 + (x * 4) + 0];
    nsrc.y = nmap_src[y * cols * 4 + (x * 4) + 1];
    nsrc.z = nmap_src[y * cols * 4 + (x * 4) + 2];

    if (!(vsrc.z == 0)) {
      ndst = nsrc;
    }

    nmap_dst.ptr(y)[x] = ndst.x;
    nmap_dst.ptr(y + rows)[x] = ndst.y;
    nmap_dst.ptr(y + 2 * rows)[x] = ndst.z;
  }
}
__global__ void copyMapsKernel(int rows, int cols, const float* vmap_src,
                               PtrStepSz<float> vmap_dst) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < cols && y < rows) {
    // vertexes
    float3 vsrc, vdst = make_float3(__int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff),
                                    __int_as_float(0x7fffffff));

    vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];
    vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];
    vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];

    if (!(vsrc.z == 0)) {
      vdst = vsrc;
    }

    vmap_dst.ptr(y)[x] = vdst.x;
    vmap_dst.ptr(y + rows)[x] = vdst.y;
    vmap_dst.ptr(y + 2 * rows)[x] = vdst.z;
  }
}

void copyMaps(const DeviceArray<float>& vmap_src,
              DeviceArray2D<float>& vmap_dst) {
  int cols = vmap_dst.cols();
  int rows = vmap_dst.rows() / 3;

  vmap_dst.create(rows * 3, cols);

  dim3 block(32, 8);
  dim3 grid(1, 1, 1);
  grid.x = getGridDim(cols, block.x);
  grid.y = getGridDim(rows, block.y);

  copyMapsKernel<<<grid, block>>>(rows, cols, vmap_src, vmap_dst);
  cudaSafeCall(cudaGetLastError());
}

void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst,
              cudaStream_t stream) {
  int cols = vmap_dst.cols();
  int rows = vmap_dst.rows() / 3;

  vmap_dst.create(rows * 3, cols);
  nmap_dst.create(rows * 3, cols);

  dim3 block(32, 8);
  dim3 grid(1, 1, 1);
  grid.x = getGridDim(cols, block.x);
  grid.y = getGridDim(rows, block.y);

  copyMapsKernel<<<grid, block, 0, stream>>>(rows, cols, vmap_src, nmap_src,
                                             vmap_dst, nmap_dst);
  cudaSafeCall(cudaGetLastError());
}

__global__ void pyrDownKernelGaussF(const PtrStepSz<float> src,
                                    PtrStepSz<float> dst, float* gaussKernel) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows) return;

  const int D = 5;

  float center = src.ptr(2 * y)[2 * x];

  int tx = min(2 * x - D / 2 + D, src.cols - 1);
  int ty = min(2 * y - D / 2 + D, src.rows - 1);
  int cy = max(0, 2 * y - D / 2);

  float sum = 0;
  int count = 0;

  for (; cy < ty; ++cy) {
    for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
      if (!isnan(src.ptr(cy)[cx])) {
        sum += src.ptr(cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
        count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
      }
    }
  }
  dst.ptr(y)[x] = (float)(sum / (float)count);
}

template <bool normalize>
__global__ void resizeMapKernel(int drows, int dcols, int srows,
                                const PtrStep<float> input,
                                PtrStep<float> output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= dcols || y >= drows) return;

  const float qnan = __int_as_float(0x7fffffff);

  int xs = x * 2;
  int ys = y * 2;

  float x00 = input.ptr(ys + 0)[xs + 0];
  float x01 = input.ptr(ys + 0)[xs + 1];
  float x10 = input.ptr(ys + 1)[xs + 0];
  float x11 = input.ptr(ys + 1)[xs + 1];

  if (isnan(x00) || isnan(x01) || isnan(x10) || isnan(x11)) {
    output.ptr(y)[x] = qnan;
    return;
  } else {
    float3 n;

    n.x = (x00 + x01 + x10 + x11) / 4;

    float y00 = input.ptr(ys + srows + 0)[xs + 0];
    float y01 = input.ptr(ys + srows + 0)[xs + 1];
    float y10 = input.ptr(ys + srows + 1)[xs + 0];
    float y11 = input.ptr(ys + srows + 1)[xs + 1];

    n.y = (y00 + y01 + y10 + y11) / 4;

    float z00 = input.ptr(ys + 2 * srows + 0)[xs + 0];
    float z01 = input.ptr(ys + 2 * srows + 0)[xs + 1];
    float z10 = input.ptr(ys + 2 * srows + 1)[xs + 0];
    float z11 = input.ptr(ys + 2 * srows + 1)[xs + 1];

    n.z = (z00 + z01 + z10 + z11) / 4;

    if (normalize) n = normalized(n);

    output.ptr(y)[x] = n.x;
    output.ptr(y + drows)[x] = n.y;
    output.ptr(y + 2 * drows)[x] = n.z;
  }
}

template <bool normalize>
void resizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output,
               cudaStream_t stream) {
  int in_cols = input.cols();
  int in_rows = input.rows() / 3;

  int out_cols = in_cols / 2;
  int out_rows = in_rows / 2;

  output.create(out_rows * 3, out_cols);

  dim3 block(32, 8);
  dim3 grid(getGridDim(out_cols, block.x), getGridDim(out_rows, block.y));
  resizeMapKernel<normalize><<<grid, block, 0, stream>>>(
      out_rows, out_cols, in_rows, input, output);
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());
}

void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output,
                cudaStream_t stream) {
  resizeMap<false>(input, output, stream);
}

void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output,
                cudaStream_t stream) {
  resizeMap<true>(input, output, stream);
}

void pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst) {
  dst.create(src.rows() / 2, src.cols() / 2);

  dim3 block(32, 8);
  dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

  const float gaussKernel[25] = {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
                                 24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1};

  float* gauss_cuda;

  cudaMalloc((void**)&gauss_cuda, sizeof(float) * 25);
  cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25,
             cudaMemcpyHostToDevice);

  pyrDownKernelGaussF<<<grid, block>>>(src, dst, gauss_cuda);
  cudaSafeCall(cudaGetLastError());

  cudaFree(gauss_cuda);
};

__global__ void pyrDownKernelIntensityGauss(const PtrStepSz<unsigned char> src,
                                            PtrStepSz<unsigned char> dst,
                                            float* gaussKernel) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows) return;

  const int D = 5;

  int center = src.ptr(2 * y)[2 * x];

  int tx = min(2 * x - D / 2 + D, src.cols - 1);
  int ty = min(2 * y - D / 2 + D, src.rows - 1);
  int cy = max(0, 2 * y - D / 2);

  float sum = 0;
  int count = 0;

  for (; cy < ty; ++cy)
    for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
      // This might not be right, but it stops incomplete model images from
      // making up colors
      if (src.ptr(cy)[cx] > 0) {
        sum += src.ptr(cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
        count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
      }
    }
  dst.ptr(y)[x] = (sum / (float)count);
}

void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src,
                       DeviceArray2D<unsigned char>& dst, cudaStream_t stream) {
  dst.create(src.rows() / 2, src.cols() / 2);

  dim3 block(32, 8);
  dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

  const float gaussKernel[25] = {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
                                 24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1};

  float* gauss_cuda;

  cudaMalloc((void**)&gauss_cuda, sizeof(float) * 25);
  cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25,
             cudaMemcpyHostToDevice);

  pyrDownKernelIntensityGauss<<<grid, block, 0, stream>>>(src, dst, gauss_cuda);
  cudaSafeCall(cudaGetLastError());

  cudaFree(gauss_cuda);
};

__global__ void verticesToDepthKernel(const float* vmap_src,
                                      PtrStepSz<float> dst, float cutOff) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows) return;

  float z = vmap_src[y * dst.cols * 4 + (x * 4) + 2];

  dst.ptr(y)[x] =
      z > cutOff || z <= 0 ? __int_as_float(0x7fffffff) /*CUDART_NAN_F*/ : z;
}

void verticesToDepth(DeviceArray<float>& vmap_src, DeviceArray2D<float>& dst,
                     float cutOff) {
  dim3 block(32, 8);
  dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

  verticesToDepthKernel<<<grid, block>>>(vmap_src, dst, cutOff);
  cudaSafeCall(cudaGetLastError());
};

__global__ void verticesToDepth2DKernel(const PtrStepSz<float> vmap_src,
  PtrStepSz<float> dst, float cutOff) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x >= dst.cols || y >= dst.rows) return;

float z = vmap_src.ptr(y + dst.rows * 2)[x];

dst.ptr(y)[x] =
z > cutOff || z <= 0 ? __int_as_float(0x7fffffff) /*CUDART_NAN_F*/ : z;
}

void verticesToDepth(DeviceArray2D<float>& vmap_src, DeviceArray2D<float>& dst,
float cutOff) {
dim3 block(32, 8);
dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

verticesToDepth2DKernel<<<grid, block>>>(vmap_src, dst, cutOff);
cudaSafeCall(cudaGetLastError());
};

texture<uchar4, 2, cudaReadModeElementType> inTex;

__global__ void bgr2IntensityKernel(PtrStepSz<unsigned char> dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows) return;

  uchar4 src = tex2D(inTex, x, y);

  int value =
      (float)src.x * 0.114f + (float)src.y * 0.299f + (float)src.z * 0.587f;

  dst.ptr(y)[x] = value;
}

void imageBGRToIntensity(cudaArray* cuArr, DeviceArray2D<unsigned char>& dst,
                         cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

  cudaSafeCall(cudaBindTextureToArray(inTex, cuArr));

  bgr2IntensityKernel<<<grid, block, 0, stream>>>(dst);

  cudaSafeCall(cudaGetLastError());

  cudaSafeCall(cudaUnbindTexture(inTex));
};

__constant__ float gsobel_x3x3[9];
__constant__ float gsobel_y3x3[9];

__global__ void applyKernel(const PtrStepSz<unsigned char> src,
                            PtrStep<short> dx, PtrStep<short> dy) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= src.cols || y >= src.rows) return;

  float dxVal = 0;
  float dyVal = 0;

  int kernelIndex = 8;
  for (int j = max(y - 1, 0); j <= min(y + 1, src.rows - 1); j++) {
    for (int i = max(x - 1, 0); i <= min(x + 1, src.cols - 1); i++) {
      dxVal += (float)src.ptr(j)[i] * gsobel_x3x3[kernelIndex];
      dyVal += (float)src.ptr(j)[i] * gsobel_y3x3[kernelIndex];
      --kernelIndex;
    }
  }

  dx.ptr(y)[x] = dxVal;
  dy.ptr(y)[x] = dyVal;
}

void computeDerivativeImages(DeviceArray2D<unsigned char>& src,
                             DeviceArray2D<short>& dx,
                             DeviceArray2D<short>& dy) {
  static bool once = false;

  if (!once) {
    float gsx3x3[9] = {0.52201,  0.00000, -0.52201, 0.79451, -0.00000,
                       -0.79451, 0.52201, 0.00000,  -0.52201};

    float gsy3x3[9] = {0.52201, 0.79451,  0.52201,  0.00000, 0.00000,
                       0.00000, -0.52201, -0.79451, -0.52201};

    cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
    cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    once = true;
  }

  dim3 block(32, 8);
  dim3 grid(getGridDim(src.cols(), block.x), getGridDim(src.rows(), block.y));

  applyKernel<<<grid, block>>>(src, dx, dy);

  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void projectPointsKernel(const PtrStepSz<float> depth,
                                    PtrStepSz<float3> cloud, const float invFx,
                                    const float invFy, const float cx,
                                    const float cy) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= depth.cols || y >= depth.rows) return;

  float z = depth.ptr(y)[x];

  cloud.ptr(y)[x].x = (float)((x - cx) * z * invFx);
  cloud.ptr(y)[x].y = (float)((y - cy) * z * invFy);
  cloud.ptr(y)[x].z = z;
}

void projectToPointCloud(const DeviceArray2D<float>& depth,
                         const DeviceArray2D<float3>& cloud,
                         CameraModel& intrinsics, const int& level) {
  dim3 block(32, 8);
  dim3 grid(getGridDim(depth.cols(), block.x),
            getGridDim(depth.rows(), block.y));

  CameraModel intrinsicsLevel = intrinsics(level);

  projectPointsKernel<<<grid, block>>>(depth, cloud, 1.0f / intrinsicsLevel.fx,
                                       1.0f / intrinsicsLevel.fy,
                                       intrinsicsLevel.cx, intrinsicsLevel.cy);
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());
}

// __global__ void mergeCopy(PtrSz<float> dst, const PtrSz<float> src, int
// dst_offset, int src_offset, int n, int vertex_size)
// {
//     for(int i = threadIdx.x; i < n; i += blockDim.x)
//     {
//         memcpy(&(dst[dst_offset + (i * vertex_size)]), &(src[src_offset + (i
//         * vertex_size)]), vertex_size * sizeof(float) * 1);
//     }
// }
// __global__ void mergePointCloudsKernel(const PtrSz<float> src_cloud_1,
//                                        const mat33 R_1,
//                                        const float3 t_1,
//                                        const int  cloud_1_size,
//                                        const PtrSz<float> src_cloud_2,
//                                        const mat33 R_2,
//                                        const float3 t_2,
//                                        const int cloud_2_size,
//                                        PtrSz<float> dst,
//                                        int vertex_size)
// {
//     int m = cloud_1_size, n = cloud_2_size;
//     int current = 0;
//     int total = n + m;
//     int i = 0, j = 0, k = 0;
//     int per_map_cache_size = 272;
//     int i_cached = 0, j_cached = 272;

//     __shared__ float cached_map[12000];
//     memcpy(cached_map, src_cloud_1, vertex_size * sizeof(float) * (
//     cloud_1_size > 272 ? 272 : cloud_1_size));
//     memcpy(&(cached_map[per_map_cache_size * vertex_size]), src_cloud_2,
//     vertex_size * sizeof(float) * ( cloud_2_size > 272 ? 272 :
//     cloud_2_size));

//     if(threadIdx.x != 0)return;

//     while(current < total)
//     {
//         k = i;
//         while(i < m && i_cached <= per_map_cache_size && ( j >= n  ||
//         cached_map[i_cached * vertex_size + 4 + 2] <= cached_map[j_cached *
//         vertex_size + 4 + 2]))
//         {
//             i++;
//             i_cached++;
//         }

//         int src_vertex_offset = k * vertex_size;
//         int dst_vertex_offset = current * vertex_size;
//         //memcpy(&dst[dst_vertex_offset], &src_cloud_1[src_vertex_offset],
//         vertex_size * sizeof(float) * (i - k));
//         mergeCopy<<<1, 256>>>(dst, src_cloud_1, dst_vertex_offset,
//         src_vertex_offset, i - k, vertex_size);
//         cudaDeviceSynchronize();
//         current += i - k;

//         if(i_cached > per_map_cache_size)
//         {
//             i_cached = 0;
//             memcpy(cached_map, &(src_cloud_1[i * vertex_size]), vertex_size *
//             sizeof(float) * ( cloud_1_size - i > 272 ? 272 : cloud_1_size -
//             i));
//             continue;
//         }

//         k = j;
//         while(j < n && j_cached <= (2 * per_map_cache_size) && (i >= m ||
//         cached_map[j_cached * vertex_size + 4 + 2] <= cached_map[i_cached *
//         vertex_size + 4 + 2]))
//         {
//             j++;
//             j_cached++;
//         }

//         src_vertex_offset = k * vertex_size;
//         dst_vertex_offset = current * vertex_size;
//         //memcpy(&dst[dst_vertex_offset], &src_cloud_2[src_vertex_offset],
//         vertex_size * sizeof(float) * (j - k));
//         mergeCopy<<<1, 256>>>(dst, src_cloud_2, dst_vertex_offset,
//         src_vertex_offset, j - k, vertex_size);
//         current+= j - k;

//         if(j_cached > 2 * per_map_cache_size)
//         {
//             j_cached = per_map_cache_size;
//             memcpy(&(cached_map[per_map_cache_size * vertex_size]),
//             &(src_cloud_2[j* vertex_size]), vertex_size * sizeof(float) * (
//             cloud_2_size - j > 272 ? 272 : cloud_2_size - j));
//         }
//     }
// }
// __global__ void transformPointCloudKernel(const PtrSz<float> cloud,
//                                           const mat33 R,
//                                           const float3 t,
//                                           const int size,
//                                           PtrSz<float> dst,
//                                           int vertex_size)
// {
//     int num_sensors = vertex_size - 12;

//     for(int i = (blockIdx.x * blockDim.x + threadIdx.x); i < size; i +=
//     blockDim.x * gridDim.x)
//     {
//         int index = i * vertex_size;
//         memcpy(&dst[index], &cloud[index], vertex_size * sizeof(float));
//         float3 newPos = {cloud[index + 0 + 0],
//                         cloud[index + 0 + 1],
//                         cloud[index + 0 + 2]};
//         float3 newNorm = {cloud[index + 8 + num_sensors + 0],
//                           cloud[index + 8 + num_sensors + 1],
//                           cloud[index + 8 + num_sensors + 2]};

//         newPos = R * newPos + t;
//         newNorm = R * newNorm;

//         dst[index + 0] = newPos.x;
//         dst[index + 1] = newPos.y;
//         dst[index + 2] = newPos.z;

//         dst[index + 8 + num_sensors + 0] = newNorm.x;
//         dst[index + 8 + num_sensors + 1] = newNorm.y;
//         dst[index + 8 + num_sensors + 2] = newNorm.z;
//     }
// }

// void mergePointClouds(const DeviceArray<float> & src_cloud_1,
//                       const mat33& R_1,
//                       const float3& t_1,
//                       const int & cloud_1_size,
//                       const DeviceArray<float> & src_cloud_2,
//                       const mat33& R_2,
//                       const float3& t_2,
//                       const int & cloud_2_size,
//                       DeviceArray<float> & dst,
//                       const int & vertex_size)
// {
//     int threads = 256;
//     int blocks = 112;
//     transformPointCloudKernel<<<threads, blocks>>>(src_cloud_2, R_2, t_2,
//     cloud_2_size, src_cloud_2, vertex_size);
//     cudaSafeCall(cudaDeviceSynchronize());
//     mergePointCloudsKernel<<<1, 1>>>(src_cloud_1, R_1, t_1, cloud_1_size,
//     src_cloud_2, R_2, t_2, cloud_2_size, dst, vertex_size);

//     cudaSafeCall ( cudaGetLastError () );
// }

__device__ __forceinline__ int bin_idx(unsigned char& intensity,
                                       const int& num_bins) {
  int b_w = 256 / num_bins;
  return intensity / b_w;
  // return floor(((float)intensity / 255.0) * num_bins);
}

__device__ __forceinline__ int bin_idx(float& depth, const float& max_depth,
                                       const int& num_bins) {
  int b_w = max_depth / num_bins;
  return depth / b_w;
  // return floor(((float)intensity / 255.0) * num_bins);
}

// __global__ void computeJointProbabilityHistogramKernelBSpline(
//     const PtrStepSz<float> vmap_kf, const PtrStepSz<unsigned char> img_kf,
//     const mat33 R_kf, const float3 t_kf,
//     const PtrStepSz<unsigned char> img_curr, const mat33 R_curr_inverse,
//     const float3 t_curr, float* histogram, const int cols, const int rows,
//     const int neighbourhood_size, const int num_bins,
//     const CameraModel intrinsics, int* num_points) {
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;

//   if (x < 0 || y < 0 || x >= cols || y >= rows) return;

//   float3 p_kf;
//   p_kf.x = (vmap_kf.ptr(y)[x]);  // use __ldg()?
//   p_kf.y = (vmap_kf.ptr(y + rows)[x]);
//   p_kf.z = (vmap_kf.ptr(y + 2 * rows)[x]);
//   float3 p_global = p_kf;  // R_kf * p_kf + t_kf;
//   float3 p_curr = R_curr_inverse * (p_global - t_curr);

//   int x_cam_kf =
//       __float2int_rn((p_kf.x * intrinsics.fx / p_kf.z) + intrinsics.cx);
//   int y_cam_kf =
//       __float2int_rn((p_kf.y * intrinsics.fy / p_kf.z) + intrinsics.cy);

//   int x_cam_curr =
//       __float2int_rn((p_curr.x * intrinsics.fx / p_curr.z) + intrinsics.cx);
//   int y_cam_curr =
//       __float2int_rn((p_curr.y * intrinsics.fy / p_curr.z) + intrinsics.cy);

//   if (x_cam_curr < 0 || y_cam_curr < 0 || x_cam_curr >= cols ||
//       y_cam_curr >= rows || x_cam_kf < 0 || y_cam_kf < 0 || x_cam_kf >= cols
//       ||
//       y_cam_kf >= rows ||
//       // img_kf.ptr(y_cam_kf)[x_cam_kf] == 0 ||
//       p_curr.z < 0)
//     return;  //

//   unsigned char a = img_kf.ptr(y_cam_kf)[x_cam_kf];
//   int bin_a = bin_idx(a, num_bins);  // the histogram column

//   int tl_x =
//       x_cam_curr - (neighbourhood_size /
//                     2);  // < 0 ? 0 : x_cam_curr - (neighbourhood_size / 2);
//   int tl_y =
//       y_cam_curr - (neighbourhood_size /
//                     2);  // < 0 ? 0 : y_cam_curr - (neighbourhood_size / 2);
//   int br_x = x_cam_curr +
//              (neighbourhood_size /
//               2);  // > cols ? cols : x_cam_curr + (neighbourhood_size / 2);
//   int br_y = y_cam_curr +
//              (neighbourhood_size /
//               2);  // > rows ? rows : y_cam_curr + (neighbourhood_size / 2);

//   const float2 coord_grid = make_float2(x_cam_curr - 0.5f, y_cam_curr -
//   0.5f);
//   float2 index = floor(coord_grid);
//   const float2 fraction = coord_grid - index;
//   index.x += 0.5f;  // move from [-0.5, extent-0.5] to [0, extent]
//   index.y += 0.5f;

//   for (int i = -1 /*tl_y*/; i < 2.5 /*br_y*/; i++) {
//     float w_y = bspline(i - fraction.y /*y_cam_curr - i*/);
//     for (int j = -1 /*tl_x*/; j < 2.5 /* br_x*/; j++) {
//       unsigned char b =
//           (i + index.y < 0 || i + index.y > rows || j + index.x < 0 ||
//            j + index.x > cols)
//               ? 0
//               : img_curr.ptr((int)(i + index.y))[(int)(j + index.x)];
//       int bin_b = bin_idx(b, num_bins);  // histogram row
//       float w = bspline(j - fraction.x /*x_cam_curr - j*/) * w_y;
//       atomicAdd(&(histogram[(bin_b * num_bins) + bin_a]),
//                 w);  // does this work across SMs? Could need to do a
//                 reduction?
//     }
//   }
//   // atomicAdd(num_points, 16);
//   atomicAdd(num_points, 1);
// }

__global__ void computeJointProbabilityHistogramKernelImgSmem(
    const PtrStepSz<unsigned char> img_kf,
    const PtrStepSz<unsigned char> img_kf_old, const PtrStepSz<float> dmap_kf,
    const PtrStepSz<float> dmap_kf_old, const PtrStepSz<unsigned char> img_curr,
    float* histogram, const int cols, const int rows, const int num_bins,
    const int num_parts, int* num_points) {
  int startX = blockIdx.x * blockDim.x + threadIdx.x;
  int startY = blockIdx.y * blockDim.y + threadIdx.y;

  if (startX < 0 || startX >= cols || startY < 0 || startY >= rows) return;

  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  // threads in workgroup
  int tx = threadIdx.x;  //+ threadIdx.y * blockDim.x; // thread index in
                         // workgroup, linear in 0..nt-1
  int ty = threadIdx.y;
  int ntx = blockDim.x;  // total threads in workgroup
  int nty = blockDim.y;

  // group index in 0..ngroups-1
  int g = blockIdx.x + blockIdx.y * gridDim.x;

  // initialize smem
  extern __shared__ unsigned int smem[];  //[num_bins * num_bins];
  for (int i = tx + ty * blockDim.x; i < num_bins * num_bins; i += ntx * nty)
    smem[i] = 0;

  // int num_points_local = 0;
  __syncthreads();

  for (int x = startX; x < cols; x += nx) {
    for (int y = startY; y < rows; y += ny) {
      unsigned char a = 0;

      if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff)) &&
          (dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
        if (dmap_kf.ptr(y)[x] <= dmap_kf_old.ptr(y)[x])
          a = img_kf.ptr(y)[x];
        else
          a = img_kf_old.ptr(y)[x];

      } else if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff))) {
        a = img_kf.ptr(y)[x];
      } else if ((dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
        a = img_kf_old.ptr(y)[x];
      } else {
        a = 0;
      }

      int bin_a = bin_idx(a, num_bins);  // the histogram column

      unsigned char b =
          img_curr.ptr(y)[x];  // img_curr.ptr(y_cam_curr)[x_cam_curr];
      int bin_b = bin_idx(b, num_bins);  // histogram row

      // if(b < 0 || b > 255 ||
      //    a < 0 || a > 255) return;
      atomicAdd(&(smem[(bin_b * num_bins) + bin_a]),
                1);  // does this work across SMs? Could need to do a reduction?
    }
  }

  __syncthreads();

  histogram += g * num_parts;

  for (int i = tx; i < num_bins; i += ntx) {
    for (int j = ty; j < num_bins; j += nty) {
      histogram[(j * num_bins) + i] = smem[(j * num_bins) + i];
    }
  }
}

__global__ void computeJointProbabilityHistogramKernelSmemAccum(
    float* partial_histogram, const int n, const int num_parts,
    const int num_bins, float* histogram) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= num_bins || y >= num_bins) return;  // out of range
  unsigned int total = 0;
  for (int j = 0; j < n; j++)
    total += partial_histogram[((num_bins * y) + x) + num_bins * num_bins * j];
  histogram[(num_bins * y) + x] = total;
}

__global__ void computeJointProbabilityHistogramKernelImg(
    const PtrStepSz<unsigned char> img_kf,
    const PtrStepSz<unsigned char> img_kf_old, const PtrStepSz<float> dmap_kf,
    const PtrStepSz<float> dmap_kf_old, const PtrStepSz<unsigned char> img_curr,
    float* histogram, const int cols, const int rows, const int num_bins,
    int* num_points) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || y < 0 || x >= cols || y >= rows) return;

  unsigned char a = 0;

  if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff)) &&
      (dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
    if (dmap_kf.ptr(y)[x] <= dmap_kf_old.ptr(y)[x])
      a = img_kf.ptr(y)[x];
    else
      a = img_kf_old.ptr(y)[x];

  } else if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff))) {
    a = img_kf.ptr(y)[x];
  } else if ((dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
    a = img_kf_old.ptr(y)[x];
  } else {
    a = 0;
  }

  int bin_a = bin_idx(a, num_bins);  // the histogram column

  unsigned char b =
      img_curr.ptr(y)[x];            // img_curr.ptr(y_cam_curr)[x_cam_curr];
  int bin_b = bin_idx(b, num_bins);  // histogram row

  atomicAdd(&(histogram[(bin_b * num_bins) + bin_a]),
            1);  // does this work across SMs? Could need to do a reduction?
}

__global__ void computeJointProbabilityHistogramKernelDepth(
    PtrStepSz<float> dmap_kf, PtrStepSz<float> dmap_kf_old,
    PtrStepSz<float> dmap_curr, float* histogram, int cols, int rows,
    int num_bins, float max_depth, int* num_points) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || y < 0 || x >= cols || y >= rows) return;

  float a = 0.0f;
  if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff)) &&
      (dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
    if (dmap_kf.ptr(y)[x] <= dmap_kf_old.ptr(y)[x])
      a = dmap_kf.ptr(y)[x] * 1000.0;
    else
      a = dmap_kf_old.ptr(y)[x] * 1000.0;

  } else if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff))) {
    a = dmap_kf.ptr(y)[x] * 1000.0f;
  } else if ((dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
    a = dmap_kf_old.ptr(y)[x] * 1000.0f;
  } else {
    a = 0.0f;
  }

  int bin_a = bin_idx(a, max_depth, num_bins);  // the histogram column

  float b =
      dmap_curr.ptr(y)[x] * 1000.0f;  // img_curr.ptr(y_cam_curr)[x_cam_curr];
  int bin_b = bin_idx(b, max_depth, num_bins);  // histogram row

  atomicAdd(&(histogram[(bin_b * num_bins) + bin_a]),
            1);  // does this work across SMs? Could need to do a reduction?
}

__global__ void computeJointProbabilityHistogramKernelDepthSMem(
    PtrStepSz<float> dmap_kf, PtrStepSz<float> dmap_kf_old,
    PtrStepSz<float> dmap_curr, float* histogram, int cols, int rows,
    int num_bins, float max_depth, int* num_points, int num_parts) {
  int startX = blockIdx.x * blockDim.x + threadIdx.x;
  int startY = blockIdx.y * blockDim.y + threadIdx.y;

  if (startX < 0 || startY < 0 || startX >= cols || startY >= rows) return;

  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  // threads in workgroup
  int tx = threadIdx.x;  //+ threadIdx.y * blockDim.x; // thread index in
                         // workgroup, linear in 0..nt-1
  int ty = threadIdx.y;
  int ntx = blockDim.x;  // total threads in workgroup
  int nty = blockDim.y;

  // group index in 0..ngroups-1
  int g = blockIdx.x + blockIdx.y * gridDim.x;

  // initialize smem
  extern __shared__ unsigned int smem[];  //[num_bins * num_bins];
  for (int i = tx + ty * blockDim.x; i < num_bins * num_bins; i += ntx * nty)
    smem[i] = 0;

  // int num_points_local = 0;
  __syncthreads();

  for (int x = startX; x < cols; x += nx) {
    for (int y = startY; y < rows; y += ny) {
      float a = 0.0f;
      if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff)) &&
          (dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
        if (dmap_kf.ptr(y)[x] <= dmap_kf_old.ptr(y)[x])
          a = dmap_kf.ptr(y)[x] * 1000.0;
        else
          a = dmap_kf_old.ptr(y)[x] * 1000.0;

      } else if ((dmap_kf.ptr(y)[x] != __int_as_float(0x7fffffff))) {
        a = dmap_kf.ptr(y)[x] * 1000.0f;
      } else if ((dmap_kf_old.ptr(y)[x] != __int_as_float(0x7fffffff))) {
        a = dmap_kf_old.ptr(y)[x] * 1000.0f;
      } else {
        a = 0.0f;
      }

      int bin_a = bin_idx(a, max_depth, num_bins);  // the histogram column

      float b = dmap_curr.ptr(y)[x] *
                1000.0f;  // img_curr.ptr(y_cam_curr)[x_cam_curr];
      int bin_b = bin_idx(b, max_depth, num_bins);  // histogram row

      atomicAdd(&(smem[(bin_b * num_bins) + bin_a]),
                1);  // does this work across SMs? Could need to do a reduction?
    }
  }

  __syncthreads();

  histogram += g * num_parts;

  for (int i = tx; i < num_bins; i += ntx) {
    for (int j = ty; j < num_bins; j += nty) {
      histogram[(j * num_bins) + i] = smem[(j * num_bins) + i];
    }
  }
}

// void computeMutualInfo(const DeviceArray2D<float>& vmap_kf,
//                        const DeviceArray2D<unsigned char>& img_kf,
//                        const mat33& R_kf, const float3& t_kf,
//                        const DeviceArray2D<unsigned char>& img_curr,
//                        const mat33& R_curr_inverse, const float3& t_curr,
//                        float& mutual_information, const int& num_bins,
//                        const CameraModel& intrinsics) {
//   float kf_marginal_entropy = 0.0f, cf_marginal_entropy = 0.0f,
//         joint_entropy = 0.0f;

//   dim3 block(32, 8);
//   dim3 grid(80, 15);
//   int cols = vmap_kf.cols();
//   int rows = vmap_kf.rows() / 3;
//   const int neighbourhood_size = 4;

//   float* histogram_device;
//   float* histogram_host = new float[num_bins * num_bins];
//   memset(histogram_host, 0.0f, sizeof(float) * num_bins * num_bins);
//   cudaMalloc(&(histogram_device), num_bins * num_bins * sizeof(float));
//   cudaMemcpy(histogram_device, histogram_host,
//              num_bins * num_bins * sizeof(float), cudaMemcpyHostToDevice);

//   int num_points_host = 0;
//   int* num_points_device;
//   cudaMalloc(&(num_points_device), sizeof(int));
//   cudaMemcpy(num_points_device, &num_points_host, sizeof(int),
//              cudaMemcpyHostToDevice);

//   computeJointProbabilityHistogramKernelBSpline<<<grid, block>>>(
//       vmap_kf, img_kf, R_kf, t_kf, img_curr, R_curr_inverse, t_curr,
//       histogram_device, cols, rows, neighbourhood_size, num_bins, intrinsics,
//       num_points_device);

//   cudaMemcpy(histogram_host, histogram_device,
//              num_bins * num_bins * sizeof(float), cudaMemcpyDeviceToHost);
//   cudaMemcpy(&num_points_host, num_points_device, sizeof(int),
//              cudaMemcpyDeviceToHost);

//   if (num_points_host == 0) {
//     mutual_information = 0.0f;
//     return;
//   }

//   float sum = 0.0f;
//   for (int a = 0; a < num_bins; a++) {
//     for (int b = 0; b < num_bins; b++) {
//       sum += histogram_host[(a * num_bins) + b];  ///=
//       (float)(num_points_host);
//     }
//   }

//   for (int a = 0; a < num_bins; a++) {
//     for (int b = 0; b < num_bins; b++) {
//       histogram_host[(a * num_bins) + b] /=
//           (float)(num_points_host);  // sum;//(float)(num_points_host);
//       std::cout << histogram_host[(a * num_bins) + b] << " ";
//     }
//     std::cout << "\n";
//   }

//   float* P_A = new float[num_bins];
//   float* P_B = new float[num_bins];

//   for (int b = 0; b < num_bins; b++)  // column-wise marginals
//   {
//     float pb = 0.0f;
//     for (int a = 0; a < num_bins; a++) {
//       pb += histogram_host[(num_bins * a) + b];
//     }
//     P_B[b] = pb;
//   }

//   for (int a = 0; a < num_bins; a++)  // row-wise marginals
//   {
//     float pa = 0.0f;
//     for (int b = 0; b < num_bins; b++) {
//       pa += histogram_host[(num_bins * a) + b];
//     }
//     P_A[a] = pa;
//   }

//   for (int a = 0; a < num_bins; a++) {
//     for (int b = 0; b < num_bins; b++) {
//       joint_entropy += histogram_host[a * num_bins + b] *
//                        (histogram_host[a * num_bins + b] == 0.0f
//                             ? 0.0f
//                             : log2(histogram_host[a * num_bins + b]));
//     }
//   }

//   for (int b = 0; b < num_bins; b++) {
//     // float p_b = 0.0f;
//     // for(int a = 0; a < num_bins; a++)
//     // {
//     //     p_b += histogram_host[a*num_bins + b];
//     // }
//     kf_marginal_entropy += P_B[b] * (P_B[b] == 0.0f ? 0.0f : log2(P_B[b]));
//   }

//   for (int a = 0; a < num_bins; a++) {
//     // float p_a = 0.0f;
//     // for(int a = 0; a < num_bins; a++)
//     // {
//     //     p_a += histogram_host[a*num_bins + b];
//     // }
//     cf_marginal_entropy += P_A[a] * (P_A[a] == 0.0f ? 0.0f : log2(P_A[a]));
//   }

//   cf_marginal_entropy = -cf_marginal_entropy;
//   kf_marginal_entropy = -kf_marginal_entropy;
//   joint_entropy = -joint_entropy;

//   mutual_information =
//       kf_marginal_entropy + cf_marginal_entropy - joint_entropy;
//   std::cout << "num points: " << num_points_host << std::endl;
//   // std::cout << "sum: " << sum << "\n";
//   std::cout << "joint entropy: " << joint_entropy << "\n";
//   std::cout << "kf marginal entropy: " << kf_marginal_entropy << "\n";
//   std::cout << "cf marginal entropy: " << cf_marginal_entropy << "\n";
//   std::cout << "mutual info: " << mutual_information << "\n";

//   delete[] histogram_host;
//   delete[] P_A;
//   delete[] P_B;
//   cudaFree(histogram_device);
//   cudaDeviceSynchronize();
// }

void computeNIDImgSmem(const DeviceArray2D<unsigned char>& img_kf,
                       const DeviceArray2D<unsigned char>& img_kf_old,
                       const DeviceArray2D<float>& dmap_kf,
                       const DeviceArray2D<float>& dmap_kf_old,
                       const DeviceArray2D<unsigned char>& img_curr, float& nid,
                       const int& num_bins, bool save) {
  float kf_marginal_entropy = 0.0f, cf_marginal_entropy = 0.0f,
        joint_entropy = 0.0f;

  dim3 block(32, 8);
  dim3 grid(16, 16);

  int cols = img_kf.cols();
  int rows = img_kf.rows();

  float* histogram_device;
  float* histogram_host = new float[num_bins * num_bins];
  memset(histogram_host, 0.0f, sizeof(float) * num_bins * num_bins);
  cudaMalloc(&histogram_device, num_bins * num_bins * sizeof(float));
  cudaMemcpy(histogram_device, histogram_host,
             num_bins * num_bins * sizeof(float), cudaMemcpyHostToDevice);

  int num_points_host = 0;
  int* num_points_device;
  cudaMalloc(&num_points_device, sizeof(int));
  cudaMemcpy(num_points_device, &num_points_host, sizeof(int),
             cudaMemcpyHostToDevice);

  float* histogram_device_partials;
  cudaMalloc(&histogram_device_partials,
             num_bins * num_bins * (grid.x * grid.y) * sizeof(float));
  cudaMemset(histogram_device_partials, 0.0,
             num_bins * num_bins * (grid.x * grid.y) * sizeof(float));

  const int num_parts = num_bins * num_bins;  // I think num parts means the
                                              // number of elements in the
                                              // histogram??
  computeJointProbabilityHistogramKernelImgSmem<<<
      grid, block, num_bins * num_bins * sizeof(float)>>>(
      img_kf, img_kf_old, dmap_kf, dmap_kf_old, img_curr,
      histogram_device_partials, cols, rows, num_bins, num_parts,
      num_points_device);

  dim3 block2(32, 8);
  dim3 grid2(getGridDim(num_bins, block2.x), getGridDim(num_bins, block2.y));
  computeJointProbabilityHistogramKernelSmemAccum<<<grid2, block2>>>(
      histogram_device_partials, grid.x * grid.y, num_parts, num_bins,
      histogram_device);

  cudaDeviceSynchronize();

  cudaMemcpy(histogram_host, histogram_device,
             num_bins * num_bins * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&num_points_host, num_points_device, sizeof(int),
             cudaMemcpyDeviceToHost);

  num_points_host = cols * rows;

  if (num_points_host == 0) {
    nid = 1.0f;
    return;
  }

  int sum = 0;
  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      sum += histogram_host[(a * num_bins) + b];
    }
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      histogram_host[(a * num_bins) + b] /= (float)(num_points_host);
    }
  }

  float* P_A = new float[num_bins];
  float* P_B = new float[num_bins];

  for (int b = 0; b < num_bins; b++)  // column-wise marginals
  {
    float pb = 0.0f;
    for (int a = 0; a < num_bins; a++) {
      pb += histogram_host[(num_bins * a) + b];
    }
    P_B[b] = pb;
  }

  for (int a = 0; a < num_bins; a++)  // row-wise marginals
  {
    float pa = 0.0f;
    for (int b = 0; b < num_bins; b++) {
      pa += histogram_host[(num_bins * a) + b];
    }
    P_A[a] = pa;
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      joint_entropy += histogram_host[(a * num_bins) + b] *
                       (histogram_host[(a * num_bins) + b] == 0.0f
                            ? 0.0f
                            : log2(histogram_host[(a * num_bins) + b]));
    }
  }

  for (int b = 0; b < num_bins; b++) {
    kf_marginal_entropy += P_B[b] * (P_B[b] == 0.0f ? 0.0f : log2(P_B[b]));
  }

  for (int a = 0; a < num_bins; a++) {
    cf_marginal_entropy += P_A[a] * (P_A[a] == 0.0f ? 0.0f : log2(P_A[a]));
  }

  cf_marginal_entropy = -cf_marginal_entropy;
  kf_marginal_entropy = -kf_marginal_entropy;
  joint_entropy = -joint_entropy;

  float mutual_information =
      kf_marginal_entropy + cf_marginal_entropy - joint_entropy;

  nid = (joint_entropy - mutual_information) / joint_entropy;
  // std::cout << "---------------nid-rgb----------------\n";
  // std::cout << "cols: " << cols << "\n";
  // std::cout << "rows: " << rows << "\n";
  // std::cout << "num points: " << num_points_host <<"\n";
  // std::cout << "sum: " << sum << "\n";
  // std::cout << "joint entropy: " << joint_entropy << "\n";
  // std::cout << "kf marginal entropy: " << kf_marginal_entropy << "\n";
  // std::cout << "cf marginal entropy: " << cf_marginal_entropy << "\n";
  // std::cout << "mutual info: " << mutual_information << "\n";
  // std::cout << "nid: " << nid << "\n";

  if (save) {
    std::ofstream ofs("histogram_array_img.txt");
    if (ofs.is_open()) {
      for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
          ofs << histogram_host[(i * num_bins) + j];
          if (j != num_bins - 1) ofs << ",";
        }
        if (i != num_bins - 1) ofs << "\n";
      }
      ofs.close();
    }
  }
  delete[] histogram_host;
  delete[] P_A;
  delete[] P_B;
  cudaFree(histogram_device);
  cudaFree(num_points_device);
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());
}

void computeNIDImg(const DeviceArray2D<unsigned char>& img_kf,
                   const DeviceArray2D<unsigned char>& img_kf_old,
                   const DeviceArray2D<float>& dmap_kf,
                   const DeviceArray2D<float>& dmap_kf_old,
                   const DeviceArray2D<unsigned char>& img_curr, float& nid,
                   const int& num_bins, bool save) {
  float kf_marginal_entropy = 0.0f, cf_marginal_entropy = 0.0f,
        joint_entropy = 0.0f;

  dim3 block(32, 8);
  dim3 grid(getGridDim(img_kf.cols(), block.x),
            getGridDim(img_kf.rows(), block.y));

  int cols = img_kf.cols();
  int rows = img_kf.rows();

  float* histogram_device;
  float* histogram_host = new float[num_bins * num_bins];
  memset(histogram_host, 0.0f, sizeof(float) * num_bins * num_bins);
  cudaMalloc(&histogram_device, num_bins * num_bins * sizeof(float));
  cudaMemcpy(histogram_device, histogram_host,
             num_bins * num_bins * sizeof(float), cudaMemcpyHostToDevice);

  int num_points_host = 0;
  int* num_points_device;
  cudaMalloc(&num_points_device, sizeof(int));
  cudaMemcpy(num_points_device, &num_points_host, sizeof(int),
             cudaMemcpyHostToDevice);

  computeJointProbabilityHistogramKernelImg<<<grid, block>>>(
      img_kf, img_kf_old, dmap_kf, dmap_kf_old, img_curr, histogram_device,
      cols, rows, num_bins, num_points_device);

  cudaDeviceSynchronize();

  cudaMemcpy(histogram_host, histogram_device,
             num_bins * num_bins * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&num_points_host, num_points_device, sizeof(int),
             cudaMemcpyDeviceToHost);

  num_points_host = cols * rows;

  if (num_points_host == 0) {
    nid = 1.0f;
    return;
  }

  int sum = 0;
  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      sum += histogram_host[(a * num_bins) + b];
    }
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      histogram_host[(a * num_bins) + b] /= (float)(num_points_host);
    }
  }

  float* P_A = new float[num_bins];
  float* P_B = new float[num_bins];

  for (int b = 0; b < num_bins; b++)  // column-wise marginals
  {
    float pb = 0.0f;
    for (int a = 0; a < num_bins; a++) {
      pb += histogram_host[(num_bins * a) + b];
    }
    P_B[b] = pb;
  }

  for (int a = 0; a < num_bins; a++)  // row-wise marginals
  {
    float pa = 0.0f;
    for (int b = 0; b < num_bins; b++) {
      pa += histogram_host[(num_bins * a) + b];
    }
    P_A[a] = pa;
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      joint_entropy += histogram_host[(a * num_bins) + b] *
                       (histogram_host[(a * num_bins) + b] == 0.0f
                            ? 0.0f
                            : log2(histogram_host[(a * num_bins) + b]));
    }
  }

  for (int b = 0; b < num_bins; b++) {
    kf_marginal_entropy += P_B[b] * (P_B[b] == 0.0f ? 0.0f : log2(P_B[b]));
  }

  for (int a = 0; a < num_bins; a++) {
    cf_marginal_entropy += P_A[a] * (P_A[a] == 0.0f ? 0.0f : log2(P_A[a]));
  }

  cf_marginal_entropy = -cf_marginal_entropy;
  kf_marginal_entropy = -kf_marginal_entropy;
  joint_entropy = -joint_entropy;

  float mutual_information =
      kf_marginal_entropy + cf_marginal_entropy - joint_entropy;

  nid = (joint_entropy - mutual_information) / joint_entropy;
  // std::cout << "---------------nid-rgb----------------\n";
  // std::cout << "cols: " << cols << "\n";
  // std::cout << "rows: " << rows << "\n";
  // std::cout << "num points: " << num_points_host <<"\n";
  // std::cout << "sum: " << sum << "\n";
  // std::cout << "joint entropy: " << joint_entropy << "\n";
  // std::cout << "kf marginal entropy: " << kf_marginal_entropy << "\n";
  // std::cout << "cf marginal entropy: " << cf_marginal_entropy << "\n";
  // std::cout << "mutual info: " << mutual_information << "\n";
  // std::cout << "nid: " << nid << "\n";

  if (save) {
    std::ofstream ofs("histogram_array_img.txt");
    if (ofs.is_open()) {
      for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
          ofs << histogram_host[(i * num_bins) + j];
          if (j != num_bins - 1) ofs << ",";
        }
        if (i != num_bins - 1) ofs << "\n";
      }
      ofs.close();
    }
  }
  delete[] histogram_host;
  delete[] P_A;
  delete[] P_B;
  cudaFree(histogram_device);
  cudaFree(num_points_device);
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());
}

void computeNIDDepthSmem(const DeviceArray2D<float>& dmap_kf,
                         const DeviceArray2D<float>& dmap_kf_old,
                         const DeviceArray2D<float>& dmap_curr, float& nid,
                         const int& num_bins, const float& max_depth,
                         bool save) {
  float kf_marginal_entropy = 0.0f, cf_marginal_entropy = 0.0f,
        joint_entropy = 0.0f;

  dim3 block(32, 8);
  dim3 grid(16, 16);

  int cols = dmap_kf.cols();
  int rows = dmap_kf.rows();  /// 3;

  float* histogram_device;
  float* histogram_host = new float[num_bins * num_bins];
  memset(histogram_host, 0.0f, sizeof(float) * num_bins * num_bins);
  cudaMalloc(&histogram_device, num_bins * num_bins * sizeof(float));
  cudaMemcpy(histogram_device, histogram_host,
             num_bins * num_bins * sizeof(float), cudaMemcpyHostToDevice);

  int num_points_host = 0;
  int* num_points_device;
  cudaMalloc(&num_points_device, sizeof(int));
  cudaMemcpy(num_points_device, &num_points_host, sizeof(int),
             cudaMemcpyHostToDevice);

  float* histogram_device_partials;
  cudaMalloc(&histogram_device_partials,
             num_bins * num_bins * (grid.x * grid.y) * sizeof(float));
  cudaMemset(histogram_device_partials, 0.0,
             num_bins * num_bins * (grid.x * grid.y) * sizeof(float));

  const int num_parts = num_bins * num_bins;  // I think num parts means the
                                              // number of elements in the
                                              // histogram??

  computeJointProbabilityHistogramKernelDepthSMem<<<
      grid, block, num_parts * sizeof(float)>>>(
      dmap_kf, dmap_kf_old, dmap_curr, histogram_device, cols, rows, num_bins,
      max_depth, num_points_device, num_parts);

  dim3 block2(32, 8);
  dim3 grid2(getGridDim(num_bins, block2.x), getGridDim(num_bins, block2.y));
  computeJointProbabilityHistogramKernelSmemAccum<<<grid2, block2>>>(
      histogram_device_partials, grid.x * grid.y, num_parts, num_bins,
      histogram_device);

  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());

  cudaMemcpy(histogram_host, histogram_device,
             num_bins * num_bins * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&num_points_host, num_points_device, sizeof(int),
             cudaMemcpyDeviceToHost);

  num_points_host = cols * rows;
  if (num_points_host == 0) {
    nid = 1.0f;  // 0.0f;
    return;
  }

  int sum = 0;
  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      sum += histogram_host[(a * num_bins) + b];
    }
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      histogram_host[(a * num_bins) + b] /= (float)(num_points_host);
    }
  }

  float* P_A = new float[num_bins];
  float* P_B = new float[num_bins];

  for (int b = 0; b < num_bins; b++)  // column-wise marginals
  {
    float pb = 0.0f;
    for (int a = 0; a < num_bins; a++) {
      pb += histogram_host[(num_bins * a) + b];
    }
    P_B[b] = pb;
  }

  for (int a = 0; a < num_bins; a++)  // row-wise marginals
  {
    float pa = 0.0f;
    for (int b = 0; b < num_bins; b++) {
      pa += histogram_host[(num_bins * a) + b];
    }
    P_A[a] = pa;
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      joint_entropy += histogram_host[(a * num_bins) + b] *
                       (histogram_host[(a * num_bins) + b] == 0.0f
                            ? 0.0f
                            : log2(histogram_host[(a * num_bins) + b]));
    }
  }

  for (int b = 0; b < num_bins; b++) {
    kf_marginal_entropy += P_B[b] * (P_B[b] == 0.0f ? 0.0f : log2(P_B[b]));
  }

  for (int a = 0; a < num_bins; a++) {
    cf_marginal_entropy += P_A[a] * (P_A[a] == 0.0f ? 0.0f : log2(P_A[a]));
  }

  cf_marginal_entropy = -cf_marginal_entropy;
  kf_marginal_entropy = -kf_marginal_entropy;
  joint_entropy = -joint_entropy;

  float mutual_information =
      kf_marginal_entropy + cf_marginal_entropy - joint_entropy;

  nid = (joint_entropy - mutual_information) / joint_entropy;

  if (save) {
    std::ofstream ofs("histogram_array_depth.txt");
    if (ofs.is_open()) {
      for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
          ofs << histogram_host[(i * num_bins) + j] << ",";
        }
        ofs << "\n";
      }
      ofs.close();
    }
  }
  delete[] histogram_host;
  delete[] P_A;
  delete[] P_B;
  cudaFree(histogram_device);
  cudaFree(num_points_device);
  cudaDeviceSynchronize();
}

void computeNIDDepth(const DeviceArray2D<float>& dmap_kf,
                     const DeviceArray2D<float>& dmap_kf_old,
                     const DeviceArray2D<float>& dmap_curr, float& nid,
                     const int& num_bins, const float& max_depth, bool save) {
  float kf_marginal_entropy = 0.0f, cf_marginal_entropy = 0.0f,
        joint_entropy = 0.0f;
  dim3 block(32, 8);
  dim3 grid(getGridDim(dmap_kf.cols(), block.x),
            getGridDim(dmap_kf.rows(), block.y));

  int cols = dmap_kf.cols();
  int rows = dmap_kf.rows();  /// 3;

  float* histogram_device;
  float* histogram_host = new float[num_bins * num_bins];
  memset(histogram_host, 0.0f, sizeof(float) * num_bins * num_bins);
  cudaMalloc(&histogram_device, num_bins * num_bins * sizeof(float));
  cudaMemcpy(histogram_device, histogram_host,
             num_bins * num_bins * sizeof(float), cudaMemcpyHostToDevice);

  int num_points_host = 0;
  int* num_points_device;
  cudaMalloc(&num_points_device, sizeof(int));
  cudaMemcpy(num_points_device, &num_points_host, sizeof(int),
             cudaMemcpyHostToDevice);

  computeJointProbabilityHistogramKernelDepth<<<grid, block>>>(
      dmap_kf, dmap_kf_old, dmap_curr, histogram_device, cols, rows, num_bins,
      max_depth, num_points_device);

  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());

  cudaMemcpy(histogram_host, histogram_device,
             num_bins * num_bins * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&num_points_host, num_points_device, sizeof(int),
             cudaMemcpyDeviceToHost);

  num_points_host = cols * rows;
  if (num_points_host == 0) {
    nid = 1.0f;  // 0.0f;
    return;
  }

  int sum = 0;
  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      sum += histogram_host[(a * num_bins) + b];
    }
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      histogram_host[(a * num_bins) + b] /= (float)(num_points_host);
    }
  }

  float* P_A = new float[num_bins];
  float* P_B = new float[num_bins];

  for (int b = 0; b < num_bins; b++)  // column-wise marginals
  {
    float pb = 0.0f;
    for (int a = 0; a < num_bins; a++) {
      pb += histogram_host[(num_bins * a) + b];
    }
    P_B[b] = pb;
  }

  for (int a = 0; a < num_bins; a++)  // row-wise marginals
  {
    float pa = 0.0f;
    for (int b = 0; b < num_bins; b++) {
      pa += histogram_host[(num_bins * a) + b];
    }
    P_A[a] = pa;
  }

  for (int a = 0; a < num_bins; a++) {
    for (int b = 0; b < num_bins; b++) {
      joint_entropy += histogram_host[(a * num_bins) + b] *
                       (histogram_host[(a * num_bins) + b] == 0.0f
                            ? 0.0f
                            : log2(histogram_host[(a * num_bins) + b]));
    }
  }

  for (int b = 0; b < num_bins; b++) {
    kf_marginal_entropy += P_B[b] * (P_B[b] == 0.0f ? 0.0f : log2(P_B[b]));
  }

  for (int a = 0; a < num_bins; a++) {
    cf_marginal_entropy += P_A[a] * (P_A[a] == 0.0f ? 0.0f : log2(P_A[a]));
  }

  cf_marginal_entropy = -cf_marginal_entropy;
  kf_marginal_entropy = -kf_marginal_entropy;
  joint_entropy = -joint_entropy;

  float mutual_information =
      kf_marginal_entropy + cf_marginal_entropy - joint_entropy;

  nid = (joint_entropy - mutual_information) / joint_entropy;

  if (save) {
    std::ofstream ofs("histogram_array_depth.txt");
    if (ofs.is_open()) {
      for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
          ofs << histogram_host[(i * num_bins) + j] << ",";
        }
        ofs << "\n";
      }
      ofs.close();
    }
  }
  delete[] histogram_host;
  delete[] P_A;
  delete[] P_B;
  cudaFree(histogram_device);
  cudaFree(num_points_device);
  cudaDeviceSynchronize();
}