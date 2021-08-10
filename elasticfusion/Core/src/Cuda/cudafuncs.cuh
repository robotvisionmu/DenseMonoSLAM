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

#ifndef CUDA_CUDAFUNCS_CUH_
#define CUDA_CUDAFUNCS_CUH_

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#include "containers/device_array.hpp"
#include "types.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iomanip>

void icpStep(const mat33& Rcurr, const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr, const mat33& Rprev_inv,
             const float3& tprev, const CameraModel& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev, float distThres,
             float angleThres, DeviceArray<JtJJtrSE3>& sum,
             DeviceArray<JtJJtrSE3>& out, float* matrixA_host,
             float* vectorB_host, float* residual_host, int threads,
             int blocks);

void icpRes(float3& trans_curr, mat33& Rcurr, float3& trans_prev,
            mat33& Rprev_inv, const CameraModel& intr, float angleThres,
            float distThres, DeviceArray2D<float>& vmap_curr,
            DeviceArray2D<float>& nmap_curr, DeviceArray2D<float>& vmap_prev,
            DeviceArray2D<float>& nmap_prev, int& inliers, float& residual);

void rgbStep(const DeviceArray2D<DataTerm>& corresImg, const float& sigma,
             const DeviceArray2D<float3>& cloud, const float& fx,
             const float& fy, const DeviceArray2D<short>& dIdx,
             const DeviceArray2D<short>& dIdy, const float& sobelScale,
             DeviceArray<JtJJtrSE3>& sum, DeviceArray<JtJJtrSE3>& out,
             float* matrixA_host, float* vectorB_host, int threads, int blocks);

void so3Step(const DeviceArray2D<unsigned char>& lastImage,
             const DeviceArray2D<unsigned char>& nextImage,
             const mat33& imageBasis, const mat33& kinv, const mat33& krlr,
             DeviceArray<JtJJtrSO3>& sum, DeviceArray<JtJJtrSO3>& out,
             float* matrixA_host, float* vectorB_host, float* residual_host,
             int threads, int blocks);

void computeRgbResidual(const float& minScale, const DeviceArray2D<short>& dIdx,
                        const DeviceArray2D<short>& dIdy,
                        const DeviceArray2D<float>& lastDepth,
                        const DeviceArray2D<float>& nextDepth,
                        const DeviceArray2D<unsigned char>& lastImage,
                        const DeviceArray2D<unsigned char>& nextImage,
                        DeviceArray2D<DataTerm>& corresImg,
                        DeviceArray<int2>& sumResidual,
                        const float maxDepthDelta, const float3& kt,
                        const mat33& krkinv, int& sigmaSum, int& count,
                        int threads, int blocks);

void createVMap(const CameraModel& intr,
                const DeviceArray2D<unsigned short>& depth,
                DeviceArray2D<float>& vmap, const float depthCutoff,
                cudaStream_t stream = 0);

void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap,
                cudaStream_t stream = 0);

void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src, const mat33& Rmat,
                  const float3& tvec, DeviceArray2D<float>& vmap_dst,
                  DeviceArray2D<float>& nmap_dst);

void tranformMaps(const DeviceArray2D<float>& vmap_src, const mat33& Rmat,
                  const float3& tvec, DeviceArray2D<float>& vmap_dst);

void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst,
              cudaStream_t stream = 0);

void copyMaps(const DeviceArray<float>& vmap_src,
              DeviceArray2D<float>& vmap_dst);

void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output,
                cudaStream_t stream = 0);

void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output,
                cudaStream_t stream = 0);

void imageBGRToIntensity(cudaArray* cuArr, DeviceArray2D<unsigned char>& dst,
                         cudaStream_t stream = 0);

void verticesToDepth(DeviceArray<float>& vmap_src, DeviceArray2D<float>& dst,
                     float cutOff);

void verticesToDepth(DeviceArray2D<float>& vmap_src, DeviceArray2D<float>& dst,
                     float cutOff);


void projectToPointCloud(const DeviceArray2D<float>& depth,
                         const DeviceArray2D<float3>& cloud,
                         CameraModel& intrinsics, const int& level);

void pyrDown(const DeviceArray2D<unsigned short>& src,
             DeviceArray2D<unsigned short>& dst, cudaStream_t stream);

void pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);

void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src,
                       DeviceArray2D<unsigned char>& dst,
                       cudaStream_t stream = 0);

void computeDerivativeImages(DeviceArray2D<unsigned char>& src,
                             DeviceArray2D<short>& dx,
                             DeviceArray2D<short>& dy);

void computeNIDImg(const DeviceArray2D<unsigned char>& img_kf,
                   const DeviceArray2D<unsigned char>& img_kf_old,
                   const DeviceArray2D<float>& dmap_kf,
                   const DeviceArray2D<float>& dmap_kf_old,
                   const DeviceArray2D<unsigned char>& img_curr, float& nid,
                   const int& num_bins, bool save = false);

void computeNIDDepth(const DeviceArray2D<float>& dmap_kf,
                     const DeviceArray2D<float>& dmap_kf_old,
                     const DeviceArray2D<float>& dmap_curr, float& nid,
                     const int& num_bins, const float& max_depth,
                     bool save = false);

void computeNIDImgSmem(const DeviceArray2D<unsigned char>& img_kf,
                       const DeviceArray2D<unsigned char>& img_kf_old,
                       const DeviceArray2D<float>& dmap_kf,
                       const DeviceArray2D<float>& dmap_kf_old,
                       const DeviceArray2D<unsigned char>& img_curr, float& nid,
                       const int& num_bins, bool save = false);

void computeNIDDepthSmem(const DeviceArray2D<float>& dmap_kf,
                         const DeviceArray2D<float>& dmap_kf_old,
                         const DeviceArray2D<float>& dmap_curr, float& nid,
                         const int& num_bins, const float& max_depth,
                         bool save = false);

void fgrStep(const mat33& Rcurr, const float3& tcurr,
             const DeviceArray<float>& pci, const DeviceArray<float>& pcj,
             DeviceArray<FGRDataTerm>& corres, float par,
             DeviceArray<JtJJtrSE3>& sum, DeviceArray<JtJJtrSE3>& out,
             float* matrixA_host, float* vectorB_host, float* residual_host,
             int threads, int blocks);

void fgrPDAStep(const mat33& Rcurr, const float3& tcurr,
                const DeviceArray2D<float>& vmap_previous,
                const DeviceArray2D<float>& vmap_curr, CameraModel& intr,
                DeviceArray2D<FGRPDADataTerm>& corres, float par,
                DeviceArray<JtJJtrSE3>& sum, DeviceArray<JtJJtrSE3>& out,
                float* matrixA_host, float* vectorB_host, float* residual_host,
                int threads, int blocks);

void computeCorrespondences(const mat33& Rcurr, const float3& tcurr,
                            const DeviceArray2D<float>& vmap_curr,
                            const DeviceArray2D<float>& nmap_curr,
                            const mat33& Rprev_inv, const float3& tprev,
                            const DeviceArray2D<float>& vmap_prev,
                            const DeviceArray2D<float>& nmap_prev,
                            DeviceArray2D<FGRPDADataTerm>& corres,
                            CameraModel intr, const float& distanceThresh,
                            const float& angleThresh, const int& blocks,
                            const int& threads);
#endif /* CUDA_CUDAFUNCS_CUH_ */
