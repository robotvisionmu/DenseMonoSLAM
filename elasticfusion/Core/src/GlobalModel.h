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
 */

#ifndef GLOBALMODEL_H_
#define GLOBALMODEL_H_

#include <pangolin/gl/gl.h>
#include <Eigen/LU>
#include "Cuda/convenience.cuh"
#include "Cuda/cudafuncs.cuh"
#include "Cuda/merging/merge.cuh"
#include "GPUTexture.h"
#include "IndexMap.h"
#include "Shaders/FeedbackBuffer.h"
#include "Shaders/Shaders.h"
#include "Shaders/Uniform.h"
#include "Utils/Intrinsics.h"
#include "Utils/Resolution.h"
#include "Utils/Stopwatch.h"

#include "Defines.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

class GlobalModel {
 public:
  GlobalModel();
  virtual ~GlobalModel();

  void initialise(const FeedbackBuffer &rawFeedback,
                  const FeedbackBuffer &filteredFeedback,
                  const int &cluster = 0,
                  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity());

  static const int TEXTURE_DIMENSION;
  static const int MAX_VERTICES;
  static const int NODE_TEXTURE_DIMENSION;
  static const int MAX_NODES;

  EFUSION_API void renderPointCloud(
      pangolin::OpenGlMatrix mvp, const float threshold,
      const bool drawUnstable, const bool drawNormals, const bool drawColors,
      const bool drawPoints, const bool drawWindow, const bool drawTimes,
      const bool drawContributions, const int time, const int timeIdx,
      const int timeDelta, std::vector<int> clusters,
      bool drawClusters = false,
      std::vector<std::tuple<float, float, float>> cluster_colors = {});

  EFUSION_API const std::pair<GLuint, GLuint> &model();
  EFUSION_API cudaGraphicsResource *cudaModel();

  void fuse(const Eigen::Matrix4f &pose, const int &time, const int &timeIdx,
            GPUTexture *rgb, GPUTexture *depthRaw, GPUTexture *depthFiltered,
            GPUTexture *indexMap, GPUTexture *vertConfMap,
            GPUTexture *colorTimeMap, GPUTexture *normRadMap,
            const float depthCutoff, const float confThreshold,
            const float weighting, const int cluster = 0);

  void clean(const Eigen::Matrix4f &pose, const int &time, const int &timeIdx,
             GPUTexture *indexMap, GPUTexture *vertConfMap,
             GPUTexture *colorTimeMap, GPUTexture *normRadMap,
             GPUTexture *depthMap, const float confThreshold,
             std::vector<float> &graph, const int timeDelta,
             const float maxDepth, const bool isFern, const int cluster = 0);

  EFUSION_API unsigned int lastCount();
  EFUSION_API unsigned int totalCount();

  void consume(const std::pair<GLuint, GLuint> &model,
               const Eigen::Matrix4f &relativeTransform, const int cluster = 0);
  void consume(cudaGraphicsResource *model, const int &modelCount,
               const Eigen::Matrix4f &relativeTransform);

  float *downloadMap();
  bool isCluster(const int cluster);
  std::vector<int> clusters();

 private:
  // First is the vbo, second is the fid
  // std::pair<GLuint, GLuint> * vbos;
  int current_cluster;
  std::map<int, std::pair<GLuint, GLuint> *> cluster_vbos;
  std::map<int, Eigen::Matrix4f, std::less<int>,Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix4f>>> cluster_global_poses;

  cudaGraphicsResource **cudaResources;
  int target, renderSource;

  const int bufferSize;

  GLuint countQuery;
  std::map<int, unsigned int> cluster_count;
  
  std::shared_ptr<Shader> initProgram;
  std::shared_ptr<Shader> drawProgram;
  std::shared_ptr<Shader> drawSurfelProgram;

  // For supersample fusing
  std::shared_ptr<Shader> dataProgram;
  std::shared_ptr<Shader> updateProgram;
  std::shared_ptr<Shader> unstableProgram;
  std::shared_ptr<Shader> consumeProgram;
  pangolin::GlRenderBuffer renderBuffer;

  // We render updated vertices vec3 + confidences to one texture
  GPUTexture updateMapVertsConfs;

  // We render updated colors vec3 + timestamps to another
  GPUTexture updateMapColorsTime;

  // We render updated normals vec3 + radii to another
  GPUTexture updateMapNormsRadii;

  // 16 floats stored column-major yo'
  GPUTexture deformationNodes;

  GLuint newUnstableVbo, newUnstableFid;

  pangolin::GlFramebuffer frameBuffer;
  GLuint uvo;
  int uvSize;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif /* GLOBALMODEL_H_ */
