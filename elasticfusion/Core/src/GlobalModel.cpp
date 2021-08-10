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

#include "GlobalModel.h"

const int GlobalModel::TEXTURE_DIMENSION = 5700;//3072 * 2;
const int GlobalModel::MAX_VERTICES =
    (GlobalModel::TEXTURE_DIMENSION * GlobalModel::TEXTURE_DIMENSION);// * 2);// / Vertex::MAX_SENSORS;  //* 1000;
const int GlobalModel::NODE_TEXTURE_DIMENSION = 16384 * 2;
const int GlobalModel::MAX_NODES =
    GlobalModel::NODE_TEXTURE_DIMENSION / 16;  // 16 floats per node

GlobalModel::GlobalModel()
    : target(0),
      renderSource(1),
      bufferSize(MAX_VERTICES * Vertex::SIZE),
      //count(0),
      initProgram(loadProgramFromFile("init_unstable.vert")),
      drawProgram(
          loadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
      drawSurfelProgram(loadProgramFromFile("draw_global_surface.vert",
                                            "draw_global_surface.frag",
                                            "draw_global_surface.geom")),
      dataProgram(loadProgramFromFile("data.vert", "data.frag", "data.geom")),
      updateProgram(loadProgramFromFile("update.vert")),
      unstableProgram(
          loadProgramGeomFromFile("copy_unstable.vert", "copy_unstable.geom")),
      consumeProgram(loadProgramFromFile("consume.vert")),
      renderBuffer(TEXTURE_DIMENSION, TEXTURE_DIMENSION),
      updateMapVertsConfs(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F,
                          GL_LUMINANCE, GL_FLOAT),
      updateMapColorsTime(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F,
                          GL_LUMINANCE, GL_FLOAT),
      updateMapNormsRadii(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F,
                          GL_LUMINANCE, GL_FLOAT),
      deformationNodes(NODE_TEXTURE_DIMENSION, 1, GL_LUMINANCE32F_ARB,
                       GL_LUMINANCE, GL_FLOAT) {
  std::pair<GLuint, GLuint>* vbos = new std::pair<GLuint, GLuint>[2];

  current_cluster = 0;
  cluster_vbos[0] = vbos;
  cluster_global_poses[0] = Eigen::Matrix4f::Identity();
  cluster_count[current_cluster] = 0;

  cudaResources = new cudaGraphicsResource*[2];
  float* vertices = new float[bufferSize];

  memset(&vertices[0], 0, bufferSize);

  glGenTransformFeedbacks(1, &vbos[0].second);
  glGenBuffers(1, &vbos[0].first);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(cudaResources[0]), vbos[0].first,
                                            cudaGraphicsMapFlagsNone));

  glGenTransformFeedbacks(1, &vbos[1].second);
  glGenBuffers(1, &vbos[1].first);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[1].first);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(cudaResources[1]), vbos[1].first,
                                            cudaGraphicsMapFlagsNone));

  delete[] vertices;

  vertices = new float[Resolution::getInstance().numPixels() * Vertex::SIZE];

  memset(&vertices[0], 0, Resolution::getInstance().numPixels() * Vertex::SIZE);

  glGenTransformFeedbacks(1, &newUnstableFid);
  glGenBuffers(1, &newUnstableVbo);
  glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);
  glBufferData(GL_ARRAY_BUFFER,
               Resolution::getInstance().numPixels() * Vertex::SIZE,
               &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] vertices;

  std::vector<Eigen::Vector2f> uv;

  for (int i = 0; i < Resolution::getInstance().width(); i++) {
    for (int j = 0; j < Resolution::getInstance().height(); j++) {
      uv.push_back(Eigen::Vector2f(
          ((float)i / (float)Resolution::getInstance().width()) +
              1.0 / (2 * (float)Resolution::getInstance().width()),
          ((float)j / (float)Resolution::getInstance().height()) +
              1.0 / (2 * (float)Resolution::getInstance().height())));
    }
  }

  uvSize = uv.size();

  glGenBuffers(1, &uvo);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glBufferData(GL_ARRAY_BUFFER, uvSize * sizeof(Eigen::Vector2f), &uv[0],
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  frameBuffer.AttachColour(*updateMapVertsConfs.texture);
  frameBuffer.AttachColour(*updateMapColorsTime.texture);
  frameBuffer.AttachColour(*updateMapNormsRadii.texture);
  frameBuffer.AttachDepth(renderBuffer);

  updateProgram->Bind();

  int locUpdate[4] = {
      glGetVaryingLocationNV(updateProgram->programId(), "vPosition0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vTimes0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vNormRad0"),
  };
  glTransformFeedbackVaryingsNV(updateProgram->programId(), 4, locUpdate,
                                GL_INTERLEAVED_ATTRIBS);
  updateProgram->Unbind();

  glActiveVaryingNV(dataProgram->programId(), "vTimes0");
  dataProgram->Link();  // TODO can I get rid of this
  dataProgram->Bind();
  int dataUpdate[4] = {
      glGetVaryingLocationNV(dataProgram->programId(), "vPosition0"),
      glGetVaryingLocationNV(dataProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(dataProgram->programId(), "vTimes0"),
      glGetVaryingLocationNV(dataProgram->programId(), "vNormRad0"),
  };

  glTransformFeedbackVaryingsNV(dataProgram->programId(), 4, dataUpdate,
                                GL_INTERLEAVED_ATTRIBS);
  dataProgram->Unbind();

  unstableProgram->Bind();

  int unstableUpdate[4] = {
      glGetVaryingLocationNV(unstableProgram->programId(), "vPosition0"),
      glGetVaryingLocationNV(unstableProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(unstableProgram->programId(), "vTimes0"),
      glGetVaryingLocationNV(unstableProgram->programId(), "vNormRad0"),
  };

  glTransformFeedbackVaryingsNV(unstableProgram->programId(), 4, unstableUpdate,
                                GL_INTERLEAVED_ATTRIBS);

  unstableProgram->Unbind();

  consumeProgram->Bind();

  int consumeUpdate[4]{
      glGetVaryingLocationNV(updateProgram->programId(), "vPosition0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vTimes0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vNormRad0"),
  };

  glTransformFeedbackVaryingsNV(consumeProgram->programId(), 4, consumeUpdate,
                                GL_INTERLEAVED_ATTRIBS);

  consumeProgram->Unbind();

  initProgram->Bind();
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  initProgram->setUniform(Uniform("t_inv", pose));

  int locInit[4] = {
      glGetVaryingLocationNV(initProgram->programId(), "vPosition0"),
      glGetVaryingLocationNV(initProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vTimes0"),
      glGetVaryingLocationNV(initProgram->programId(), "vNormRad0"),
  };

  glTransformFeedbackVaryingsNV(initProgram->programId(), 4, locInit,
                                GL_INTERLEAVED_ATTRIBS);

  glGenQueries(1, &countQuery);

  // Empty both transform feedbacks
  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].first);

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, 0);

  glEndTransformFeedback();

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[1].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[1].first);

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, 0);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  initProgram->Unbind();

  CheckGlDieOnError();
}

GlobalModel::~GlobalModel() {
  cudaSafeCall(cudaGraphicsUnregisterResource(cudaResources[0]));
  cudaSafeCall(cudaGraphicsUnregisterResource(cudaResources[1]));

  for (auto& c : cluster_vbos) {
    std::pair<GLuint, GLuint>* vbos = c.second;

    glDeleteBuffers(1, &vbos[0].first);
    glDeleteTransformFeedbacks(1, &vbos[0].second);

    glDeleteBuffers(1, &vbos[1].first);
    glDeleteTransformFeedbacks(1, &vbos[1].second);

    delete[] vbos;
  }

  glDeleteQueries(1, &countQuery);

  glDeleteBuffers(1, &uvo);

  glDeleteTransformFeedbacks(1, &newUnstableFid);
  glDeleteBuffers(1, &newUnstableVbo);
}

bool GlobalModel::isCluster(const int cluster) {
  return cluster_vbos.count(cluster);
}

std::vector<int> GlobalModel::clusters()
{
  std::vector<int> clusters;
  for(const auto & kv : cluster_vbos)
  {
    clusters.push_back(kv.first);
  }

  return clusters;
}

void GlobalModel::initialise(const FeedbackBuffer& rawFeedback,
                             const FeedbackBuffer& filteredFeedback,
                             const int& cluster, Eigen::Matrix4f pose) {
  const auto cluster_vbo = cluster_vbos.find(cluster);
  std::pair<GLuint, GLuint>* vbos;
  if (cluster_vbo == cluster_vbos.end()) {
    vbos = new std::pair<GLuint, GLuint>[2];

    current_cluster = cluster;
    cluster_vbos[cluster] = vbos;
    cluster_global_poses[cluster] = pose;

    cudaResources = new cudaGraphicsResource*[2];
    float* vertices = new float[bufferSize];

    memset(&vertices[0], 0, bufferSize);

    glGenTransformFeedbacks(1, &vbos[0].second);
    glGenBuffers(1, &vbos[0].first);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(
        &(cudaResources[0]), vbos[0].first, cudaGraphicsMapFlagsNone));

    glGenTransformFeedbacks(1, &vbos[1].second);
    glGenBuffers(1, &vbos[1].first);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1].first);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(
        &(cudaResources[1]), vbos[1].first, cudaGraphicsMapFlagsNone));

    delete[] vertices;

    initProgram->Bind();

    initProgram->setUniform(Uniform("t_inv", Eigen::Matrix4f(pose)));

    int locInit[4] = {
        glGetVaryingLocationNV(initProgram->programId(), "vPosition0"),
        glGetVaryingLocationNV(initProgram->programId(), "vColor0"),
        glGetVaryingLocationNV(updateProgram->programId(), "vTimes0"),
        glGetVaryingLocationNV(initProgram->programId(), "vNormRad0"),
    };

    glTransformFeedbackVaryingsNV(initProgram->programId(), 4, locInit,
                                  GL_INTERLEAVED_ATTRIBS);

    glGenQueries(1, &countQuery);

    // Empty both transform feedbacks
    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].first);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, 0);

    glEndTransformFeedback();

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[1].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[1].first);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, 0);

    glEndTransformFeedback();

    glDisable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    initProgram->Unbind();

    CheckGlDieOnError();
  } else {
    vbos = cluster_vbo->second;
  }

  initProgram->Bind();

  glBindBuffer(GL_ARRAY_BUFFER, rawFeedback.vbo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  int timeOffset = sizeof(Eigen::Vector4f) * 2;
  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glEnableVertexAttribArray(2 + i);
    glVertexAttribPointer(
        2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
  }

  glBindBuffer(GL_ARRAY_BUFFER, filteredFeedback.vbo);

  glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
  glVertexAttribPointer(2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE,
                        Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(
                            timeOffset + sizeof(float) * Vertex::MAX_SENSORS));

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].first);

  glBeginTransformFeedback(GL_POINTS);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  // It's ok to use either fid because both raw and filtered have the same
  // amount of vertices
  glDrawTransformFeedback(GL_POINTS, rawFeedback.fid);

  glEndTransformFeedback();

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &cluster_count[current_cluster]);

  glDisable(GL_RASTERIZER_DISCARD);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);

  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glDisableVertexAttribArray(2 + i);
  }

  glDisableVertexAttribArray(2 + Vertex::MAX_SENSORS);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  initProgram->Unbind();

  glFinish();
  CheckGlDieOnError();
}

void GlobalModel::renderPointCloud(
    pangolin::OpenGlMatrix mvp, const float threshold, const bool drawUnstable,
    const bool drawNormals, const bool drawColors, const bool drawPoints,
    const bool drawWindow, const bool drawTimes, const bool drawContributions,
    const int time, const int timeIdx, const int timeDelta,
    std::vector<int> clusters,
    bool drawClusters,
    std::vector<std::tuple<float, float, float>> cluster_colors) {
  std::shared_ptr<Shader> program =
      drawPoints ? drawProgram : drawSurfelProgram;

  program->Bind();

  program->setUniform(Uniform("MVP", mvp));

  program->setUniform(Uniform("threshold", threshold));

  program->setUniform(
      Uniform("colorType",
              (drawContributions
                   ? 4
                   : drawNormals ? 1 : drawColors ? 2 : drawTimes ? 3 : 0)));

  program->setUniform(Uniform("unstable", drawUnstable));

  program->setUniform(Uniform("drawWindow", drawWindow));

  program->setUniform(Uniform("time", time));

  program->setUniform(Uniform("timeIdx", timeIdx));

  program->setUniform(Uniform("timeDelta", timeDelta));

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  // This is for the point shader
  program->setUniform(Uniform("pose", pose));

  program->setUniform(Uniform("cluster", drawClusters));

  for (size_t i = 0; i < clusters.size(); i++) { 
    if(drawClusters)
    {
      std::tuple<float, float, float> color = cluster_colors[i];
      program->setUniform(Uniform("cluster_color", Eigen::Vector3f(std::get<0>(color),std::get<1>(color),std::get<2>(color))));
    }

    std::pair<GLuint, GLuint>* vbos = cluster_vbos[i];

    glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

    int timeOffset = sizeof(Eigen::Vector4f) * 2;
    for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
      glEnableVertexAttribArray(2 + i);
      glVertexAttribPointer(
          2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
          reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
    }

    glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
    glVertexAttribPointer(
        2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset +
                                  sizeof(float) * Vertex::MAX_SENSORS));

    glDrawTransformFeedback(GL_POINTS, vbos[target].second);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
      glDisableVertexAttribArray(2 + i);
    }
    glDisableVertexAttribArray(2 + Vertex::MAX_SENSORS);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  program->Unbind();
  CheckGlDieOnError();
}

const std::pair<GLuint, GLuint>& GlobalModel::model() {
  return cluster_vbos[current_cluster][target];
}

cudaGraphicsResource* GlobalModel::cudaModel() { return cudaResources[target]; }

void GlobalModel::fuse(const Eigen::Matrix4f& pose, const int& time,
                       const int& timeIdx, GPUTexture* rgb,
                       GPUTexture* depthRaw, GPUTexture* depthFiltered,
                       GPUTexture* indexMap, GPUTexture* vertConfMap,
                       GPUTexture* colorTimeMap, GPUTexture* normRadMap,
                       const float depthCutoff, const float confThreshold,
                       const float weighting, const int cluster) {
  TICK("Fuse::Data");
  // This first part does data association and computes the vertex to merge
  // with, storing
  // in an array that sets which vertices to update by index
  frameBuffer.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, renderBuffer.width, renderBuffer.height);

  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  dataProgram->Bind();

  dataProgram->setUniform(Uniform("cSampler", 0));
  dataProgram->setUniform(Uniform("drSampler", 1));
  dataProgram->setUniform(Uniform("drfSampler", 2));
  dataProgram->setUniform(Uniform("indexSampler", 3));
  dataProgram->setUniform(Uniform("vertConfSampler", 4));
  dataProgram->setUniform(Uniform("colorTimeSampler", 5));
  dataProgram->setUniform(Uniform("normRadSampler", 6));
  dataProgram->setUniform(Uniform("time", (float)time));
  dataProgram->setUniform(Uniform("timeIdx", timeIdx));
  dataProgram->setUniform(Uniform("weighting", weighting));

  dataProgram->setUniform(
      Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                     Intrinsics::getInstance().cy(),
                                     1.0 / Intrinsics::getInstance().fx(),
                                     1.0 / Intrinsics::getInstance().fy())));
  dataProgram->setUniform(
      Uniform("cols", (float)Resolution::getInstance().cols()));
  dataProgram->setUniform(
      Uniform("rows", (float)Resolution::getInstance().rows()));
  dataProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
  dataProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  dataProgram->setUniform(Uniform("pose", pose));
  dataProgram->setUniform(Uniform("maxDepth", depthCutoff));

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, newUnstableFid);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, newUnstableVbo);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, rgb->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depthRaw->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, uvSize);

  glEndTransformFeedback();

  frameBuffer.Unbind();

  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  dataProgram->Unbind();

  glPopAttrib();

  glFinish();
  TOCK("Fuse::Data");

  TICK("Fuse::Update");
  // Next we update the vertices at the indexes stored in the update textures
  // Using a transform feedback conditional on a texture sample
  updateProgram->Bind();

  updateProgram->setUniform(Uniform("vertSamp", 0));
  updateProgram->setUniform(Uniform("colorSamp", 1));
  updateProgram->setUniform(Uniform("normSamp", 2));
  updateProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  updateProgram->setUniform(Uniform("time", time));
  updateProgram->setUniform(Uniform("timeIdx", timeIdx));

  std::pair<GLuint, GLuint>* vbos = cluster_vbos[current_cluster];

  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  int timeOffset = sizeof(Eigen::Vector4f) * 2;
  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glEnableVertexAttribArray(2 + i);
    glVertexAttribPointer(
        2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
  }

  glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
  glVertexAttribPointer(2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE,
                        Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(
                            timeOffset + sizeof(float) * Vertex::MAX_SENSORS));

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, updateMapVertsConfs.texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, updateMapColorsTime.texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, updateMapNormsRadii.texture->tid);

  glDrawTransformFeedback(GL_POINTS, vbos[target].second);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);

  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glDisableVertexAttribArray(2 + i);
  }

  glDisableVertexAttribArray(2 + Vertex::MAX_SENSORS);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  updateProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Update");
  CheckGlDieOnError();
}

void GlobalModel::clean(const Eigen::Matrix4f& pose, const int& time,
                        const int& timeIdx, GPUTexture* indexMap,
                        GPUTexture* vertConfMap, GPUTexture* colorTimeMap,
                        GPUTexture* normRadMap, GPUTexture* depthMap,
                        const float confThreshold, std::vector<float>& graph,
                        const int timeDelta, const float maxDepth,
                        const bool isFern, const int cluster) {
  assert(graph.size() / 16 < MAX_NODES);

  if (graph.size() > 0) {
    // Can be optimised by only uploading new nodes with offset
    glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, graph.size(), 1, GL_LUMINANCE,
                    GL_FLOAT, graph.data());
  }

  TICK("Fuse::Copy");
  // Next we copy the new unstable vertices from the newUnstableFid transform
  // feedback into the global map
  unstableProgram->Bind();
  unstableProgram->setUniform(Uniform("time", time));
  unstableProgram->setUniform(Uniform("timeIdx", timeIdx));
  unstableProgram->setUniform(Uniform("confThreshold", confThreshold));
  unstableProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
  unstableProgram->setUniform(Uniform("indexSampler", 0));
  unstableProgram->setUniform(Uniform("vertConfSampler", 1));
  unstableProgram->setUniform(Uniform("colorTimeSampler", 2));
  unstableProgram->setUniform(Uniform("normRadSampler", 3));
  unstableProgram->setUniform(Uniform("nodeSampler", 4));
  unstableProgram->setUniform(Uniform("depthSampler", 5));
  unstableProgram->setUniform(Uniform("nodes", (float)(graph.size() / 16)));
  unstableProgram->setUniform(
      Uniform("nodeCols", (float)NODE_TEXTURE_DIMENSION));
  unstableProgram->setUniform(Uniform("timeDelta", timeDelta));
  unstableProgram->setUniform(Uniform("maxDepth", maxDepth));
  unstableProgram->setUniform(Uniform("isFern", (int)isFern));

  Eigen::Matrix4f t_inv = pose.inverse();
  unstableProgram->setUniform(Uniform("t_inv", t_inv));

  unstableProgram->setUniform(
      Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                     Intrinsics::getInstance().cy(),
                                     Intrinsics::getInstance().fx(),
                                     Intrinsics::getInstance().fy())));
  unstableProgram->setUniform(
      Uniform("cols", (float)Resolution::getInstance().cols()));
  unstableProgram->setUniform(
      Uniform("rows", (float)Resolution::getInstance().rows()));

  std::pair<GLuint, GLuint>* vbos = cluster_vbos[current_cluster];

  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  int timeOffset = sizeof(Eigen::Vector4f) * 2;
  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glEnableVertexAttribArray(2 + i);
    glVertexAttribPointer(
        2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
  }

  glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
  glVertexAttribPointer(2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE,
                        Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(
                            timeOffset + sizeof(float) * Vertex::MAX_SENSORS));

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, depthMap->texture->tid);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  glDrawTransformFeedback(GL_POINTS, vbos[target].second);

  glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glEnableVertexAttribArray(2 + i);
    glVertexAttribPointer(
        2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
  }

  glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
  glVertexAttribPointer(2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE,
                        Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(
                            timeOffset + sizeof(float) * Vertex::MAX_SENSORS));

  glDrawTransformFeedback(GL_POINTS, newUnstableFid);

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &cluster_count[current_cluster]);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glDisableVertexAttribArray(2 + i);
  }
  glDisableVertexAttribArray(2 + Vertex::MAX_SENSORS);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  unstableProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Copy");
  CheckGlDieOnError();
}

unsigned int GlobalModel::lastCount() { return cluster_count[current_cluster]; }
unsigned int GlobalModel::totalCount()
{
  unsigned int total_count = 0;
  for(const auto & kv : cluster_count)
  {
    total_count += kv.second;
  }
   return total_count; 
}

float* GlobalModel::downloadMap() {
  glFinish();

  float* vertices = new float[cluster_count[current_cluster] * (3 * 4 + Vertex::MAX_SENSORS)];

  memset(&vertices[0], 0, cluster_count[current_cluster] * Vertex::SIZE);

  GLuint downloadVbo;

  glGenBuffers(1, &downloadVbo);
  glBindBuffer(GL_ARRAY_BUFFER, downloadVbo);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  std::pair<GLuint, GLuint>* vbos = cluster_vbos[current_cluster];

  glBindBuffer(GL_COPY_READ_BUFFER, vbos[renderSource].first);
  glBindBuffer(GL_COPY_WRITE_BUFFER, downloadVbo);

  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                      cluster_count[current_cluster] * Vertex::SIZE);
  glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, cluster_count[current_cluster] * Vertex::SIZE, vertices);

  glBindBuffer(GL_COPY_READ_BUFFER, 0);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  glDeleteBuffers(1, &downloadVbo);

  glFinish();
  CheckGlDieOnError();
  return vertices;
}

void GlobalModel::consume(const std::pair<GLuint, GLuint>& model,
                          const Eigen::Matrix4f& relativeTransform,
                          const int cluster) {
  TICK("GlobalModel::consume");

  consumeProgram->Bind();
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  consumeProgram->setUniform(Uniform("transform", transform));

  std::pair<GLuint, GLuint>* vbos = cluster_vbos[current_cluster];
  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  int timeOffset = sizeof(Eigen::Vector4f) * 2;
  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glEnableVertexAttribArray(2 + i);
    glVertexAttribPointer(
        2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
  }

  glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
  glVertexAttribPointer(2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE,
                        Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(
                            timeOffset + sizeof(float) * Vertex::MAX_SENSORS));

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

  glBeginTransformFeedback(GL_POINTS);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  glDrawTransformFeedback(GL_POINTS, vbos[target].second);

  consumeProgram->setUniform(Uniform("transform", relativeTransform));
  glBindBuffer(GL_ARRAY_BUFFER, model.first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glEnableVertexAttribArray(2 + i);
    glVertexAttribPointer(
        2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
  }

  glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
  glVertexAttribPointer(2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE,
                        Vertex::SIZE,
                        reinterpret_cast<GLvoid*>(
                            timeOffset + sizeof(float) * Vertex::MAX_SENSORS));

  glDrawTransformFeedback(GL_POINTS, model.second);

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &cluster_count[current_cluster]);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
    glDisableVertexAttribArray(2 + i);
  }
  glDisableVertexAttribArray(2 + Vertex::MAX_SENSORS);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  consumeProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();

  TOCK("GlobalModel::consume");
}

void GlobalModel::consume(cudaGraphicsResource* model, const int& modelCount,
                          const Eigen::Matrix4f& relativeTransform) {
  TICK("GlobalModel::Consume");
  DeviceArray<float> src_map_1(cluster_count[current_cluster] * (Vertex::MAX_SENSORS + 3 * 4)),
      src_map_2(modelCount * (Vertex::MAX_SENSORS + 3 * 4)),
      dst_map((cluster_count[current_cluster] + modelCount) * (Vertex::MAX_SENSORS + 3 * 4));
  float *src_1_ptr, *src_2_ptr, *dst_ptr;

  cudaSafeCall(cudaGraphicsMapResources(1, &(cudaResources[target]), 0));
  cudaSafeCall(cudaGraphicsMapResources(1, &(cudaResources[renderSource]), 0));
  cudaSafeCall(cudaGraphicsMapResources(1, &model, 0));

  size_t num_bytes;
  cudaSafeCall(cudaGraphicsResourceGetMappedPointer(
      (void**)&src_1_ptr, &num_bytes, cudaResources[target]));
  cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&src_2_ptr,
                                                    &num_bytes, model));
  cudaSafeCall(cudaGraphicsResourceGetMappedPointer(
      (void**)&dst_ptr, &num_bytes, cudaResources[renderSource]));

  cudaSafeCall(cudaMemcpy(src_map_1, src_1_ptr, cluster_count[current_cluster] * Vertex::SIZE,
                          cudaMemcpyDeviceToDevice));
  cudaSafeCall(cudaMemcpy(src_map_2, src_2_ptr, modelCount * Vertex::SIZE,
                          cudaMemcpyDeviceToDevice));

  cudaSafeCall(cudaGraphicsUnmapResources(1, &(cudaResources[target]), 0));
  cudaSafeCall(cudaGraphicsUnmapResources(1, &model, 0));

  Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity();
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> r_1 = T1.topLeftCorner(3, 3);
  Eigen::Vector3f t_1 = T1.topRightCorner(3, 1);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> r_2 =
      relativeTransform.topLeftCorner(3, 3);
  Eigen::Vector3f t_2 = relativeTransform.topRightCorner(3, 1);

  mat33 device_r_1 = r_1;
  float3 device_t_1 = *reinterpret_cast<float3*>(t_1.data());

  mat33 device_r_2 = r_2;
  float3 device_t_2 = *reinterpret_cast<float3*>(t_2.data());

  mergePointClouds(src_map_1, device_r_1, device_t_1, cluster_count[current_cluster], src_map_2,
                   device_r_2, device_t_2, modelCount, dst_map,
                   Vertex::MAX_SENSORS + 3 * 4);

  cudaSafeCall(cudaDeviceSynchronize());

  cudaSafeCall(cudaMemcpy(dst_ptr, dst_map, (modelCount + cluster_count[current_cluster]) * Vertex::SIZE,
                          cudaMemcpyDeviceToDevice));

  cudaSafeCall(
      cudaGraphicsUnmapResources(1, &(cudaResources[renderSource]), 0));

  cluster_count[current_cluster] += modelCount;

  std::swap(target, renderSource);

  TOCK("GlobalModel::Consume");
}