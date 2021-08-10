#ifndef CONTEXT_H_
#define CONTEXT_H_

#include "Defines.h"
#include "Deformation.h"
#include "Ferns.h"
#include "GlobalModel.h"
#include "IndexMap.h"
#include "MutualInformation.h"
#include "PoseMatch.h"
#include "Shaders/ComputePack.h"
#include "Shaders/FeedbackBuffer.h"
#include "Shaders/FillIn.h"
#include "Shaders/Shaders.h"
#include "Utils/Intrinsics.h"
#include "Utils/RGBDOdometry.h"
#include "Utils/Resolution.h"
#include "Utils/Stats.h"
#include "Utils/Stopwatch.h"

#include <iomanip>

#include <pangolin/gl/glcuda.h>

class Context {
 public:
  Context(const int id, const int num_bins_depth, const int num_bins_img,
          const std::string filename = "", const bool iclnuim = false,
          const bool reloc = false)
      : m_mi(num_bins_depth, num_bins_img),
        m_frameToModel(
            Resolution::getInstance().width(),
            Resolution::getInstance().height(), Intrinsics::getInstance().cx(),
            Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(),
            Intrinsics::getInstance().fy()),
        m_modelToModel(
            Resolution::getInstance().width(),
            Resolution::getInstance().height(), Intrinsics::getInstance().cx(),
            Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(),
            Intrinsics::getInstance().fy()),
        m_currPose(Eigen::Matrix4f::Identity()),
        m_id(id),
        m_tick(1),
        m_deforms(0),
        m_consSample(20),
        m_poseGraph(Eigen::aligned_allocator<
                    std::pair<unsigned long long int, Eigen::Matrix4f>>()),
        m_imageBuff(Resolution::getInstance().rows() / m_consSample,
                    Resolution::getInstance().cols() / m_consSample),
        m_consBuff(Resolution::getInstance().rows() / m_consSample,
                   Resolution::getInstance().cols() / m_consSample),
        m_timesBuff(Resolution::getInstance().rows() / m_consSample,
                    Resolution::getInstance().cols() / m_consSample),
        m_iclnuim(iclnuim),
        m_reloc(reloc),
        m_lost(false),
        m_lastFrameRecovery(false),
        m_trackingCount(0),
        m_maxDepthProcessed(20.0f),
        m_rgbOnly(false),
        m_saveFilename(filename),
        m_weighting(0.0f),
        m_trackingOk(false),
        m_new_keyframe(true),
        m_num_fused(0),
        m_frames_since_last_fusion(0) {
    createTextures();
    createCompute();
    createFeedbackBuffers();

    cudaStreamCreate(&m_stream);

    std::string fn = filename;
    fn.append(".freiburg");

    std::ofstream file;
    file.open(fn.c_str(), std::fstream::out);
    file.close();
  }

  virtual ~Context() {
    for (std::map<std::string, GPUTexture*>::iterator it = m_textures.begin();
         it != m_textures.end(); ++it) {
      delete it->second;
    }

    m_textures.clear();

    for (std::map<std::string, ComputePack*>::iterator it =
             m_computePacks.begin();
         it != m_computePacks.end(); ++it) {
      delete it->second;
    }

    m_computePacks.clear();

    for (std::map<std::string, FeedbackBuffer*>::iterator it =
             m_feedbackBuffers.begin();
         it != m_feedbackBuffers.end(); ++it) {
      delete it->second;
    }

    m_feedbackBuffers.clear();
  }

  void saveStats(std::string dir) {
    std::string fname = dir;
    std::string logname = m_saveFilename.substr(m_saveFilename.find_last_of("/")+1);
    fname.append(logname);
    fname.append(".stats");

    std::cout << "saving session statistics: " << fname << "\n";
    
    m_stats.write(fname);
  }

  void saveTrajectory(std::string dir) {
    std::string fname = dir;
    std::string logname = m_saveFilename.substr(m_saveFilename.find_last_of("/"));
    fname.append(logname);
    fname.append(".freiburg");

    std::cout << "saving to " << fname << std::endl;

    std::ofstream f;
    f.open(fname.c_str(), std::fstream::out);

    for (size_t i = 0; i < m_poseGraph.size(); i++) {
      std::stringstream strs;

      if (m_iclnuim) {
        strs << std::setprecision(6) << std::fixed
             << (double)m_poseLogTimes.at(i) << " ";
      } else {
        strs << std::setprecision(6) << std::fixed
             << (double)m_poseLogTimes.at(i) / 1000000.0 << " ";
      }

      Eigen::Vector3f trans = m_poseGraph.at(i).second.topRightCorner(3, 1);
      Eigen::Matrix3f rot = m_poseGraph.at(i).second.topLeftCorner(3, 3);

      // f << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";

      // Eigen::Quaternionf currentCameraRotation(rot);

      // f << currentCameraRotation.x() << " " << currentCameraRotation.y() << " "
      //   << currentCameraRotation.z() << " " << currentCameraRotation.w()
      //   << "\n";
      f << rot(0,0) << " " << rot(0, 1) << " " << rot(0,2) << " " << trans(0) << " "
        << rot(1,0) << " " << rot(1, 1) << " " << rot(1,2) << " " << trans(1) << " "
        << rot(2,0) << " " << rot(2, 1) << " " << rot(2,2) << " " << trans(2) << " "
        << "\n";
    }

    f.close();
  }
  void createTextures() {
    m_textures[GPUTexture::RGB] = new GPUTexture(
        Resolution::getInstance().width(), Resolution::getInstance().height(),
        GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);

    m_textures[GPUTexture::DEPTH_RAW] = new GPUTexture(
        Resolution::getInstance().width(), Resolution::getInstance().height(),
        GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);

    m_textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(
        Resolution::getInstance().width(), Resolution::getInstance().height(),
        GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT,
        false, true);

    m_textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(
        Resolution::getInstance().width(), Resolution::getInstance().height(),
        GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT, true);

    m_textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(
        Resolution::getInstance().width(), Resolution::getInstance().height(),
        GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);

    m_textures[GPUTexture::DEPTH_NORM] = new GPUTexture(
        Resolution::getInstance().width(), Resolution::getInstance().height(),
        GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT, true);
  }

  void createCompute() {
    m_computePacks[ComputePack::NORM] = new ComputePack(
        loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"),
        m_textures[GPUTexture::DEPTH_NORM]->texture);

    m_computePacks[ComputePack::FILTER] = new ComputePack(
        loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom"),
        m_textures[GPUTexture::DEPTH_FILTERED]->texture);

    m_computePacks[ComputePack::METRIC] = new ComputePack(
        loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
        m_textures[GPUTexture::DEPTH_METRIC]->texture);

    m_computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(
        loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
        m_textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);
  }

  void createFeedbackBuffers() {
    m_feedbackBuffers[FeedbackBuffer::RAW] =
        new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert",
                                                   "vertex_feedback.geom"));
    m_feedbackBuffers[FeedbackBuffer::FILTERED] =
        new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert",
                                                   "vertex_feedback.geom"));
  }

  void computeFeedbackBuffers(const int& maxDepthProcessed) {
    TICK("feedbackBuffers");
    m_feedbackBuffers[FeedbackBuffer::RAW]->compute(
        m_textures[GPUTexture::RGB]->texture,
        m_textures[GPUTexture::DEPTH_METRIC]->texture, m_tick, m_id,
        maxDepthProcessed);

    m_feedbackBuffers[FeedbackBuffer::FILTERED]->compute(
        m_textures[GPUTexture::RGB]->texture,
        m_textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, m_tick, m_id,
        maxDepthProcessed);
    TOCK("feedbackBuffers");
  }

  const int& id() { return m_id; }

  int& tick() { return m_tick; }

  const int& tick() const { return m_tick; }

  std::map<std::string, GPUTexture*>& textures() { return m_textures; }

  std::map<std::string, ComputePack*>& computePacks() { return m_computePacks; }

  std::map<std::string, FeedbackBuffer*>& feedbackBuffers() {
    return m_feedbackBuffers;
  }

  RGBDOdometry& frameToModel() { return m_frameToModel; }

  RGBDOdometry& modelToModel() { return m_modelToModel; }

  Eigen::Matrix4f& currPose() { return m_currPose; }
  const Eigen::Matrix4f& currPose() const { return m_currPose; }

  std::vector<Deformation::Constraint, Eigen::aligned_allocator<Deformation::Constraint>>& relativeCons() {
    return m_relativeCons;
  }

  IndexMap& indexMap() { return m_indexMap; }

  FillIn& fillIn() { return m_fillIn; }

  // Deformation & localDeformation()
  // {
  //     return m_localDeformation;
  // }

  int& deforms() { return m_deforms; }

  std::vector<PoseMatch, Eigen::aligned_allocator<PoseMatch>>& poseMatches() {
    return m_poseMatches;
  }

  std::vector<std::pair<unsigned long long int, Eigen::Matrix4f>,
              Eigen::aligned_allocator<
                  std::pair<unsigned long long int, Eigen::Matrix4f>>>&
  poseGraph() {
    return m_poseGraph;
  }

  std::vector<unsigned long long int>& poseLogTimes() { return m_poseLogTimes; }

  Img<Eigen::Matrix<unsigned char, 3, 1>>& imgBuff() { return m_imageBuff; }

  Img<Eigen::Vector4f>& consBuff() { return m_consBuff; }

  Img<unsigned short>& timesBuff() { return m_timesBuff; }

  const bool& reloc() { return m_reloc; }

  bool& lost() { return m_lost; }

  bool& lastFrameRecovery() { return m_lastFrameRecovery; }

  int& trackingCount() { return m_trackingCount; }

  const float& maxDepthProcessed() { return m_maxDepthProcessed; }

  bool& rgbOnly() { return m_rgbOnly; }

  std::string filename() { return m_saveFilename; }

  float& weighting() { return m_weighting; }

  bool& trackingOk() { return m_trackingOk; }

  cudaStream_t& stream() { return m_stream; }

  std::vector<KeyFrame*>& miKeyframes() { return m_keyframes; }

  Stats& stats() { return m_stats; }

  KeyFrame* currentKeyFrame() { return m_currentKeyFrame; }

  MutualInformation& mi() { return m_mi; }
  void setCurrentKeyFrame(KeyFrame* newKeyframe) {
    m_currentKeyFrame = newKeyframe;
  }
  std::vector<float>& nidScores() { return m_nid_scores; }
  int& numFused() { return m_num_fused; }
  int& framesSinceLastFusion() { return m_frames_since_last_fusion; };
  MutualInformation& getMI() { return m_mi; }
  float lastKFScore() { return m_nid_scores.size() ? m_nid_scores.back() : 0; }

 private:
  MutualInformation m_mi;
  IndexMap m_indexMap;
  RGBDOdometry m_frameToModel;
  RGBDOdometry m_modelToModel;
  FillIn m_fillIn;
  // Deformation m_localDeformation;
  // Deformation m_localGlobalDeformation; // a global deformation that is local
  // to this map for now

  std::map<std::string, GPUTexture*> m_textures;
  std::map<std::string, ComputePack*> m_computePacks;
  std::map<std::string, FeedbackBuffer*> m_feedbackBuffers;

  Eigen::Matrix4f m_currPose;

  const int m_id;
  int m_tick;

  int m_deforms;

  const int m_consSample;

  std::vector<PoseMatch, Eigen::aligned_allocator<PoseMatch>> m_poseMatches;
  std::vector<Deformation::Constraint, Eigen::aligned_allocator<Deformation::Constraint>> m_relativeCons;

  std::vector<std::pair<unsigned long long int, Eigen::Matrix4f>,
              Eigen::aligned_allocator<
                  std::pair<unsigned long long int, Eigen::Matrix4f>>>
      m_poseGraph;
  std::vector<unsigned long long int> m_poseLogTimes;

  Img<Eigen::Matrix<unsigned char, 3, 1>> m_imageBuff;
  Img<Eigen::Vector4f> m_consBuff;
  Img<unsigned short> m_timesBuff;

  const bool m_iclnuim;

  const bool m_reloc;
  bool m_lost;
  bool m_lastFrameRecovery;
  int m_trackingCount;
  const float m_maxDepthProcessed;

  bool m_rgbOnly;

  bool m_frameToFrameRGB;

  const std::string m_saveFilename;

  float m_weighting;
  bool m_trackingOk;

  cudaStream_t m_stream;

  std::vector<KeyFrame*> m_keyframes;
  KeyFrame* m_currentKeyFrame;
  std::vector<float> m_nid_scores;
  bool m_new_keyframe;
  int m_num_fused;
  int m_frames_since_last_fusion;

  Stats m_stats;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
#endif /*CONTEXT_H_*/