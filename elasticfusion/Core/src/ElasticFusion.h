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

#ifndef ELASTICFUSION_H_
#define ELASTICFUSION_H_

#include "Context.h"
#include "Defines.h"
#include "Deformation.h"
#include "Ferns.h"
#include "GlobalModel.h"
#include "IndexMap.h"
#include "MutualInformation.h"
#include "PoseMatch.h"
#include "ReferenceFrame.h"
#include "Shaders/ComputePack.h"
#include "Shaders/FeedbackBuffer.h"
#include "Shaders/FillIn.h"
#include "Shaders/Shaders.h"
#include "Utils/Intrinsics.h"
#include "Utils/RGBDOdometry.h"
#include "Utils/Resolution.h"
#include "Utils/Stopwatch.h"

#include "KeyFrame.h"

#include <pangolin/gl/glcuda.h>
#include <iomanip>
#include <memory>
#include <thread>

#include <Eigen/StdVector>

class ElasticFusion {
 public:
  enum SamplingScheme {
    NONE,
    NID_KEYFRAMING,
    TRANSLATION,
    ROTATION,
    TRANSROT,
    TRACKIN_RES,
    MODULO,
    UNIFORM,
  };

  EFUSION_API ElasticFusion(
      const int timeDelta = 200, const int countThresh = 35000,
      const float errThresh = 5e-05, const float covThresh = 1e-05,
      const bool closeLoops = true, const bool iclnuim = false,
      const bool reloc = false, const float photoThresh = 115,
      const float confidence = 10, const float depthCut = 3,
      const float icpThresh = 10, const bool fastOdom = false,
      const float fernThresh = 0.3095, const bool so3 = true,
      const bool frameToFrameRGB = false, const std::string fileName = "",
      const SamplingScheme sampling_scheme = NID_KEYFRAMING,
      const float nid_threshold = 0.80f, const float nidDepthLambda = 0.7f,
      const int num_bins_depth = 500, const int num_bins_img = 64,
      const int m_nid_pyramid_level = 0);

  virtual ~ElasticFusion();

  /**
   * Process an rgb/depth map pair
   * @param rgb unsigned char row major order
   * @param depth unsigned short z-depth in millimeters, invalid depths are 0
   * @param timestamp nanoseconds (actually only used for the output poses, not
   * important otherwise)
   * @param context
   * @param inPose optional input SE3 pose (if provided, we don't attempt to
   * perform tracking)
   * @param weightMultiplier optional full frame fusion weight
   * @param bootstrap if true, use inPose as a pose guess rather than
   * replacement
   */
  EFUSION_API void processFrame(const std::shared_ptr<unsigned char> &rgb,
                                const std::shared_ptr<unsigned short> &depth,
                                const int64_t &timestamp, Context &context,
                                const Eigen::Matrix4f *inPose = 0,
                                const Eigen::Matrix4f* orbTcwOld = 0,
                                const Eigen::Matrix4f* orbTcwNew = 0,
                                const int cluster = 0,
                                const float weightMultiplier = 1.f,
                                const bool bootstrap = false);

  /**
   * Predicts the current view of the scene, updates the
   * [vertex/normal/image]Tex() members
   * of the indexMap class
   */
  EFUSION_API void predict(Context &context, ReferenceFrame &rf);
  EFUSION_API void predict(Context &context, ReferenceFrame &rf, float confidence);

 EFUSION_API void applyGlobalLoop(Context & context, Eigen::Matrix4f & orbTcwOld, Eigen::Matrix4f &orbTcwNew);
// EFUSION_API void predict(Context& context, ReferenceFrame& rf, float confidence);
  /**
   * This class contains all of the predicted renders
   * @return reference
   */
  EFUSION_API IndexMap &getIndexMap();

  /**
   * This class contains the surfel map
   * @return
   */
  EFUSION_API GlobalModel &getGlobalModel(Context &ctx);

  /**
   * This class contains the fern keyframe database
   * @return
   */
  EFUSION_API Ferns &getFerns(Context &ctx);

  /**
   * This class contains the local deformation graph
   * @return
   */
  EFUSION_API Deformation &getLocalDeformation(Context &ctx);

  /**
   * This is the map of raw input textures (you can display these)
   * @return
   */
  EFUSION_API std::map<std::string, GPUTexture *> &getTextures();

  /**
   * This is the list of deformation constraints
   * @return
   */
  EFUSION_API const std::vector<PoseMatch, Eigen::aligned_allocator<PoseMatch>> &getPoseMatches();

  /**
   * This is the tracking class, if you want access
   * @return
   */
  EFUSION_API const RGBDOdometry &getModelToModel();

  /**
   * The point fusion confidence threshold
   * @return
   */
  EFUSION_API const float &getConfidenceThreshold();

  /**
   * If you set this to true we just do 2.5D RGB-only Lucas–Kanade tracking (no
   * fusion)
   * @param val
   */
  EFUSION_API void setRgbOnly(const bool &val);

  /**
   * Weight for ICP in tracking
   * @param val if 100, only use depth for tracking, if 0, only use RGB. Best
   * value is 10
   */
  EFUSION_API void setIcpWeight(const float &val);

  /**
   * Whether or not to use a pyramid for tracking
   * @param val default is true
   */
  EFUSION_API void setPyramid(const bool &val);

  /**
   * Controls the number of tracking iterations
   * @param val default is false
   */
  EFUSION_API void setFastOdom(const bool &val);

  /**
   * Turns on or off SO(3) alignment bootstrapping
   * @param val
   */
  EFUSION_API void setSo3(const bool &val);

  /**
   * Turns on or off frame to frame tracking for RGB
   * @param val
   */
  EFUSION_API void setFrameToFrameRGB(const bool &val);

  /**
   * Raw data fusion confidence threshold
   * @param val default value is 10, but you can play around with this
   */
  EFUSION_API void setConfidenceThreshold(const float &val);

  /**
   * Threshold for sampling new keyframes
   * @param val default is some magic value, change at your own risk
   */
  EFUSION_API void setFernThresh(const float &val);

  /**
   * Cut raw depth input off at this point
   * @param val default is 3 meters
   */
  EFUSION_API void setDepthCutoff(const float &val);

  /**
   * Returns whether or not the camera is lost, if relocalisation mode is on
   * @return
   */
  EFUSION_API const bool &getLost();

  /**
   * Get the internal clock value of the fusion process
   * @return monotonically increasing integer value (not real-world time)
   */
  EFUSION_API const int &getTick();

  /**
   * Get the time window length for model matching
   * @return
   */
  EFUSION_API const int &getTimeDelta();

  /**
   * Cheat the clock, only useful for multisession/log fast forwarding
   * @param val control time itself!
   */
  EFUSION_API void setTick(const int &val);

  /**
   * Internal maximum depth processed, this is defaulted to 20 (for rescaling
   * depth buffers)
   * @return
   */
  EFUSION_API const float &getMaxDepthProcessed();

  /**
   * The current global camera pose estimate
   * @return SE3 pose
   */
  EFUSION_API const Eigen::Matrix4f &getCurrPose();

  /**
   * The number of local deformations that have occurred
   * @return
   */
  EFUSION_API const int &getDeforms();

  /**
   * The number of global deformations that have occured
   * @return
   */
  EFUSION_API const int &getFernDeforms();

  /**
   * These are the vertex buffers computed from the raw input data
   * @return can be rendered
   */
  EFUSION_API std::map<std::string, FeedbackBuffer *> &getFeedbackBuffers();

  /**
   * Calculate the above for the current frame (only done on the first frame
   * normally)
   */
  EFUSION_API void computeFeedbackBuffers();

  /**
   * Saves out a .ply mesh file of the current model
   */
  EFUSION_API void savePly(std::string dir);

  EFUSION_API void saveStats(std::string dir);

  EFUSION_API void saveTimes(std::string dir);

  /**
   * saves out the trajectory (.freiburg) of each sensor.
   *
   */
  EFUSION_API void saveTrajectories(std::string dir);

  /**
   * Renders a normalised view of the input raw depth for displaying as an
   * OpenGL texture
   * (this is stored under textures[GPUTexture::DEPTH_NORM]
   * @param minVal minimum depth value to render
   * @param maxVal maximum depth value to render
   */
  EFUSION_API void normaliseDepth(Context &context, const float &minVal,
                                  const float &maxVal);

  /**
   * Gets a Context by name, if it exists. If it doesn't exist a context is
   * created with the
   * given name. For collaborative (multiple camera) ElasticFusion each camera
   * should have its
   * own Context. Contexts created by calling this function are managed by
   * ElasticFusion.
   * In essence, a context represents a SLAM frontend.
   *
   * @param name name of the context to create or retrieve (if it doesn't
   * already exist)
   * @return a reference to context corresponding to @param
   */
  EFUSION_API std::shared_ptr<Context> frontend(std::string name);

  /**
   * Returns a vector containing all the active reference frames. When a
   * new context is created it is assumed to be in it's own reference frame.
   * As inter-map global loop closures occur reference frames are aligned, with
   * one essentially consuming the other.
   *
   */
  EFUSION_API std::vector<std::shared_ptr<ReferenceFrame>> &referenceFrames();

  /**
  */
  EFUSION_API std::vector<std::shared_ptr<Context>> contexts();

  /**
  */
  EFUSION_API ReferenceFrame &whichReferenceFrame(Context &ctx);

  EFUSION_API void batchAlign(Context &context);
  // Here be dragons

  EFUSION_API std::vector<KeyFrame *> miKeyframes(Context &context) {
    return context.miKeyframes();
  }
  EFUSION_API float &nidThreshold() { return nid_threshold; }
  EFUSION_API SamplingScheme &samplingScheme() { return sampling_scheme; }
  EFUSION_API int &numFused(Context &context) { return context.numFused(); }
  EFUSION_API MutualInformation &getMI(Context &context) {
    return context.getMI();
  }
  EFUSION_API float &nidDepthLambda() { return nid_depth_lambda; }
  EFUSION_API void setNumBinsImg(int numBinsImg) {
    for (auto &ctx : contexts()) {
      ctx->getMI().numBinsImg() = numBinsImg;
    }
  }

  EFUSION_API void setNumBinsDepth(int numBinsDepth) {
    for (auto &ctx : contexts()) {
      ctx->getMI().numBinsDepth() = numBinsDepth;
    }
  }
  EFUSION_API float lastKFScore(Context &context) {
    return context.lastKFScore();
  }

  EFUSION_API int &nidPyramidLevel() { return m_nid_pyramid_level; }

  EFUSION_API float kFThreshold() {
    if (sampling_scheme == NONE) {
      return 0;
    } else if (sampling_scheme == NID_KEYFRAMING) {
      return nid_threshold;
    } else if (sampling_scheme == TRANSROT) {
      return 0.0f;
      // } else if ((sampling_scheme == TRANSLATION)) {
      //   return trans_thresh;
      // } else if ((sampling_scheme == ROTATION)) {
      //   return rot_thresh;
      // } else if ((sampling_scheme == UNIFORM)) {
      //   return sample_rate;
      // } else if ((sampling_scheme == TRACKIN_RES)) {
      //   return res_thresh;
      // } else if (sampling_scheme == MODULO) {
      //   return modulo;
    } else {
      return 0;
    }
  }

  /*
    returns the total number of surfels across all maps
  */
  EFUSION_API int surfelCount();
 private:
  // IndexMap indexMap;
  // RGBDOdometry frameToModel;
  // RGBDOdometry modelToModel;
  // GlobalModel globalModel;
  // FillIn fillIn;
  // Ferns ferns;
  // Deformation localDeformation;
  // Deformation globalDeformation;

  const std::string saveFilename;
  // std::map<std::string, GPUTexture*> textures;
  // std::map<std::string, ComputePack*> computePacks;
  // std::map<std::string, FeedbackBuffer*> feedbackBuffers;

  // void createTextures();
  // void createCompute();
  // void createFeedbackBuffers();

  void filterDepth(Context &context);
  void metriciseDepth(Context &context);

  bool denseEnough(const Img<Eigen::Matrix<unsigned char, 3, 1>> &img);

  void processFerns(Context &context, ReferenceFrame &rf);

  Eigen::Vector3f rodrigues2(const Eigen::Matrix3f &matrix);

  bool fuseFrame(Context &context, bool deforming);
  Eigen::Matrix4f currPose;

  int nextId;
  int tick;
  const int timeDelta;
  const int icpCountThresh;
  const float icpErrThresh;
  const float covThresh;
  const float photoThresh;

  int deforms;
  int fernDeforms;
  const int consSample;
  Resize resize;

  std::vector<PoseMatch, Eigen::aligned_allocator<PoseMatch>> poseMatches;
  std::vector<Deformation::Constraint, Eigen::aligned_allocator<Deformation::Constraint>> relativeCons;

  std::vector<std::pair<unsigned long long int, Eigen::Matrix4f>, Eigen::aligned_allocator<std::pair<unsigned long long int, Eigen::Matrix4f>>> poseGraph;
  std::vector<unsigned long long int> poseLogTimes;

  // Img<Eigen::Matrix<unsigned char, 3, 1>> imageBuff;
  // Img<Eigen::Vector4f> consBuff;
  // Img<unsigned short> timesBuff;

  const bool closeLoops;
  const bool iclnuim;

  const bool reloc;
  bool lost;
  bool lastFrameRecovery;
  int trackingCount;
  const float maxDepthProcessed;

  bool rgbOnly;
  float icpWeight;
  bool pyramid;
  bool fastOdom;
  float confidenceThreshold;
  float fernThresh;
  bool so3;
  bool frameToFrameRGB;
  float depthCutoff;

  std::vector<std::shared_ptr<ReferenceFrame>>
      m_referenceFrames; /*All the currently active contexts*/
  std::map<int, std::shared_ptr<ReferenceFrame>>
      m_contextToReferenceFrameMap; /*Map from context id to the reference
                                       frame it belongs to*/

  SamplingScheme sampling_scheme;
  float nid_threshold;
  float nid_depth_lambda;
  int num_bins_depth;
  int num_bins_img;
  int m_nid_pyramid_level;

 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif /* ELASTICFUSION_H_ */
