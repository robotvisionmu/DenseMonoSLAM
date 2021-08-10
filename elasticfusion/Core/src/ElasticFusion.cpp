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

#include "ElasticFusion.h"

ElasticFusion::ElasticFusion(
    const int timeDelta, const int countThresh, const float errThresh,
    const float covThresh, const bool closeLoops, const bool iclnuim,
    const bool reloc,
    const float photoThresh, const float confidence, const float depthCut,
    const float icpThresh, const bool fastOdom, const float fernThresh,
    const bool so3, const bool frameToFrameRGB, const std::string fileName,
    const SamplingScheme sampling_scheme, const float nid_threshold,
    const float nidDepthLambda, const int num_bins_depth,
    const int num_bins_img, const int m_nid_pyramid_level)
    : saveFilename(fileName),
      currPose(Eigen::Matrix4f::Identity()),
      nextId(0),
      tick(1),
      timeDelta(timeDelta),
      icpCountThresh(countThresh),
      icpErrThresh(errThresh),
      covThresh(covThresh),
      photoThresh(photoThresh),
      deforms(0),
      fernDeforms(0),
      consSample(20),
      resize(Resolution::getInstance().width(),
             Resolution::getInstance().height(),
             Resolution::getInstance().width() / consSample,
             Resolution::getInstance().height() / consSample),
      poseGraph(Eigen::aligned_allocator<
                std::pair<unsigned long long int, Eigen::Matrix4f>>()),
      closeLoops(closeLoops),
      iclnuim(iclnuim),
      reloc(reloc),
      lost(false),
      lastFrameRecovery(false),
      trackingCount(0),
      maxDepthProcessed(25.0f),
      rgbOnly(false),
      icpWeight(icpThresh),
      pyramid(true),
      fastOdom(fastOdom),
      confidenceThreshold(confidence),
      fernThresh(fernThresh),
      so3(so3),
      frameToFrameRGB(frameToFrameRGB),
      depthCutoff(depthCut),
      sampling_scheme(sampling_scheme),
      nid_threshold(nid_threshold),
      nid_depth_lambda(nidDepthLambda),
      num_bins_depth(num_bins_depth),
      num_bins_img(num_bins_img),
      m_nid_pyramid_level(m_nid_pyramid_level) {
  Stopwatch::getInstance().setCustomSignature(12431231);
}

ElasticFusion::~ElasticFusion() {
  if (iclnuim) {
    savePly(Options::get().outDirectory);
    saveTrajectories(Options::get().outDirectory);
    saveTimes(Options::get().outDirectory);
    saveStats(Options::get().outDirectory);
  }
}

bool ElasticFusion::denseEnough(
    const Img<Eigen::Matrix<unsigned char, 3, 1>>& img) {
  int sum = 0;

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      sum += img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(0) > 0 &&
             img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(1) > 0 &&
             img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(2) > 0;
    }
  }

  return float(sum) / float(img.rows * img.cols) > 0.95f;//0.30f;// 0.75f;
}

void ElasticFusion::processFrame(const std::shared_ptr<unsigned char>& rgb,
                                 const std::shared_ptr<unsigned short>& depth,
                                 const int64_t& timestamp, Context& context,
                                 const Eigen::Matrix4f* inPose,
                                 const Eigen::Matrix4f* orbTcwOld,
                                 const Eigen::Matrix4f* orbTcwNew,
                                 const int cluster, 
                                 const float weightMultiplier,
                                 const bool bootstrap) {
  std::string c("camera" + std::to_string(context.id()));
  // TICK("Run");
  ReferenceFrame& rf = whichReferenceFrame(context);
  context.textures()[GPUTexture::RGB]->texture->Upload(rgb.get(), GL_RGB,
                                                       GL_UNSIGNED_BYTE);
  context.textures()[GPUTexture::DEPTH_RAW]->texture->Upload(
      depth.get(), GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);

  TICK(c + "Preprocess");

  filterDepth(context);
  metriciseDepth(context);

  TOCK(c + "Preprocess");
  TICK(c + "mapping");
  bool fuse = false;
  // First run
  bool firstRun = true;
  for (const auto& c : rf.contexts()) {
    if (c.second->tick() > 1) {
      firstRun = false;
      break;
    }
  }
  if (firstRun) {
    context.computeFeedbackBuffers(maxDepthProcessed);
    Eigen::Matrix4f pose = inPose ? *inPose : Eigen::Matrix4f::Identity();
    rf.globalModel().initialise(
        *context.feedbackBuffers()[FeedbackBuffer::RAW],
        *context.feedbackBuffers()[FeedbackBuffer::FILTERED],
        cluster,
        pose);

    context.frameToModel().initFirstRGB(context.textures()[GPUTexture::RGB]);

    context.miKeyframes().push_back(new KeyFrame(
        *(context.textures()[GPUTexture::RGB]),
        *(context.textures()[GPUTexture::DEPTH_FILTERED]),
        pose/*Eigen::Matrix4f::Identity()*/, depthCutoff,
        Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(),
        Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy()));
    context.setCurrentKeyFrame(context.miKeyframes().back());
    context.numFused() += 1;
    fuse = true;
    context.currPose() = pose;
  } else if (context.tick() == 1) {
    Eigen::Matrix4f pose = inPose ? *inPose : Eigen::Matrix4f::Identity();
    context.currPose() = pose;
    context.frameToModel().initFirstRGB(context.textures()[GPUTexture::RGB]);
  } else {
    Eigen::Matrix4f lastPose = context.currPose();

    bool trackingOk = true;

    TICK(c + "Tracking");
    TICK(c + "autoFill");
    context.currPose() = *inPose;
    predict(context, rf, 0.7);
    resize.image(context.indexMap().imageTex(), context.imgBuff());
    bool shouldFillIn = !denseEnough(context.imgBuff());
    if (Options::get().hybrid_tracking){
      TOCK(c + "autoFill");

      TICK(c + "odomInit");
      // WARNING initICP* must be called before initRGB*
      context.frameToModel().initICPModel(
          shouldFillIn ? &context.fillIn().vertexTexture
                       : context.indexMap().vertexTex(),
          shouldFillIn ? &context.fillIn().normalTexture
                       : context.indexMap().normalTex(),
          maxDepthProcessed, context.currPose());
      context.frameToModel().initRGBModel((shouldFillIn || frameToFrameRGB)
                                              ? &context.fillIn().imageTexture
                                              : context.indexMap().imageTex());

      context.frameToModel().initICP(
          context.textures()[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
      context.frameToModel().initRGB(context.textures()[GPUTexture::RGB]);
      TOCK(c + "odomInit");

      if (bootstrap) {
        assert(inPose);
        context.currPose() = /*context.currPose() **/ (*inPose);
      }

      Eigen::Vector3f trans = context.currPose().topRightCorner(3, 1);
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot =
          context.currPose().topLeftCorner(3, 3);

      TICK(c + "odom");
      context.frameToModel().getIncrementalTransformation(
          trans, rot, context.rgbOnly(),
          Options::get().icp[context.id()],
          pyramid, fastOdom, so3);
      TOCK(c + "odom");

      trackingOk =
          !context.reloc() || context.frameToModel().lastICPError < 1e-04;

      if (context.reloc()) {
        if (!context.lost()) {
          Eigen::MatrixXd covariance = context.frameToModel().getCovariance();

          for (int i = 0; i < 6; i++) {
            if (covariance(i, i) > 1e-04) {
              trackingOk = false;
              break;
            }
          }

          if (!trackingOk) {
            context.trackingCount()++;

            if (context.trackingCount() > 10) {
              context.lost() = true;
            }
          } else {
            context.trackingCount() = 0;
          }
        } else if (context.lastFrameRecovery()) {
          Eigen::MatrixXd covariance = context.frameToModel().getCovariance();

          for (int i = 0; i < 6; i++) {
            if (covariance(i, i) > 1e-04) {
              trackingOk = false;
              break;
            }
          }

          if (trackingOk) {
            context.lost() = false;
            context.trackingCount() = 0;
          }

          context.lastFrameRecovery() = false;
        }
      }

      context.currPose().topRightCorner(3, 1) = trans;
      context.currPose().topLeftCorner(3, 3) = rot;
    } else {
      context.currPose() = *inPose;
    }
    TOCK(c + "Tracking");
    Eigen::Matrix4f diff = context.currPose().inverse() * lastPose;

    Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
    Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

    // Weight by velocity
    float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());

    float largest = 0.01;
    float minWeight = 0.5;

    if (weighting > largest) {
      weighting = largest;
    }

    weighting =
        std::max(1.0f - (weighting / largest), minWeight) * weightMultiplier;

    std::vector<Ferns::SurfaceConstraint, Eigen::aligned_allocator<Ferns::SurfaceConstraint>> constraints;

    TICK(c + "GlobalPredict");
    predict(context, rf);
    TOCK(c + "GlobalPredict");
    Eigen::Matrix4f recoveryPose = context.currPose();

    TICK(c + "Intramap");
    TICK(c + "IntramapGlobal");
    if (false){ // turn off ferns for now //(closeLoops) {
      context.lastFrameRecovery() = false;

      TICK(c + "Ferns::findFrame");
      recoveryPose = rf.ferns().findFrame(
          constraints, context.currPose(), &context.fillIn().vertexTexture,
          &context.fillIn().normalTexture, &context.fillIn().imageTexture,
          context.tick(), context.lost(), 0);
      TOCK(c + "Ferns::findFrame");
    }

    std::vector<float> rawGraph;
    bool orbLoopClosureAccepted = false;
    if (Options::get().hybrid_loops && orbTcwOld && orbTcwNew) {
      Eigen::Matrix4f cp = context.currPose();
      context.currPose() = *orbTcwOld;
      predict(context, rf);
      context.currPose() = cp;

      resize.vertex(context.indexMap().vertexTex(), context.consBuff());
      context.indexMap().combinedPredict(
          *orbTcwNew, rf.globalModel().model(), maxDepthProcessed,
          confidenceThreshold, 0, context.id(), context.tick() - timeDelta,
          timeDelta, IndexMap::INACTIVE);
      resize.time(context.indexMap().oldTimeTex(), context.timesBuff());
      Eigen::Matrix4f estPose = *orbTcwNew;
      for (int i = 0; i < context.consBuff().cols; i++) {
        for (int j = 0; j < context.consBuff().rows; j++) {
          if (context.consBuff().at<Eigen::Vector4f>(j, i)(2) > 0 &&
              context.consBuff().at<Eigen::Vector4f>(j, i)(2) <
                  maxDepthProcessed /*&&
              context.timesBuff().at<unsigned short>(j, i) > 0*/) {
            Eigen::Vector4f worldRawPoint =
                (*orbTcwOld) *
                Eigen::Vector4f(context.consBuff().at<Eigen::Vector4f>(j, i)(0),
                                context.consBuff().at<Eigen::Vector4f>(j, i)(1),
                                context.consBuff().at<Eigen::Vector4f>(j, i)(2),
                                1.0f);

            Eigen::Vector4f worldModelPoint =
                estPose *
                Eigen::Vector4f(context.consBuff().at<Eigen::Vector4f>(j, i)(0),
                                context.consBuff().at<Eigen::Vector4f>(j, i)(1),
                                context.consBuff().at<Eigen::Vector4f>(j, i)(2),
                                1.0f);

            constraints.push_back(
                Ferns::SurfaceConstraint(worldRawPoint, worldModelPoint));
              rf.globalDeformation().addConstraint(
                constraints.back().sourcePoint, constraints.back().targetPoint,
                context.tick(),
                    context.timesBuff().at<unsigned short>(j, i), true);
          }
        }
      } 
 
        for (auto& c : rf.contexts()) 
        {
          for (size_t i = 0; i < c.second->relativeCons().size(); i++) {
            rf.globalDeformation().addConstraint(
                c.second->relativeCons().at(i));
          }
        }

        if(rf.globalDeformation().constrain(rf.ferns().frames, rawGraph,
                                             context.tick(), true,
                                             context.poseGraph(), true)) {
          context.poseMatches().push_back(
              PoseMatch(rf.ferns().frames.size() - 1, rf.ferns().frames.size(),
                        *orbTcwOld, *orbTcwNew, constraints, true));

          fernDeforms += rawGraph.size() > 0;

          orbLoopClosureAccepted = true;
        }
      predict(context, rf);
    }

    bool fernAccepted = false;

    if (false && rf.ferns().lastClosest != -1 && !orbLoopClosureAccepted) {
      if (context.lost()) {
        context.currPose() = recoveryPose;
        context.lastFrameRecovery() = true;
      } else {
        for (size_t i = 0; i < constraints.size(); i++) {
          rf.globalDeformation().addConstraint(
              constraints.at(i).sourcePoint, constraints.at(i).targetPoint,
              context.tick(),
              rf.ferns().frames.at(rf.ferns().lastClosest)->srcTime, true);
        }

        for (auto& c : rf.contexts())  // for(auto & c : m_contexts)
        {
          for (size_t i = 0; i < c.second->relativeCons().size(); i++) {
            rf.globalDeformation().addConstraint(
                c.second->relativeCons().at(i));
          }
        }

        if (rf.globalDeformation().constrain(rf.ferns().frames, rawGraph,
                                             context.tick(), true,
                                             context.poseGraph(), true)) {
          context.currPose() = recoveryPose;

          context.poseMatches().push_back(
              PoseMatch(rf.ferns().lastClosest, rf.ferns().frames.size(),
                        rf.ferns().frames.at(rf.ferns().lastClosest)->pose,
                        context.currPose(), constraints, true));

          fernDeforms += rawGraph.size() > 0;

          fernAccepted = true;
        }
      }
    }
    TOCK(c + "IntramapGlobal");

    TICK(c + "IntramapLocal");
    // If we didn't match to a fern
    if (!context.lost() && closeLoops && rawGraph.size() == 0) {
      // Only predict old view, since we just predicted the current view for the
      // rf.ferns() (which failed!)
      TICK(c + "IndexMap::INACTIVE");
      context.indexMap().combinedPredict(
          context.currPose(), rf.globalModel().model(), maxDepthProcessed,
          confidenceThreshold, 0, context.id(), context.tick() - timeDelta,
          timeDelta, IndexMap::INACTIVE);
      TOCK(c + "IndexMap::INACTIVE");

      // WARNING initICP* must be called before initRGB*
      context.modelToModel().initICPModel(
          context.indexMap().oldVertexTex(), context.indexMap().oldNormalTex(),
          maxDepthProcessed, context.currPose());
      context.modelToModel().initRGBModel(context.indexMap().oldImageTex());

      context.modelToModel().initICP(context.indexMap().vertexTex(),
                                     context.indexMap().normalTex(),
                                     maxDepthProcessed);
      context.modelToModel().initRGB(context.indexMap().imageTex());

      Eigen::Vector3f trans = context.currPose().topRightCorner(3, 1);
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot =
          context.currPose().topLeftCorner(3, 3);

      context.modelToModel().getIncrementalTransformation(
          trans, rot, false, 10, pyramid, fastOdom, false);

      Eigen::MatrixXd covar = context.modelToModel().getCovariance();
      bool covOk = true;

      for (int i = 0; i < 6; i++) {
        if (covar(i, i) > 8e-05/*covThresh*/) {
          covOk = false;
          break;
        }
      }

      Eigen::Matrix4f estPose = Eigen::Matrix4f::Identity();

      estPose.topRightCorner(3, 1) = trans;
      estPose.topLeftCorner(3, 3) = rot;
      if (covOk && context.modelToModel().lastICPCount > 15000/*25000*/ &&
          context.modelToModel().lastICPError < 0.0003/*0.0002*/) {
        resize.vertex(context.indexMap().vertexTex(), context.consBuff());
        resize.time(context.indexMap().oldTimeTex(), context.timesBuff());

        for (int i = 0; i < context.consBuff().cols; i++) {
          for (int j = 0; j < context.consBuff().rows; j++) {
            if (context.consBuff().at<Eigen::Vector4f>(j, i)(2) > 0 &&
                context.consBuff().at<Eigen::Vector4f>(j, i)(2) <
                    maxDepthProcessed &&
                context.timesBuff().at<unsigned short>(j, i) > 0) {
              Eigen::Vector4f worldRawPoint =
                  context.currPose() *
                  Eigen::Vector4f(
                      context.consBuff().at<Eigen::Vector4f>(j, i)(0),
                      context.consBuff().at<Eigen::Vector4f>(j, i)(1),
                      context.consBuff().at<Eigen::Vector4f>(j, i)(2), 1.0f);

              Eigen::Vector4f worldModelPoint =
                  estPose * Eigen::Vector4f(
                                context.consBuff().at<Eigen::Vector4f>(j, i)(0),
                                context.consBuff().at<Eigen::Vector4f>(j, i)(1),
                                context.consBuff().at<Eigen::Vector4f>(j, i)(2),
                                1.0f);

              constraints.push_back(
                  Ferns::SurfaceConstraint(worldRawPoint, worldModelPoint));

              rf.localDeformation().addConstraint(
                  worldRawPoint, worldModelPoint, context.tick(),
                  context.timesBuff().at<unsigned short>(j, i), deforms == 0);
            }
          }
        }

        std::vector<Deformation::Constraint, Eigen::aligned_allocator<Deformation::Constraint>> newRelativeCons;

        if (rf.localDeformation().constrain(
                rf.ferns().frames, rawGraph, context.tick(), false,
                context.poseGraph(), false, &newRelativeCons)) {
          context.poseMatches().push_back(
              PoseMatch(rf.ferns().frames.size() - 1, rf.ferns().frames.size(),
                        estPose, context.currPose(), constraints, false));

          deforms += rawGraph.size() > 0;

          context.currPose() = estPose;

          for (size_t i = 0; i < newRelativeCons.size();
               i += (newRelativeCons.size()) / 3) {
            context.relativeCons().push_back(newRelativeCons.at(i));
          }
        }
      }
    }
    else{TICK(c + "IndexMap::INACTIVE");TOCK(c + "IndexMap::INACTIVE");} // hack to record 0 time
    TOCK(c + "IntramapLocal");
    TOCK(c + "Intramap");

    
    TICK(c + "NID");
    fuse = fuseFrame(context, rawGraph.size() > 0);
    TOCK(c + "NID");

    TICK(c + "fusion");
    if (!context.rgbOnly() && trackingOk && !context.lost() && fuse) {
      TICK(c + "indexMap::prefusion");
      if(!rf.globalModel().isCluster(cluster))
      {
        rf.globalModel().initialise(
          *context.feedbackBuffers()[FeedbackBuffer::RAW],
          *context.feedbackBuffers()[FeedbackBuffer::FILTERED],
          cluster,
          context.currPose());
      }
      context.numFused() += 1;
      context.indexMap().predictIndices(
          context.currPose(), context.tick(), context.id(),
          rf.globalModel().model(), maxDepthProcessed,
          timeDelta + context.framesSinceLastFusion());
      TOCK(c + "indexMap::prefusion");

      TICK(c + "fuse");
      rf.globalModel().fuse(
          context.currPose(), context.tick(), context.id(),
          context.textures()[GPUTexture::RGB],
          context.textures()[GPUTexture::DEPTH_METRIC],
          context.textures()[GPUTexture::DEPTH_METRIC_FILTERED],
          context.indexMap().indexTex(), context.indexMap().vertConfTex(),
          context.indexMap().colorTimeTex(), context.indexMap().normalRadTex(),
          maxDepthProcessed, confidenceThreshold, weighting);
      TOCK(c + "fuse");
      
      TICK(c + "indexMap::postfusion");
      context.indexMap().predictIndices(
          context.currPose(), context.tick(), context.id(),
          rf.globalModel().model(), maxDepthProcessed,
          timeDelta + context.framesSinceLastFusion());
      TOCK(c + "indexMap::postfusion");

      // If we're deforming we need to predict the depth again to figure out
      // which
      // points to update the timestamp's of, since a deformation means a second
      // pose update
      // this loop
      if (rawGraph.size() > 0 && !fernAccepted && !orbLoopClosureAccepted) {
        //TICK("DepthPredict");
        context.indexMap().synthesizeDepth(
            context.currPose(), rf.globalModel().model(), maxDepthProcessed,
            confidenceThreshold, context.tick(), context.id(),
            context.tick() - (timeDelta + context.framesSinceLastFusion()),
            std::numeric_limits<unsigned short>::max());
        //TOCK("DepthPredict");
      }

      TICK(c  + "clean");
      rf.globalModel().clean(
          context.currPose(), context.tick(), context.id(),
          context.indexMap().indexTex(), context.indexMap().vertConfTex(),
          context.indexMap().colorTimeTex(), context.indexMap().normalRadTex(),
          context.indexMap().depthTex(), confidenceThreshold, rawGraph,
          timeDelta + context.framesSinceLastFusion(), maxDepthProcessed,
          fernAccepted || orbLoopClosureAccepted);
    }
    TOCK(c + "clean");
    context.framesSinceLastFusion() =
        fuse ? 0 : context.framesSinceLastFusion() + 1;
  }
  TOCK(c + "fusion");
  
  context.poseGraph().push_back(
      std::pair<unsigned long long int, Eigen::Matrix4f>(context.tick(),
                                                         context.currPose()));
  context.poseLogTimes().push_back(timestamp);

  TICK("sampleGraph");

  rf.localDeformation().sampleGraphModel(rf.globalModel().model(),
                                         context.id(), Options::get().defGraphSampleRate);

  rf.globalDeformation().sampleGraphFrom(rf.localDeformation());

  TOCK("sampleGraph");

  TICK(c + "finalPredict");
  predict(context, rf);
  TOCK(c + "finalPredict");
  if (!context.lost()) {//turn off ferns for now
    //processFerns(context, rf);
    context.tick()++;
  }

  TOCK(c + "mapping");

  std::string im(c + "Intermap");
  TICK(im);
  if (false){ //make sure//(Options::get().interMap) {
    for (auto& kv : m_contextToReferenceFrameMap) {
      if (kv.second != m_contextToReferenceFrameMap[context.id()]) {
        std::vector<Ferns::SurfaceConstraint, Eigen::aligned_allocator<Ferns::SurfaceConstraint>> constraints;
        Eigen::Matrix4f relativeTransform;
        bool success = kv.second->resolveRelativeTransformationFern(
            constraints, relativeTransform, context.currPose(),
            context.fillIn().vertexTexture, context.fillIn().normalTexture,
            context.fillIn().imageTexture, context.tick(), context.lost(),
            maxDepthProcessed, confidenceThreshold, context.id(), timeDelta,
            context.tick());
        // If the optimisation was successful then do this stuff
        if (success) {
          kv.second->consumeReferenceFrame(rf, relativeTransform);
          for (int i = 0; (size_t)i < m_referenceFrames.size(); i++) {
            if (m_referenceFrames[i] ==
                m_contextToReferenceFrameMap[context.id()]) {
              m_referenceFrames.erase(m_referenceFrames.begin() + i);
              break;
            }
          }
          for (auto& ctx : kv.second->contexts()) {
            m_contextToReferenceFrameMap[ctx.second->id()] = kv.second;
            context.indexMap().synthesizeDepth(
                ctx.second->currPose(), kv.second->globalModel().model(),
                maxDepthProcessed, confidenceThreshold, ctx.second->tick(),
                ctx.second->id(), ctx.second->tick() - timeDelta,
                std::numeric_limits<unsigned short>::max());
          }

          break;
        }
      }
    }
  }
  TOCK(im);

  context.stats().record(0,0,0, surfelCount(), context.numFused(), fuse);

  // TOCK("Run");
}

bool ElasticFusion::fuseFrame(Context& context, bool deforming) {
  TICK("MI::CalcMI");
  bool fuse = false;
  if (sampling_scheme == NONE || deforming) {
    context.nidScores().push_back(0);
    fuse = true;
  } else if (sampling_scheme == NID_KEYFRAMING) {
    KeyFrame* kf = new KeyFrame(
        *context.indexMap().imageTex(), *context.indexMap().vertexTex(),
        *context.indexMap().normalTex(), *context.indexMap().oldImageTex(),
        *context.indexMap().oldVertexTex(), *context.indexMap().oldNormalTex(),
        context.currPose(), context.maxDepthProcessed());
    TICK("NIDImg");
    float nid_img =
        context.mi().nidImg(context.frameToModel().nextImg(m_nid_pyramid_level),
                            *kf, context.currPose(), Options::get().nidPyramidLevel, false);
    TOCK("NIDImg");
    TICK("NIDDepth");
    float nid_depth = context.mi().nidDepth(
        context.frameToModel().nextD(m_nid_pyramid_level), *kf,
        context.currPose(), context.maxDepthProcessed(), Options::get().nidPyramidLevel, false);
    TOCK("NIDDepth");
    float nid_score =
        (nid_depth_lambda * nid_depth) + ((1.0f - nid_depth_lambda) * nid_img);
    context.nidScores().push_back(nid_score);

    if (nid_score > nid_threshold) {
      context.currentKeyFrame()->freemem();
      context.setCurrentKeyFrame(kf);
      context.miKeyframes().push_back(new KeyFrame(context.currPose()));
      fuse = true;
    } else {
      delete kf;
      fuse = false;
    }
  }
  TOCK("MI::CalcMI");
  return fuse;
}

void ElasticFusion::processFerns(Context& context, ReferenceFrame& rf) {
  TICK("Ferns::addFrame");
  rf.ferns().addFrame(&context.fillIn().imageTexture,
                      &context.fillIn().vertexTexture,
                      &context.fillIn().normalTexture, context.currPose(),
                      context.tick(), fernThresh);
  TOCK("Ferns::addFrame");
}

void ElasticFusion::predict(Context& context, ReferenceFrame& rf) {
  TICK("IndexMap::ACTIVE");

  if (context.lastFrameRecovery()) {
    context.indexMap().combinedPredict(
        context.currPose(), rf.globalModel().model(), maxDepthProcessed,
        confidenceThreshold, 0, context.id(), context.tick(), timeDelta,
        IndexMap::ACTIVE);
  } else {
    context.indexMap().combinedPredict(
        context.currPose(), rf.globalModel().model(), maxDepthProcessed,
        confidenceThreshold, context.tick(), context.id(), context.tick(),
        timeDelta, IndexMap::ACTIVE);
  }

  TICK("FillIn");
  context.fillIn().vertex(context.indexMap().vertexTex(),
                          context.textures()[GPUTexture::DEPTH_FILTERED],
                          context.lost());
  context.fillIn().normal(context.indexMap().normalTex(),
                          context.textures()[GPUTexture::DEPTH_FILTERED],
                          context.lost());
  context.fillIn().image(context.indexMap().imageTex(),
                         context.textures()[GPUTexture::RGB],
                         context.lost() || frameToFrameRGB);
  TOCK("FillIn");

  TOCK("IndexMap::ACTIVE");
}

void ElasticFusion::predict(Context& context, ReferenceFrame& rf, float confidence) {
  TICK("IndexMap::ACTIVE");

  if (context.lastFrameRecovery()) {
    context.indexMap().combinedPredict(
        context.currPose(), rf.globalModel().model(), maxDepthProcessed,
        confidence, 0, context.id(), context.tick(), timeDelta,
        IndexMap::ACTIVE);
  } else {
    context.indexMap().combinedPredict(
        context.currPose(), rf.globalModel().model(), maxDepthProcessed,
        confidence, context.tick(), context.id(), context.tick(),
        timeDelta, IndexMap::ACTIVE);
  }

  TICK("FillIn");
  context.fillIn().vertex(context.indexMap().vertexTex(),
                          context.textures()[GPUTexture::DEPTH_FILTERED],
                          context.lost());
  context.fillIn().normal(context.indexMap().normalTex(),
                          context.textures()[GPUTexture::DEPTH_FILTERED],
                          context.lost());
  context.fillIn().image(context.indexMap().imageTex(),
                         context.textures()[GPUTexture::RGB],
                         context.lost() || frameToFrameRGB);
  TOCK("FillIn");

  TOCK("IndexMap::ACTIVE");
}

void ElasticFusion::metriciseDepth(Context& context) {
  std::vector<Uniform, Eigen::aligned_allocator<Uniform>> uniforms;

  uniforms.push_back(Uniform("maxD", depthCutoff));

  context.computePacks()[ComputePack::METRIC]->compute(
      context.textures()[GPUTexture::DEPTH_RAW]->texture, &uniforms);
  context.computePacks()[ComputePack::METRIC_FILTERED]->compute(
      context.textures()[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);
}

void ElasticFusion::filterDepth(Context& context) {
  std::vector<Uniform, Eigen::aligned_allocator<Uniform>> uniforms;

  uniforms.push_back(Uniform("cols", (float)Resolution::getInstance().cols()));
  uniforms.push_back(Uniform("rows", (float)Resolution::getInstance().rows()));
  uniforms.push_back(Uniform("maxD", depthCutoff));

  context.computePacks()[ComputePack::FILTER]->compute(
      context.textures()[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void ElasticFusion::normaliseDepth(Context& context, const float& minVal,
                                   const float& maxVal) {
  std::vector<Uniform, Eigen::aligned_allocator<Uniform>> uniforms;

  uniforms.push_back(Uniform("maxVal", maxVal * 1000.f));
  uniforms.push_back(Uniform("minVal", minVal * 1000.f));

  context.computePacks()[ComputePack::NORM]->compute(
      context.textures()[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void ElasticFusion::savePly(std::string dir) {
  int j = 1;
  for (auto& rf : referenceFrames()) {
    std::string filename = dir;
    filename.append(saveFilename);
    std::string refName = rf->name.substr(rf->name.find_last_of("/")+1);
    filename.append("." + std::to_string(j++) + "." + refName +".ply");
    std::cout << "saving " << saveFilename << std::endl;
    std::cout << "reference frame " << rf->name << std::endl;
    // Open file
    std::ofstream fs;
    fs.open(filename.c_str());

    float* mapData = rf->globalModel().downloadMap();

    int validCount = 0;

    for (unsigned int i = 0; i < rf->globalModel().lastCount(); i++) {
      float con = mapData[(i * (3 * 4 + Vertex::MAX_SENSORS)) + 3];

      if (con > confidenceThreshold) {
        validCount++;
      }
    }

    // Write header
    fs << "ply";
    fs << "\nformat "
       << "binary_little_endian"
       << " 1.0";

    // Vertices
    fs << "\nelement vertex " << validCount;
    fs << "\nproperty float x"
          "\nproperty float y"
          "\nproperty float z";

    fs << "\nproperty uchar red"
          "\nproperty uchar green"
          "\nproperty uchar blue";

    fs << "\nproperty float nx"
          "\nproperty float ny"
          "\nproperty float nz";

    fs << "\nproperty float radius";

    fs << "\nend_header\n";

    // Close the file
    fs.close();

    // Open file in binary appendable
    std::ofstream fpout(filename.c_str(), std::ios::app | std::ios::binary);

    for (unsigned int i = 0; i < rf->globalModel().lastCount(); i++) {
      int vertexOffset = i * (3 * 4 + Vertex::MAX_SENSORS);
      Eigen::Vector4f pos(mapData[vertexOffset + 0], mapData[vertexOffset + 1],
                          mapData[vertexOffset + 2], mapData[vertexOffset + 3]);

      if (pos[3] > confidenceThreshold) {
        Eigen::Vector4f col(
            mapData[vertexOffset + 4 + 0], mapData[vertexOffset + 4 + 1],
            mapData[vertexOffset + 4 + 2], mapData[vertexOffset + 4 + 3]);
        Eigen::Vector4f nor(
            mapData[vertexOffset + 18 + 0], mapData[vertexOffset + 18 + 1],
            mapData[vertexOffset + 18 + 2], mapData[vertexOffset + 18 + 3]);

        nor[0] *= -1;
        nor[1] *= -1;
        nor[2] *= -1;

        float value;
        memcpy(&value, &pos[0], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &pos[1], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &pos[2], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        unsigned char r = int(col[0]) >> 16 & 0xFF;
        unsigned char g = int(col[0]) >> 8 & 0xFF;
        unsigned char b = int(col[0]) & 0xFF;

        fpout.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
        fpout.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
        fpout.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));

        memcpy(&value, &nor[0], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &nor[1], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &nor[2], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &nor[3], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));
      }
    }

    // Close file
    fs.close();

    delete[] mapData;
  }
}

void ElasticFusion::saveStats(std::string dir) {
  for (auto& ctx : contexts()) {
    ctx->saveStats(dir);
  }
}

void ElasticFusion::saveTimes(std::string dir) {
  std::string fname = dir;
  fname.append(saveFilename);

  fname.append(".timings");
  std::cout << "saving session timings: " << fname << "\n";

  std::vector<std::string> names;

  for(auto & ctx : contexts()){
    std::string c = "camera" + std::to_string(ctx->id());
    names.push_back(c);
    names.push_back(c + "OrbTracking");
    names.push_back(c + "DepthPrediction");
    names.push_back(c + "mapping");
    names.push_back("DepthPredict::Load");
    names.push_back("DepthPredict::Unload");
    names.push_back("DepthPredict::inference");
    // names.push_back(c + "fuse");
    // names.push_back(c + "clean");
    // names.push_back(c + "IndexMap::INACTIVE");//old view for local loops
    // names.push_back(c + "indexMap::prefusion");//prefusion predict
    // names.push_back(c + "indexMap::postfusion");//pre clean predict
    // //names.push_back("DepthPredict");//new depth prediction
    names.push_back(c + "finalPredict");// final global prediction
    names.push_back(c + "GlobalPredict");//global loop closure prediction
    names.push_back(c + "IntramapGlobal");
    names.push_back(c + "IntramapLocal");
    names.push_back(c + "Intermap");
    //names.push_back(c + "NID");
    names.push_back(c + "fusion");
  }
  Stopwatch::getInstance().writeToFile(names, fname);
}

void ElasticFusion::saveTrajectories(std::string dir) {
  for (auto rf : referenceFrames()) {
    for (auto& nc : rf->contexts()) {
      nc.second->saveTrajectory(dir);
    }
  }
}

Eigen::Vector3f ElasticFusion::rodrigues2(const Eigen::Matrix3f& matrix) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(
      matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

  double rx = R(2, 1) - R(1, 2);
  double ry = R(0, 2) - R(2, 0);
  double rz = R(1, 0) - R(0, 1);

  double s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
  double c = (R.trace() - 1) * 0.5;
  c = c > 1. ? 1. : c < -1. ? -1. : c;

  double theta = acos(c);

  if (s < 1e-5) {
    double t;

    if (c > 0)
      rx = ry = rz = 0;
    else {
      t = (R(0, 0) + 1) * 0.5;
      rx = sqrt(std::max(t, 0.0));
      t = (R(1, 1) + 1) * 0.5;
      ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
      t = (R(2, 2) + 1) * 0.5;
      rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

      if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) &&
          (R(1, 2) > 0) != (ry * rz > 0))
        rz = -rz;
      theta /= sqrt(rx * rx + ry * ry + rz * rz);
      rx *= theta;
      ry *= theta;
      rz *= theta;
    }
  } else {
    double vth = 1 / (2 * s);
    vth *= theta;
    rx *= vth;
    ry *= vth;
    rz *= vth;
  }
  return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

// Sad times ahead
/*IndexMap & ElasticFusion::getIndexMap()
{
    return indexMap;
}*/

GlobalModel& ElasticFusion::getGlobalModel(Context& ctx) {
  return whichReferenceFrame(ctx).globalModel();
}

Ferns& ElasticFusion::getFerns(Context& ctx) {
  return whichReferenceFrame(ctx).ferns();
}

Deformation& ElasticFusion::getLocalDeformation(Context& ctx) {
  return whichReferenceFrame(ctx).localDeformation();  // localDeformation;
}

/*std::map<std::string, GPUTexture*> & ElasticFusion::getTextures()
{
    return textures;
}*/

const std::vector<PoseMatch,Eigen::aligned_allocator<PoseMatch>>& ElasticFusion::getPoseMatches() {
  return poseMatches;
}

/*const RGBDOdometry & ElasticFusion::getModelToModel()
{
    return modelToModel;
}
*/
const float& ElasticFusion::getConfidenceThreshold() {
  return confidenceThreshold;
}

void ElasticFusion::setRgbOnly(const bool& val) { rgbOnly = val; }

void ElasticFusion::setIcpWeight(const float& val) { icpWeight = val; }

void ElasticFusion::setPyramid(const bool& val) { pyramid = val; }

void ElasticFusion::setFastOdom(const bool& val) { fastOdom = val; }

void ElasticFusion::setSo3(const bool& val) { so3 = val; }

void ElasticFusion::setFrameToFrameRGB(const bool& val) {
  frameToFrameRGB = val;
}

void ElasticFusion::setConfidenceThreshold(const float& val) {
  confidenceThreshold = val;
}

void ElasticFusion::setFernThresh(const float& val) { fernThresh = val; }

void ElasticFusion::setDepthCutoff(const float& val) { depthCutoff = val; }

const bool& ElasticFusion::getLost()  // lel
{
  return lost;
}

const int& ElasticFusion::getTick() { return tick; }

const int& ElasticFusion::getTimeDelta() { return timeDelta; }

void ElasticFusion::setTick(const int& val) { tick = val; }

const float& ElasticFusion::getMaxDepthProcessed() { return maxDepthProcessed; }

const Eigen::Matrix4f& ElasticFusion::getCurrPose() { return currPose; }

const int& ElasticFusion::getDeforms() { return deforms; }

const int& ElasticFusion::getFernDeforms() { return fernDeforms; }

/*std::map<std::string, FeedbackBuffer*> & ElasticFusion::getFeedbackBuffers()
{
    return feedbackBuffers;
}*/

std::shared_ptr<Context> ElasticFusion::frontend(std::string name) {
  for (const auto& rf : referenceFrames()) {
    auto ctx = rf->contexts().find(name);

    if (ctx != rf->contexts().end()) return ctx->second;
  }

  std::shared_ptr<ReferenceFrame> rf(new ReferenceFrame());
  rf->name = name;
  std::shared_ptr<Context> ctx(new Context(nextId++, num_bins_depth,
                                           num_bins_img, name, iclnuim, reloc));
  m_referenceFrames.push_back(rf);
  m_contextToReferenceFrameMap[ctx->id()] = rf;
  return rf->contexts()
      .insert(std::pair<std::string, std::shared_ptr<Context>>(name, ctx))
      .first->second;
}

std::vector<std::shared_ptr<Context>> ElasticFusion::contexts() {
  std::vector<std::shared_ptr<Context>> ctxs;

  for (auto& rf : referenceFrames()) {
    for (auto& kv : rf->contexts()) {
      ctxs.push_back(kv.second);
    }
  }

  return ctxs;
}

std::vector<std::shared_ptr<ReferenceFrame>>& ElasticFusion::referenceFrames() {
  return m_referenceFrames;
}

ReferenceFrame& ElasticFusion::whichReferenceFrame(Context& ctx) {
  return *(m_contextToReferenceFrameMap[ctx.id()]);
}

int ElasticFusion::surfelCount()
{
  int count = 0;
  for(const auto & rf : referenceFrames())
  {
    count += rf->globalModel().lastCount();
  }

  return count;
}

void ElasticFusion::batchAlign(Context& context) {
  // for (auto& kv : m_contextToReferenceFrameMap) {
  //   if (kv.second != m_contextToReferenceFrameMap[context.id()]) {
  //     Eigen::Matrix4f relativeTransform;

  //     bool success = kv.second->resolveRelativeTransformationFGR(
  //         relativeTransform, *(m_contextToReferenceFrameMap[context.id()]));

  //     if (success) {
  //       kv.second->consumeReferenceFrame(
  //           *(m_contextToReferenceFrameMap[context.id()]), relativeTransform);
  //       for (int i = 0; (size_t)i < m_referenceFrames.size(); i++) {
  //         if (m_referenceFrames[i] ==
  //             m_contextToReferenceFrameMap[context.id()]) {
  //           m_referenceFrames.erase(m_referenceFrames.begin() + i);
  //           break;
  //         }
  //       }

  //       for (auto& ctx : kv.second->contexts()) {
  //         m_contextToReferenceFrameMap[ctx.second->id()] = kv.second;
  //       }

  //       break;
  //     }
  //   }
  // }
}


void ElasticFusion::applyGlobalLoop(Context & context, Eigen::Matrix4f & orbTcwOld, Eigen::Matrix4f &orbTcwNew)
{
  ReferenceFrame& rf = whichReferenceFrame(context);
  std::vector<float> rawGraph;
  bool orbLoopClosureAccepted = false;
  std::vector<Ferns::SurfaceConstraint, Eigen::aligned_allocator<Ferns::SurfaceConstraint>> constraints;
  std::cout << "ORB LOOP CLOSURE " << std::endl;
  std::cout << "OLD POSE: " << orbTcwOld << std::endl;
  std::cout << "New POSE: " << orbTcwNew << std::endl;

  Eigen::Matrix4f cp = context.currPose();
  context.currPose() = orbTcwNew;
  predict(context, rf);
  context.currPose() = cp;

  resize.vertex(context.indexMap().vertexTex(), context.consBuff());
  context.indexMap().combinedPredict(
      orbTcwOld, rf.globalModel().model(), maxDepthProcessed,
      confidenceThreshold, 0, context.id(), context.tick() - timeDelta,
      timeDelta, IndexMap::INACTIVE);
  resize.time(context.indexMap().oldTimeTex(), context.timesBuff());
  Eigen::Matrix4f estPose = orbTcwNew;
  for (int i = 0; i < context.consBuff().cols; i++) {
    for (int j = 0; j < context.consBuff().rows; j++) {
      if (context.consBuff().at<Eigen::Vector4f>(j, i)(2) > 0 &&
          context.consBuff().at<Eigen::Vector4f>(j, i)(2) <
              maxDepthProcessed &&
          context.timesBuff().at<unsigned short>(j, i) > 0) {
        Eigen::Vector4f worldRawPoint =
            (orbTcwOld) *
            Eigen::Vector4f(context.consBuff().at<Eigen::Vector4f>(j, i)(0),
                            context.consBuff().at<Eigen::Vector4f>(j, i)(1),
                            context.consBuff().at<Eigen::Vector4f>(j, i)(2),
                            1.0f);

        Eigen::Vector4f worldModelPoint =
            estPose *
            Eigen::Vector4f(context.consBuff().at<Eigen::Vector4f>(j, i)(0),
                            context.consBuff().at<Eigen::Vector4f>(j, i)(1),
                            context.consBuff().at<Eigen::Vector4f>(j, i)(2),
                            1.0f);

        constraints.push_back(
            Ferns::SurfaceConstraint(worldRawPoint, worldModelPoint));
        // if((j * context.consBuff().cols) + i % 10 == 0)
          rf.globalDeformation().addConstraint(
            constraints.back().sourcePoint, constraints.back().targetPoint,
            context.tick(),
                context.timesBuff().at<unsigned short>(j, i), true);
      }
    }
  } 
 
  for (auto& c : rf.contexts())  // for(auto & c : m_contexts)
  {
    for (size_t i = 0; i < c.second->relativeCons().size(); i++) {
      rf.globalDeformation().addConstraint(
          c.second->relativeCons().at(i));
    }
  }

  if(rf.globalDeformation().constrain(rf.ferns().frames, rawGraph,
                                        context.tick(), true,
                                        context.poseGraph(), true)) {
    std::cout << "POSE MATCH ACCEPTED: " << std::endl;

    // for (size_t i = 0; i < newRelativeCons.size();
    //      i += (newRelativeCons.size()) / 3) {
    //   context.relativeCons().push_back(newRelativeCons.at(i));
    // }

    context.poseMatches().push_back(
        PoseMatch(rf.ferns().frames.size() - 1, rf.ferns().frames.size(),
                  orbTcwOld, orbTcwNew, constraints, true));

    fernDeforms += rawGraph.size() > 0;

    orbLoopClosureAccepted = true;
  }
  predict(context, rf);
  context.indexMap().predictIndices(
          context.currPose(), context.tick(), context.id(),
          rf.globalModel().model(), maxDepthProcessed,
          timeDelta + context.framesSinceLastFusion());

   rf.globalModel().clean(
          context.currPose(), context.tick(), context.id(),
          context.indexMap().indexTex(), context.indexMap().vertConfTex(),
          context.indexMap().colorTimeTex(), context.indexMap().normalRadTex(),
          context.indexMap().depthTex(), confidenceThreshold, rawGraph,
          timeDelta + context.framesSinceLastFusion(), maxDepthProcessed,
          orbLoopClosureAccepted);
}