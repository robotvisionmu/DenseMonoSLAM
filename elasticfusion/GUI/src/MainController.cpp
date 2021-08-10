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

#include "MainController.h"

MainController::MainController(int argc, char *argv[])
    : good(true),
      eFusion(0),
      gui(0),
      // groundTruthOdometry(0),
      groundTruthClusters(0),
      framesToSkip(0),
      resetButton(false),
      resizeStream(0),
      numMaps(0)
      {
  cudaSafeCall(cudaGLSetGLDevice(0));
  Options::get(argc, argv);
  iclnuim = Options::get().iclnuim;

  std::string calibrationFile = Options::get().calibrationFile;

  Resolution::getInstance(1024, 320);  //(1241, 376);//(1024, 320);//(640, 480);

  if (calibrationFile.length()) {
    loadCalibration(calibrationFile);
  } else {
    Intrinsics::getInstance(528, 528, 320, 240);
  }

  numFusing = Options::get().numFusing;
  numCameras = Options::get().numSensors;

  cameraManager = MultiCameraManagerFactory::get();
  std::cout << "waiting for cameras..." << std::endl;

  while (cameraManager->devices().size() < (uint32_t)numCameras)
    ;

  for (int i = 0; i < numCameras; i++) {
    std::cout << cameraManager->devices()[i]->getFile() << " connected."
              << std::endl;
  }

  good = true;

  if (Options::get().posesFiles.size())  //(Parse::get().arg(argc, argv, "-p",
                                         // poseFile) > 0)
  {
    for (const auto pf : Options::get().posesFiles) {
      groundTruthOdometry.push_back(new GroundTruthOdometry(pf));
    }
  }

  if (Options::get().clustersFile.length()) {
    groundTruthClusters = new GroundTruthClusters(Options::get().clustersFile);
  }

  confidence = Options::get().confidence;
  depth = Options::get().depth;
  icp = Options::get().icp[0];
  icpErrThresh = Options::get().icpErrThresh;
  covThresh = Options::get().covThresh;
  photoThresh = Options::get().photoThresh;
  fernThresh = Options::get().fernThresh;

  timeDelta = Options::get().timeDelta;
  icpCountThresh = Options::get().icpCountThresh;
  start = Options::get().start;
  so3 = Options::get().so3;
  end = std::numeric_limits<unsigned short>::max();

  for (int i = 0; i < numCameras; i++) {
    cameraManager->devices()[0]->flipColors = Options::get().flip;
  }

  openLoop = !groundTruthOdometry.size() && Options::get().openLoop;
  reloc = Options::get().reloc;
  frameskip = Options::get().frameskip;
  quiet = Options::get().quiet;
  fastOdom = Options::get().fastOdom;
  rewind = Options::get().rewind;
  frameToFrameRGB = Options::get().frameToFrameRGB;

  std::vector<std::string> cams;
  for (auto &d : cameraManager->devices()) {
    cams.push_back(d->getFile());
  }
  gui = new GUI(Options::get().logfiles.size() == 0, Options::get().sc, cams);

  gui->flipColors->Ref().Set(Options::get().flip);
  gui->rgbOnly->Ref().Set(false);
  gui->pyramid->Ref().Set(true);
  gui->fastOdom->Ref().Set(fastOdom);
  gui->confidenceThreshold->Ref().Set(confidence);
  gui->depthCutoff->Ref().Set(depth);
  gui->icpWeight->Ref().Set(Options::get().icp[0]);
  gui->so3->Ref().Set(so3);
  gui->frameToFrameRGB->Ref().Set(frameToFrameRGB);
  gui->nidThreshold->Ref().Set(Options::get().nidThreshold);
  gui->nidDepthWeight->Ref().Set(Options::get().nidDepthWeight);
  gui->numBinsImg->Ref().Set(Options::get().numBinsImg);
  gui->numBinsDepth->Ref().Set(Options::get().numBinsDepth);
  gui->nidPyramidLevel->Ref().Set(Options::get().nidPyramidLevel);

  resizeStream = new Resize(Resolution::getInstance().width(),
                            Resolution::getInstance().height(),
                            Resolution::getInstance().width() / 2,
                            Resolution::getInstance().height() / 2);

  for (int i = 0; i < numCameras; i++) {
    gui->addCamera(cameraManager->devices()[i]->getFile());
  }

  orb_slam_tracker = new ORB_SLAM3::System(
          Options::get().orb_vocabulary,
          Options::get().orb_config_yaml,
          ORB_SLAM3::System::RGBD, false);

  depth_prediction = new DepthPrediction(Options::get().half_float);
}

MainController::~MainController() {
  if (eFusion) {
    delete eFusion;
  }

  if (gui) {
    delete gui;
  }

  if (groundTruthOdometry.size()) {
    for (const auto gto : groundTruthOdometry) delete gto;
  }

  if (groundTruthClusters) {
    delete groundTruthClusters;
  }

  if (resizeStream) {
    delete resizeStream;
  }

  delete cameraManager;

  if (orb_slam_tracker) {
    delete orb_slam_tracker;
  }
  if(depth_prediction)
  {
    delete depth_prediction;
  }
}

void MainController::loadCalibration(const std::string &filename) {
  std::ifstream file(filename);
  std::string line;

  assert(!file.eof());

  double fx, fy, cx, cy;

  std::getline(file, line);

  int n = sscanf(line.c_str(), "%lg %lg %lg %lg", &fx, &fy, &cx, &cy);

  assert(n == 4 &&
         "Ooops, your calibration file should contain a single line with fx fy "
         "cx cy!");

  Intrinsics::getInstance(fx, fy, cx, cy);
}

void MainController::launch() {
  while (good) {
    if (eFusion) {
      run();
    }

    if (eFusion == 0 || resetButton) {
      resetButton = false;

      if (eFusion) {
        delete eFusion;
      }

      eFusion = new ElasticFusion(
          openLoop ? std::numeric_limits<int>::max() / 2 : timeDelta,
          icpCountThresh, icpErrThresh, covThresh, !openLoop,
          Options::get().iclnuim, reloc, photoThresh, confidence, depth, icp,
          fastOdom, fernThresh, so3, frameToFrameRGB, "model",
          Options::get().noKeyframe
              ? ElasticFusion::SamplingScheme::NONE
              : ElasticFusion::SamplingScheme::NID_KEYFRAMING,
          Options::get().nidThreshold, Options::get().nidDepthWeight,
          Options::get().numBinsDepth, Options::get().numBinsImg,
          Options::get().nidPyramidLevel);  // TODO replace this param with
                                            // something better

      logReaders.clear();
      gui->clearLogReaders();
      cameraManager->reset();

      while (cameraManager->devices().size() < (uint32_t)numCameras)
        ;

      int nf = numFusing;
      for (int i = 0; i < numCameras; i++) {
        cameraManager->devices()[i]->rewind();
        Context &ctx =
            *(eFusion->frontend(cameraManager->devices()[i]->getFile()));

        ctx.rgbOnly() = nf-- > 0 ? false : true;

        logReaders[cameraManager->devices()[i]->getFile()] =
            cameraManager->devices()[i];
      }

      for (int i = 0; i < numCameras; i++) {
        gui->addCamera(cameraManager->devices()[i]->getFile());
      }

      // orb_slam_tracker.Reset();
    } else {
      break;
    }
  }
}

void MainController::run() {
  std::shared_ptr<LogReader> &activeLogReader = logReaders[gui->activeCamera()];
  Context &activeCtx = *(eFusion->frontend(activeLogReader->getFile()));
  cv::Mat im;
  cv::Mat im_rgb;
  cv::Mat im_depth;
  cv::Mat im_d;
  Eigen::Matrix4f orb_pose;
  while (!pangolin::ShouldQuit() &&
         /*!((!logReader->hasMore()) && quiet) &&*/ !(
             eFusion->getTick() == end && quiet)) {
    std::vector<std::shared_ptr<LogReader>> lrs;
    std::vector<Eigen::Matrix4f *> ps;
    std::vector<float> ws;
    std::vector<std::shared_ptr<Context>> cs;
    TICK("overall");
    for (auto lr : logReaders) {
      std::shared_ptr<LogReader> logReader = lr.second;
      std::shared_ptr<Context> ctx = eFusion->frontend(lr.first);

      if (!gui->pause->Get() || pangolin::Pushed(*gui->step)) {
        if (((logReader->hasMore()) || rewind) && eFusion->getTick() < end) {
          TICK("LogRead");
          if (rewind) {
            if (!logReader->hasMore()) {
              logReader->getBack();
            } else {
              logReader->getNext();
            }

            if (logReader->rewound()) {
              logReader->currentFrame = 0;
            }
          } else {
            logReader->getNext();
          }
          TOCK("LogRead");

          if (eFusion->getTick() < start) {
            eFusion->setTick(start);
            logReader->fastForward(start);
          }

          float weightMultiplier = framesToSkip + 1;

          if (framesToSkip > 0) {
            eFusion->setTick(activeCtx.tick() + framesToSkip);
            logReader->fastForward(logReader->currentFrame + framesToSkip);
            framesToSkip = 0;
          }

          Eigen::Matrix4f *currentPose = 0;
          Eigen::Matrix4f *orb_lc_Tcw_old = 0;
          Eigen::Matrix4f *orb_lc_Tcw_new = 0;
          int currentCluster = 0;

          if (groundTruthOdometry.size()) {
            currentPose = new Eigen::Matrix4f;
            currentPose->setIdentity();
            *currentPose =
                groundTruthOdometry[activeCtx.id()]->getTransformation(
                    logReader->timestamp);
          }
          if (groundTruthClusters) {
            currentCluster =
                groundTruthClusters->getCluster(logReader->timestamp);
          }
          std::string c("camera" + std::to_string(ctx->id()));
          TICK(c);
          TICK(c + "DepthPrediction");
          std::shared_ptr<unsigned short> depth_frame;
          if(Options::get().predict_depth)
          {
            depth_prediction->predict(logReader->rgb());
            depth_frame = depth_prediction->depth();
          } 
          else
          {
            depth_frame = logReader->depth();
          }
          TOCK(c + "DepthPrediction");
          if (Options::get().orb_tracking){
            TICK(c + "OrbTracking");
            im_rgb = cv::Mat(Resolution::getInstance().rows(),
                             Resolution::getInstance().cols(), CV_8UC3,
                             logReader->rgb().get());
            im_d = cv::Mat(Resolution::getInstance().rows(),
                           Resolution::getInstance().cols(), CV_16UC1,
                           depth_frame.get());
            double tframe = logReader->timestamp;
            // while (!orb_slam_tracker->localMapper()->AcceptKeyFrames())
            //   usleep(10);
            orb_slam_tracker->TrackRGBD(im_rgb, im_d, tframe);
            cv::Mat p = orb_slam_tracker->GetLastPose();

            if (!p.empty()) {
              orb_pose = Eigen::Matrix4f::Identity();
              orb_pose(0, 0) = p.at<float>(0, 0);
              orb_pose(0, 1) = p.at<float>(0, 1);
              orb_pose(0, 2) = p.at<float>(0, 2);
              orb_pose(0, 3) = p.at<float>(0, 3);
              orb_pose(1, 0) = p.at<float>(1, 0);
              orb_pose(1, 1) = p.at<float>(1, 1);
              orb_pose(1, 2) = p.at<float>(1, 2);
              orb_pose(1, 3) = p.at<float>(1, 3);
              orb_pose(2, 0) = p.at<float>(2, 0);
              orb_pose(2, 1) = p.at<float>(2, 1);
              orb_pose(2, 2) = p.at<float>(2, 2);
              orb_pose(2, 3) = p.at<float>(2, 3); 

              currentPose = new Eigen::Matrix4f();
              currentPose->setIdentity();
              *currentPose = orb_pose;
            }
            std::vector<Eigen::Matrix4d> orb_lc = orb_slam_tracker->loopClosing()->getLoopClosureCandidate();
            if(orb_lc.size())
            {
              orb_lc_Tcw_old = new Eigen::Matrix4f();
              orb_lc_Tcw_new = new Eigen::Matrix4f();
              orb_lc_Tcw_old->setIdentity();
              orb_lc_Tcw_new->setIdentity();
              *orb_lc_Tcw_old = orb_lc.front().cast<float>();
              *orb_lc_Tcw_new = orb_lc.back().cast<float>();
            }
            TOCK(c + "OrbTracking");
          }

          eFusion->processFrame(logReader->rgb(), depth_frame,
                                logReader->timestamp, *ctx, currentPose,
                                orb_lc_Tcw_old, orb_lc_Tcw_new,
                                currentCluster, weightMultiplier,
                                false);  // currentPose ? true : false);
          TOCK(c);
          if (currentPose) {
            delete currentPose;
          }
           if (orb_lc_Tcw_old) {
            delete orb_lc_Tcw_old;
          }
           if (orb_lc_Tcw_new) {
            delete orb_lc_Tcw_new;
          }

          if (frameskip &&
              Stopwatch::getInstance().getTimings().at("Run").back() >
                  1000.f / 30.f) {
            framesToSkip =
                int(Stopwatch::getInstance().getTimings().at("Run").back() /
                    (1000.f / 30.f));
          }
        }
      } else {
        eFusion->predict(*ctx, eFusion->whichReferenceFrame(*ctx));
      }
    }
    TOCK("overall");
    TICK("GUI");

    std::shared_ptr<LogReader> &activeLogReader =
        logReaders[gui->activeCamera()];
    Context &activeCtx = *(eFusion->frontend(activeLogReader->getFile()));

    if ((size_t)numMaps != eFusion->referenceFrames().size())
      gui->createViews(eFusion->referenceFrames());

    numMaps = eFusion->referenceFrames().size();
    gui->preCall();

    if (gui->followPose->Get()) {
      for (auto &rf : eFusion->referenceFrames()) {
        pangolin::OpenGlMatrix mv;

        Context &c =
            rf->contexts().find(activeCtx.filename()) == rf->contexts().end()
                ? *(rf->contexts().begin()->second)
                : activeCtx;
        Eigen::Matrix4f currPose = c.currPose();
        Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

        Eigen::Quaternionf currQuat(currRot);
        Eigen::Vector3f forwardVector(0, 0, 1);
        Eigen::Vector3f upVector(0, Options::get().iclnuim ? 1 : -1, 0);

        Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
        Eigen::Vector3f up = (currQuat * upVector).normalized();

        Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

        eye -= forward;

        Eigen::Vector3f at = eye + forward;

        Eigen::Vector3f z = (eye - at).normalized();   // Forward
        Eigen::Vector3f x = up.cross(z).normalized();  // Right
        Eigen::Vector3f y = z.cross(x);

        Eigen::Matrix4d m;
        m << x(0), x(1), x(2), -(x.dot(eye)), y(0), y(1), y(2), -(y.dot(eye)),
            z(0), z(1), z(2), -(z.dot(eye)), 0, 0, 0, 1;

        memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

        gui->s_cams[c.id()]->SetModelViewMatrix(mv);
      }
    }

    gui->activateView(activeCtx.id());

    std::stringstream stri;
    stri << activeCtx.modelToModel().lastICPCount;
    gui->trackInliers->Ref().Set(stri.str());

    std::stringstream stre;
    stre << (std::isnan(activeCtx.modelToModel().lastICPError)
                 ? 0
                 : activeCtx.modelToModel().lastICPError);
    gui->trackRes->Ref().Set(stre.str());

    if (!gui->pause->Get()) {
      gui->resLog.Log((std::isnan(activeCtx.modelToModel().lastICPError)
                           ? std::numeric_limits<float>::max()
                           : activeCtx.modelToModel().lastICPError),
                      icpErrThresh);
      gui->inLog.Log(activeCtx.modelToModel().lastICPCount, icpCountThresh);
      gui->miLog.Log(eFusion->lastKFScore(activeCtx), eFusion->kFThreshold());
    }

    Eigen::Matrix4f pose = activeCtx.currPose();

    if (gui->drawRawCloud->Get() || gui->drawFilteredCloud->Get()) {
      activeCtx.computeFeedbackBuffers(eFusion->getMaxDepthProcessed());
    }

    if (gui->drawRawCloud->Get()) {
      activeCtx.feedbackBuffers()
          .at(FeedbackBuffer::RAW)
          ->render(gui->s_cams[activeCtx.id()]->GetProjectionModelViewMatrix(),
                   pose, gui->drawNormals->Get(), gui->drawColors->Get());
    }

    if (gui->drawFilteredCloud->Get()) {
      activeCtx.feedbackBuffers()
          .at(FeedbackBuffer::FILTERED)
          ->render(gui->s_cams[activeCtx.id()]->GetProjectionModelViewMatrix(),
                   pose, gui->drawNormals->Get(), gui->drawColors->Get());
    }

    if (gui->drawGlobalModel->Get()) {
      glFinish();
      TICK("Global");

      for (auto &rf : eFusion->referenceFrames()) {
        Context &c =
            rf->contexts().find(activeCtx.filename()) == rf->contexts().end()
                ? *(rf->contexts().begin()->second)
                : activeCtx;
        gui->activateView(c.id());

        if (gui->drawFxaa->Get()) {
          gui->drawFXAA(gui->s_cams[rf->contexts().begin()->second->id()]
                            ->GetProjectionModelViewMatrix(),
                        gui->s_cams[rf->contexts().begin()->second->id()]
                            ->GetModelViewMatrix(),
                        rf->globalModel().model(),
                        eFusion->getConfidenceThreshold(), c.tick(), c.id(),
                        eFusion->getTimeDelta(), Options::get().iclnuim);
        } else {
          std::vector<std::tuple<float, float, float>> colors;
          if (groundTruthClusters)
            for (const auto &c : rf->globalModel().clusters()) {
              colors.push_back(groundTruthClusters->getClusterColor(c));
            }
          rf->globalModel().renderPointCloud(
              gui->s_cams[rf->contexts().begin()->second->id()]
                  ->GetProjectionModelViewMatrix(),
              eFusion->getConfidenceThreshold(), gui->drawUnstable->Get(),
              gui->drawNormals->Get(), gui->drawColors->Get(),
              gui->drawPoints->Get(), gui->drawWindow->Get(),
              gui->drawTimes->Get(),
              gui->drawContributions->Get() && c.id() == activeCtx.id(),
              c.tick(), c.id(), eFusion->getTimeDelta(),
              rf->globalModel().clusters(), groundTruthClusters, colors);
        }
      }

      glFinish();
      TOCK("Global");
    }

    if (activeCtx.lost()) {
      glColor3f(1, 1, 0);
    } else {
      glColor3f(1, 0, 1);
    }

    for (const auto lr : logReaders) {
      if (lr.first == activeLogReader->getFile()) {
        if (eFusion->frontend(lr.first)->lost()) {
          glColor3f(1.0, 0.0, 0.0);
        } else /*localised*/
        {
          glColor3f(0.0, 1.0, 0.0);
        }
      } else /*inactive*/
      {
        if (eFusion->frontend(lr.first)->lost()) {
          glColor3f(0.0, 0.0, 0.0);
        } else /*localised*/
        {
          glColor3f(1.0, 0.0, 1.0);
        }
      }
      gui->activateView(eFusion->frontend(lr.first)->id());
      gui->drawFrustum(eFusion->frontend(lr.first)->currPose());
    }

    glColor3f(1, 1, 1);

    gui->activateView(activeCtx.id());
    if (gui->drawFerns->Get()) {
      glColor3f(0, 0, 0);
      for (size_t i = 0; i < eFusion->getFerns(activeCtx).frames.size(); i++) {
        if ((int)i == eFusion->getFerns(activeCtx).lastClosest) continue;

        gui->drawFrustum(eFusion->getFerns(activeCtx).frames.at(i)->pose);
      }
      glColor3f(1, 1, 1);
    }

    glColor3f(0, 0.8, 0.0);
    for (size_t i = 0; i < eFusion->miKeyframes(activeCtx).size(); i++) {
      gui->drawFrustum(eFusion->miKeyframes(activeCtx).at(i)->pose(), 0.05f);
    }
    glColor3f(1, 1, 1);

    if (gui->drawDefGraph->Get()) {
      const std::vector<GraphNode *> &graph =
          eFusion->whichReferenceFrame(activeCtx).globalDeformation().getGraph();

      for (size_t i = 0; i < graph.size(); i++) {
        pangolin::glDrawCross(graph.at(i)->position(0),
                              graph.at(i)->position(1),
                              graph.at(i)->position(2), 0.1);

        for (size_t j = 0; j < graph.at(i)->neighbours.size(); j++) {
          pangolin::glDrawLine(
              graph.at(i)->position(0), graph.at(i)->position(1),
              graph.at(i)->position(2),
              graph.at(graph.at(i)->neighbours.at(j))->position(0),
              graph.at(graph.at(i)->neighbours.at(j))->position(1),
              graph.at(graph.at(i)->neighbours.at(j))->position(2));
        }
      }
    }

    if (eFusion->getFerns(activeCtx).lastClosest != -1) {
      glColor3f(1, 0, 0);
      gui->drawFrustum(eFusion->getFerns(activeCtx)
                           .frames.at(eFusion->getFerns(activeCtx).lastClosest)
                           ->pose);
      glColor3f(1, 1, 1);
    }

    const std::vector<PoseMatch, Eigen::aligned_allocator<PoseMatch>>
        &poseMatches = activeCtx.poseMatches();

    int maxDiff = 0;
    for (size_t i = 0; i < poseMatches.size(); i++) {
      if (poseMatches.at(i).secondId - poseMatches.at(i).firstId > maxDiff) {
        maxDiff = poseMatches.at(i).secondId - poseMatches.at(i).firstId;
      }
    }

    for (size_t i = 0; i < poseMatches.size(); i++) {
      if (gui->drawDeforms->Get()) {
        if (poseMatches.at(i).fern) {
          glColor3f(1, 0, 0);
        } else {
          glColor3f(0, 1, 0);
        }
        for (size_t j = 0; j < poseMatches.at(i).constraints.size(); j++) {
          pangolin::glDrawLine(
              poseMatches.at(i).constraints.at(j).sourcePoint(0),
              poseMatches.at(i).constraints.at(j).sourcePoint(1),
              poseMatches.at(i).constraints.at(j).sourcePoint(2),
              poseMatches.at(i).constraints.at(j).targetPoint(0),
              poseMatches.at(i).constraints.at(j).targetPoint(1),
              poseMatches.at(i).constraints.at(j).targetPoint(2));
        }
        glColor3f(1, 0, 0);
        gui->drawFrustum(poseMatches.at(i).first);
        glColor3f(0, 1, 0);
        gui->drawFrustum(poseMatches.at(i).second);
      }
    }
    glColor3f(1, 1, 1);

    glColor3f(0, 0, 1);
    for(const auto & tp : activeCtx.poseGraph())
    {
       gui->drawFrustum(tp.second);
    }
    glColor3f(1, 1, 1);
    eFusion->normaliseDepth(activeCtx, 0.3f, gui->depthCutoff->Get());

     glColor3f(1, 0.5, 0.6);
    for(const auto & oKF : orb_slam_tracker->GetKeyFramePoses())
    {
        Eigen::Matrix4f eigPose;
        cv::cv2eigen(oKF,eigPose);
        gui->drawFrustum(eigPose);
    }
    glColor3f(1, 1, 1);
   
    gui->displayImg(GPUTexture::DEPTH_NORM, activeCtx.textures()[GPUTexture::DEPTH_NORM]);
    activeCtx.indexMap().renderDepth(gui->depthCutoff->Get());
    gui->displayImg("Model", activeCtx.indexMap().drawTex());
    gui->displayImg(GPUTexture::RGB, activeCtx.textures()[GPUTexture::RGB]);
    gui->displayImg("ModelImage", activeCtx.indexMap().imageTex());


    if(pangolin::Pushed(*gui->save_images))
    {
      pangolin::TypedImage viewport_image;
      pangolin::TypedImage rgb_live_image;
      pangolin::TypedImage rgb_model_image;
      pangolin::TypedImage depth_live_image;
      pangolin::TypedImage depth_model_image;
      pangolin::TypedImage depth_model_values_image;

      activeCtx.textures()[GPUTexture::RGB]->texture->Download(rgb_live_image);
      activeCtx.textures()[GPUTexture::DEPTH_NORM]->texture->Download(depth_live_image);
      activeCtx.indexMap().imageTex()->texture->Download(rgb_model_image);
      activeCtx.indexMap().drawTex()->texture->Download(depth_model_image);
      activeCtx.indexMap().synthesizeDepth(activeCtx.currPose(),
                               eFusion->whichReferenceFrame(activeCtx).globalModel().model(),
                               20.0f,
                               0.7,
                               activeCtx.tick(),
                               activeCtx.id(),
                               activeCtx.tick(),
                               200);
      activeCtx.indexMap().depthTex()->texture->Download(depth_model_values_image);
      gui->colorTexture->texture->Download(viewport_image);

      cv::Mat viewport(2160,3840,CV_32FC4,
                       viewport_image.begin());
      cv::Mat rgb_live(Resolution::getInstance().rows(),
                       Resolution::getInstance().cols(), CV_32FC4,
                       rgb_live_image.begin());
      cv::Mat rgb_model(Resolution::getInstance().rows(),
                       Resolution::getInstance().cols(), CV_32FC4,
                       rgb_model_image.begin());
      cv::Mat depth_live(Resolution::getInstance().rows(),
                       Resolution::getInstance().cols(), CV_32FC1,
                       depth_live_image.begin());
      cv::Mat depth_model(Resolution::getInstance().rows(),
                       Resolution::getInstance().cols(), CV_32FC4,
                       depth_model_image.begin());
      cv::Mat depth_model_values(Resolution::getInstance().rows(),
                       Resolution::getInstance().cols(), CV_32FC1,
                       depth_model_values_image.begin());

      viewport.convertTo(viewport, CV_16UC4, 65535.0);
      rgb_live.convertTo(rgb_live, CV_16UC4, 65535.0);
      rgb_model.convertTo(rgb_model, CV_16UC4, 65535.0);
      depth_live.convertTo(depth_live, CV_16UC1, 65535.0);
      depth_model.convertTo(depth_model, CV_16UC4, 65535.0);
      depth_model_values.convertTo(depth_model_values, CV_16UC1, 1000.0);
      cv::cvtColor(viewport, viewport, cv::COLOR_RGBA2BGR);
      cv::cvtColor(rgb_live, rgb_live, cv::COLOR_RGBA2BGR);
      cv::cvtColor(depth_live, depth_live, cv::COLOR_GRAY2BGR);
      cv::cvtColor(rgb_model, rgb_model, cv::COLOR_RGBA2BGR);
      cv::cvtColor(depth_model, depth_model, cv::COLOR_RGBA2BGR);
      
      cv::flip(viewport, viewport, 0);
      
      cv::imwrite(Options::get().outDirectory + "/live_rgb_" + std::to_string(activeCtx.tick()) + ".png", rgb_live);
      cv::imwrite(Options::get().outDirectory + "/live_depth_" + std::to_string(activeCtx.tick()) + ".png", depth_live);
      cv::imwrite(Options::get().outDirectory + "/model_rgb_" + std::to_string(activeCtx.tick()) + ".png", rgb_model);
      cv::imwrite(Options::get().outDirectory + "/depth_model_" + std::to_string(activeCtx.tick()) + ".png", depth_model);
      cv::imwrite(Options::get().outDirectory + "/depth_model_values_" + std::to_string(activeCtx.tick()) + ".png", depth_model_values);

      pangolin::Display("Map").SaveOnRender(Options::get().outDirectory + "/viewport" + std::to_string(activeCtx.tick()));

    }

    gui->icpWeight->Ref().Set(Options::get().icp[activeCtx.id()]);

    std::stringstream strs;
    strs << eFusion->getGlobalModel(activeCtx).lastCount();

    gui->totalPoints->operator=(strs.str());

    std::stringstream strs2;
    strs2 << eFusion->whichReferenceFrame(activeCtx).globalDeformation().getGraph().size();

    gui->totalNodes->operator=(strs2.str());

    std::stringstream strs3;
    strs3 << eFusion->getFerns(activeCtx).frames.size();

    gui->totalFerns->operator=(strs3.str());

    std::stringstream strs4;
    strs4 << eFusion->getDeforms();

    gui->totalDefs->operator=(strs4.str());

    std::stringstream strs5;
    strs5 << activeCtx.tick() << "/" << activeLogReader->getNumFrames();

    gui->logProgress->operator=(strs5.str());

    std::stringstream strs6;
    strs6 << eFusion->getFernDeforms();

    gui->totalFernDefs->operator=(strs6.str());

    gui->numKfs->operator=(eFusion->numFused(activeCtx));
    // gui->postCall();

    activeLogReader->flipColors = gui->flipColors->Get();
    eFusion->setRgbOnly(gui->rgbOnly->Get());
    eFusion->setPyramid(gui->pyramid->Get());
    eFusion->setFastOdom(gui->fastOdom->Get());
    eFusion->setConfidenceThreshold(gui->confidenceThreshold->Get());
    eFusion->setDepthCutoff(gui->depthCutoff->Get());
    eFusion->setIcpWeight(gui->icpWeight->Get());
    eFusion->setSo3(gui->so3->Get());
    eFusion->setFrameToFrameRGB(gui->frameToFrameRGB->Get());
    eFusion->nidThreshold() = gui->nidThreshold->Get();
    eFusion->nidDepthLambda() = gui->nidDepthWeight->Get();
    eFusion->setNumBinsImg(gui->numBinsImg->Get());
    eFusion->setNumBinsDepth(gui->numBinsDepth->Get());
    eFusion->nidPyramidLevel() = gui->nidPyramidLevel->Get();

    resetButton = pangolin::Pushed(*gui->reset);

    if (gui->autoSettings) {
      static bool last = gui->autoSettings->Get();

      if (gui->autoSettings->Get() != last) {
        last = gui->autoSettings->Get();
        std::static_pointer_cast<LiveLogReader>(activeLogReader)->setAuto(last);
      }
    }

    Stopwatch::getInstance().sendAll();

    if (resetButton) {
      break;
    }
    bool empty = true;
    for (auto &lr : logReaders) {
      empty = lr.second->hasMore() ? false : true;
      if (!empty) break;
    }

    if (pangolin::Pushed(*gui->save) || (empty && Options::get().quiet)) {
      eFusion->savePly(Options::get().outDirectory);
      eFusion->saveTrajectories(Options::get().outDirectory);
      eFusion->saveTimes(Options::get().outDirectory);
      eFusion->saveStats(Options::get().outDirectory);
      if (empty) {
        break;
      }
    }

    if (pangolin::Pushed(*gui->batchAlign)) {
      eFusion->batchAlign(activeCtx);
    }

    gui->postCall();
    TOCK("GUI");
  }
}
