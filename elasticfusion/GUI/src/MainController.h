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

#include <Cuda/convenience.cuh>
#include <ElasticFusion.h>
#include <Utils/Options.h>
#include <Utils/Parse.h>
#include <future>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "Tools/DepthPrediction.h"
#include "Tools/GUI.h"
#include "Tools/GroundTruthClusters.h"
#include "Tools/GroundTruthOdometry.h"
#include "Tools/LiveLogReader.h"
#include "Tools/MultiCameraManagerFactory.h"
#include "Tools/RawLogReader.h"

#include <System.h>

#include <opencv2/core/eigen.hpp>

#ifndef MAINCONTROLLER_H_
#define MAINCONTROLLER_H_

class MainController {
public:
  MainController(int argc, char *argv[]);
  virtual ~MainController();

  void launch();

private:
  void run();

  void loadCalibration(const std::string &filename);

  MultiCameraManager *cameraManager;

  bool good;
  ElasticFusion *eFusion;
  GUI *gui;
  std::vector<GroundTruthOdometry *> groundTruthOdometry;
  GroundTruthClusters *groundTruthClusters;
  bool iclnuim;
  std::string logFile;
  std::string poseFile;

  float confidence, depth, icp, icpErrThresh, covThresh, photoThresh,
      fernThresh;

  int timeDelta, icpCountThresh, numFusing, start, end;

  bool fillIn, openLoop, reloc, frameskip, quiet, fastOdom, so3, rewind,
      frameToFrameRGB;

  int framesToSkip;
  bool streaming;
  bool resetButton;

  Resize *resizeStream;

  int numCameras;
  int numMaps;
  std::map<std::string, std::shared_ptr<LogReader>> logReaders;

  ORB_SLAM3::System *orb_slam_tracker;
  DepthPrediction *depth_prediction;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif /* MAINCONTROLLER_H_ */
