#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/program_options.hpp>
#include <pangolin/pangolin.h>
#define XSTR(x) #x
#define STR(x) XSTR(x)

namespace po = boost::program_options;

class Options {
public:
  static const Options &get(int argc = 0, char **argv = 0) {
    static const Options instance(argc, argv);
    return instance;
  }

  bool iclnuim;
  bool openLoop;
  bool reloc;
  bool frameskip;
  bool quiet;
  bool fastOdom;
  bool rewind;
  bool frameToFrameRGB;
  bool sc;
  bool flip;
  bool live;
  bool so3;
  bool interMap;
  bool lcm;
  bool predict_depth;
  bool hybrid_tracking;
  bool hybrid_loops;
  bool half_float;

  int numSensors;
  int numFusing;
  int timeDelta;
  int icpCountThresh;
  int start;
  int end;
  int pbSpeed;
  int port;
  int ttl;
  int defGraphSampleRate;
  float confidence;
  float depth;
  std::vector<float> icp;
  float icpErrThresh;
  float covThresh;
  float photoThresh;
  float interMapPhotoThresh;
  float fernThresh;

  std::string calibrationFile;
  std::vector<std::string> logfiles;
  std::vector<std::string> posesFiles;
  std::string lcmChannel;
  std::string multicast;
  std::string clustersFile;
  std::string outDirectory;

  float nidDepthWeight;
  float nidThreshold;
  int numBinsImg;
  int numBinsDepth;
  bool noKeyframe;
  int nidPyramidLevel;

  bool orb_tracking;
  std::string orb_vocabulary;
  std::string orb_config_yaml;

private:
  Options(int argc, char **argv)
      : iclnuim(false), openLoop(false), reloc(false), frameskip(false),
        quiet(false), fastOdom(false), rewind(false), frameToFrameRGB(false),
        sc(false), flip(false), live(false), so3(true),
        interMap(
            true), // set to false to turn off online intermap loop closures
        lcm(false),
        predict_depth(false), hybrid_tracking(false),hybrid_loops(false), half_float(false), numSensors(1), numFusing(1), timeDelta(200),
        icpCountThresh(35000), start(1),
        end(std::numeric_limits<unsigned short>::max()), pbSpeed(1), port(7667),
        ttl(0), defGraphSampleRate(5000), confidence(10.0f), depth(3.0f), icp(numSensors, 10.0f),
        icpErrThresh(2e-05), covThresh(1e-05), photoThresh(115),
        interMapPhotoThresh(115), fernThresh(0.3095f),
        lcmChannel("ELASTIC_FUSION.*"), multicast("239.255.76.67"),
        clustersFile(""), outDirectory("./"), nidDepthWeight(0.7f),
        nidThreshold(0.85f), numBinsImg(64), numBinsDepth(500),
        noKeyframe(false), nidPyramidLevel(0), orb_tracking(false),
        orb_vocabulary("/home/louis/Development/elasticfusion/orb_vocab/ORBvoc.txt"), orb_config_yaml("") {
    po::options_description desc(
        "ElasticFusion, collaborative mapping version.\n Options:");

    desc.add_options()("help", "Print this help message")(
        "icl",
        "Set if using ICL-NUIM datasets (flips normals to account for negative "
        "focal length on that data). [Default = False]")(
        "o", "Open loop mode (i.e. no loop closures). [Default = False]")(
        "rl", "Enable relocalisation. [Default = False]")(
        "fs",
        "If playing back log files skip frames to simulate real-time. [Default "
        "= False]")("q", "Quit when finished a log. [Default = False]")(
        "fo", "Fast odometry (single pyramid level). [Default = False]")(
        "r", "Rewind log and loop forever. [Default = False]")(
        "ftf", "Frame-to-frame RGB tracking. [Default = False]")(
        "sc", "Showcase mode (minimal GUI). [Default = False]")(
        "f", "Flip RGB/BGR. [Default = False]")(
        "live",
        "If playing back LCM logs simulate playing them back over the wire "
        "(LCM only). [Default = False]")(
        "nso", "Disable SO3 pre-alignment in tracking. [Default = False]")(
        "nim", "Disable inter map loop closures. [Default = False]")(
        "n", po::value<int>(),
        "Total number of sensors (mapping + tracking). [Default = 1]")(
        "nf", po::value<int>(), "Number of sensors fusing. [Default = 1]")(
        "t", po::value<int>(), "Time window length. [Default = 200]")(
        "ic", po::value<int>(),
        "Local loop closure inlier threshold. [Default = 35000]")(
        "s", po::value<int>(), "Frames to skip at start of log. [Default = 0]")(
        "e", po::value<int>(), "Cut off frame of log. [Default = MAX_INT]")(
        "pbs", po::value<int>(), "Playback speed (LCM only). [Default = 1]")(
        "p", po::value<int>(),
        "Port number to listen on (LCM only). [Default = 7667]")(
        "ttl", po::value<int>(),
        "Multicast ttl (loopback = 0, local network = 1). [Default = 0]")
        ("dgs", po::value<int>(), "Rate for sampling surface points for deformation graph. [Default = 5000]")(
        "c", po::value<float>(), "Surfel confidence threshold. [Default = 10]")(
        "d", po::value<float>(),
        "Cutoff distance for depth processing (m). [Default = 3m]")(
        "i", po::value<std::vector<float>>()->multitoken(),
        "Relative ICP/RGB tracking weights. One for each input sequence "
        "[Default = 10]")(
        "ie", po::value<float>(),
        "Local loop closure residual threshold. [Default = 5e-05]")(
        "cv", po::value<float>(),
        "Local loop closure covariance threshold. [Default = 1e-05]")(
        "pt", po::value<float>(),
        "Global loop closure photometric threshold. [Default = 115]")(
        "ipt", po::value<float>(),
        "Intermap loop closure photo threshold. [Default = 115]")(
        "ft", po::value<float>(),
        "Fern encoding threshold. [Default =  0.3095]")(
        "ch", po::value<std::string>(),
        "LCM channel to listen on (LCM only). [Default = ELASTIC_FUSION.*]")(
        "m", po::value<std::string>(),
        "Underlying multicast channel for LCM (LCM only). [Default = "
        "239.255.76.67]")("cal", po::value<std::string>(),
                          "Loads a camera calibration file specified as fx fy "
                          "cx cy. [Default = None]")(
        "l", po::value<std::vector<std::string>>()->multitoken(),
        "One or more log files in klg or lcm format (see lcmtypes). [Default = "
        "None]")("lcm", "Stream camera feed using LCM. [Default = False]")(
        "predict_depth", "Predict depth map using NN. [Default = False]")("hybrid_tracking", "Perform Hybrid ORB + dense frame-to-model Camera tracking. [Default = False]")("hybrid_loops", "Perform Hybrid Loop closures using ORB constraint and deformation graph. [Default = False]")("half_float", "Use float 16 version of depth prediction network. [Default = False]")(
        "poses", po::value<std::vector<std::string>>()->multitoken(),
        "Ground truth poses files for the input sequences")(
        "clusters", po::value<std::string>(),
        "Ground truth clusters for the input sequence")(
        "od", po::value<std::string>(),
        "Directory for mapping output artefacts. [Default=./]")(
        "ndw", po::value<float>(),
        "Relative weight between depth and rgb NID scores. [Default = 0.7]")(
        "nid", po::value<float>(),
        "NID threshold to reach in order to fuse current frame. [Default = "
        "0.85]")(
        "nbi", po::value<int>(),
        "Number of histogram bins for intensity image. [Default = 64]")(
        "nbd", po::value<int>(),
        "Number of histogram bins for depth image. [Default = 500]")(
        "nkf", "If true NID keyframing is turned off. [Default = False]")(
        "npl", po::value<int>(),
        "Pyramid level to compute NID at. [Default = 0]")(
        "orb_tracking", "Use ORBSLAM3 camera tracking [Default = false]")(
        "orb_vocab", po::value<std::string>(),
        "DBoW vocabulary used by ORBSLAM3 [Default = \"\"]")(
        "orb_params", po::value<std::string>(),
        "Configuration parameters used by ORBSLAM3 [Default = \"\"]");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

    if (vm.count("icl"))
      iclnuim = true;

    if (vm.count("o"))
      openLoop = true;

    if (vm.count("rl"))
      reloc = true;

    if (vm.count("fs"))
      frameskip = true;

    if (vm.count("q"))
      quiet = true;

    if (vm.count("fo"))
      fastOdom = true;

    if (vm.count("r"))
      rewind = true;

    if (vm.count("ftf"))
      frameToFrameRGB = true;

    if (vm.count("sc"))
      sc = true;

    if (vm.count("f"))
      flip = true;

    if (vm.count("live"))
      live = true;

    if (vm.count("nso"))
      so3 = false;

    if (vm.count("nim"))
      interMap = false;

    if (vm.count("n"))
      numSensors = vm["n"].as<int>();

    if (vm.count("nf"))
      numFusing = vm["nf"].as<int>();

    if (vm.count("t"))
      timeDelta = vm["t"].as<int>();

    if (vm.count("ic"))
      icpCountThresh = vm["ic"].as<int>();

    if (vm.count("s"))
      start = vm["s"].as<int>();

    if (vm.count("e"))
      end = vm["e"].as<int>();

    if (vm.count("pbs"))
      pbSpeed = vm["pbs"].as<int>();

    if (vm.count("p"))
      port = vm["p"].as<int>();

    if (vm.count("ttl"))
      ttl = vm["ttl"].as<int>();
    
    if(vm.count("dgs"))
      defGraphSampleRate = vm["dgs"].as<int>();
    
    if (vm.count("c"))
      confidence = vm["c"].as<float>();

    if (vm.count("d"))
      depth = vm["d"].as<float>();

    icp = std::vector<float>(numSensors, 10.0f);

    if (vm.count("i")) {
      std::vector<float> icpWeights = vm["i"].as<std::vector<float>>();
      std::cout << icpWeights.size() << std::endl;
      for (int i = 0; (size_t)i < icpWeights.size(); i++)
        icp[i] = icpWeights[i];
      // icp = vm["i"].as<float>();
    }
    if (vm.count("ie"))
      icpErrThresh = vm["ie"].as<float>();

    if (vm.count("cv"))
      covThresh = vm["cv"].as<float>();

    if (vm.count("pt"))
      photoThresh = vm["pt"].as<float>();

    if (vm.count("ipt"))
      interMapPhotoThresh = vm["ipt"].as<float>();

    if (vm.count("ft"))
      fernThresh = vm["ft"].as<float>();

    if (vm.count("ch"))
      lcmChannel = vm["ch"].as<std::string>();

    if (vm.count("m"))
      multicast = vm["m"].as<std::string>();

    if (vm.count("cal"))
      calibrationFile = vm["cal"].as<std::string>();

    if (vm.count("l"))
      logfiles = vm["l"].as<std::vector<std::string>>();

    if (vm.count("lcm"))
      lcm = true;
    if (vm.count("poses"))
      posesFiles = vm["poses"].as<std::vector<std::string>>();

    if (vm.count("clusters"))
      clustersFile = vm["clusters"].as<std::string>();

    if (vm.count("od"))
      outDirectory = vm["od"].as<std::string>();

    struct stat info;
    if(stat(outDirectory.c_str(), &info ) != 0 )
      system(("mkdir -p " + outDirectory).c_str());

    if (vm.count("ndw"))
      nidDepthWeight = vm["ndw"].as<float>();

    if (vm.count("nid"))
      nidThreshold = vm["nid"].as<float>();

    if (vm.count("nbi"))
      numBinsImg = vm["nbi"].as<int>();

    if (vm.count("nbd"))
      numBinsDepth = vm["nbd"].as<int>();

    if (vm.count("nkf"))
      noKeyframe = true;

    if (vm.count("npl"))
      nidPyramidLevel = vm["npl"].as<int>();

    if (vm.count("orb_tracking"))
      orb_tracking = true;
    if (vm.count("orb_vocab"))
      orb_vocabulary = vm["orb_vocab"].as<std::string>();
    if (vm.count("orb_params"))
      orb_config_yaml = vm["orb_params"].as<std::string>();

    if (vm.count("predict_depth"))
      predict_depth = true;
    
    if (vm.count("hybrid_tracking"))
      hybrid_tracking = true;
    
    if (vm.count("hybrid_loops"))
      hybrid_loops = true;
    
     if (vm.count("half_float"))
       half_float = true;
  }

public:
  std::string shaderDir() const {
    std::string currentVal = STR(SHADER_DIR);

    assert(pangolin::FileExists(currentVal) && "Shader directory not found!");

    return currentVal;
  }

  std::string baseDir() const {
    char buf[256];
#ifdef WIN32
    int length = GetModuleFileName(NULL, buf, sizeof(buf));
#else
    int length = readlink("/proc/self/exe", buf, sizeof(buf));
#endif
    std::string currentVal;
    currentVal.append((char *)&buf, length);

    currentVal = currentVal.substr(0, currentVal
#ifdef WIN32
                                          .rfind("\\build\\"));
#else
                                          .rfind("/build/"));
#endif
    return currentVal;
  }

  static std::string lcmUrl() {
    std::stringstream ss;
    std::string logfile;

    if (Options::get().logfiles.size() >= (size_t)Options::get().numSensors) {
      ss << "file://" << Options::get().logfiles[0] << "?"
         << "mode=r"
         << "&"
         << "speed=" << Options::get().pbSpeed;
    } else {
      ss << "udpm://" << Options::get().multicast << ":" << Options::get().port
         << "?"
         << "ttl=" << Options::get().ttl;
      std::cout << ss.str() << std::endl;
    }

    return ss.str();
  }

  static void print() {
    if (Options::get().openLoop)
      std::cout << "openloop" << std::endl;
    if (Options::get().iclnuim)
      std::cout << "icl nuim" << std::endl;
    if (Options::get().frameskip)
      std::cout << "frame skip" << std::endl;
    if (Options::get().flip)
      std::cout << "flip" << std::endl;
    if (Options::get().fastOdom)
      std::cout << "fast odom" << std::endl;
  }

  Options(Options const &) = delete;
  void operator=(Options const &) = delete;
};
#endif // OPTIONS_H_