# DenseMonoSLAM
A dense monocular SLAM system for capturing dense surfel-based maps of outdoor environments using a single monocular camera.
The code base also supports collaborative mapping sessions with multiple independently moving cameras. Check out the following links to see examples of the system in action; [__dense monocular session__](https://youtu.be/Pn2uaVqjskY), [__collaborative session__](https://youtu.be/GUtHrKEM85M)

# Related Publication
Please cite this work if you make use of our system in your own research.
- __A Hybrid Sparse-Dense Monocular SLAM framework for Autonomous Driving__, _Louis Gallagher, Varun Ravi Kumar, Senthil Yogamani and John B. McDonald_, ECMR'21 
- [__Collaborative Dense SLAM__](https://arxiv.org/abs/1811.07632), _Louis Gallagher & John B, McDonald_, IMVIP'18, _Best Paper Winner_
- __Efficient Surfel Fusion Using Normalised Information Distance__, _Louis Gallagher & John B, McDonald_, CVPR'19 Workshop 3D Scene Understanding for Vision, Graphics and Robotics

# Building The System
First checkout the code and all of the submodules. Change directory into the orb_slam subdirectory and follow the build instructions there. Next change directory into the ElasticFusion sub directory and follow the build instructions there. 

# Data Format
We have opted to use [LCM](https://lcm-proj.github.io/) in our system for handling logs and streaming from cameras into the system. Frames in our log format have the following LCM signature
```c
package eflcm;

struct Frame
{
	boolean trackOnly;
	boolean compressed;
	boolean last;
	
	int32_t depthSize;
	int32_t imageSize;

	byte depth[depthSize];
	byte image[imageSize];

	int64_t timestamp;

	int32_t frameNumber;

	string senderName;
}
```
A log file can be specified upon launch of the system from the command line. Alternatively, a LCM channel can be specified, the system will then listen to that channel for LCM packets. Sample data of a sequence from the KITTI odometry benchmark is available [here]().

## Generating Your Own Data
To generate your own data from the ICL-NUIM dataset, the TUM RGB-D and KITTI we have provided a set of translation tools in the [log](./log) sub directory.

# Running The System
The following command line parameters are supported
```
  --help                Print this help message
  --icl                 Set if using ICL-NUIM datasets (flips normals to 
                        account for negative focal length on that data). 
                        [Default = False]
  --o                   Open loop mode (i.e. no loop closures). [Default = 
                        False]
  --rl                  Enable relocalisation. [Default = False]
  --fs                  If playing back log files skip frames to simulate 
                        real-time. [Default = False]
  --q                   Quit when finished a log. [Default = False]
  --fo                  Fast odometry (single pyramid level). [Default = False]
  --r                   Rewind log and loop forever. [Default = False]
  --ftf                 Frame-to-frame RGB tracking. [Default = False]
  --sc                  Showcase mode (minimal GUI). [Default = False]
  --f                   Flip RGB/BGR. [Default = False]
  --live                If playing back LCM logs simulate playing them back 
                        over the wire (LCM only). [Default = False]
  --nso                 Disable SO3 pre-alignment in tracking. [Default = 
                        False]
  --nim                 Disable inter map loop closures. [Default = False]
  --n arg               Total number of sensors (mapping + tracking). [Default 
                        = 1]
  --nf arg              Number of sensors fusing. [Default = 1]
  --t arg               Time window length. [Default = 200]
  --ic arg              Local loop closure inlier threshold. [Default = 35000]
  --s arg               Frames to skip at start of log. [Default = 0]
  --e arg               Cut off frame of log. [Default = MAX_INT]
  --pbs arg             Playback speed (LCM only). [Default = 1]
  --p arg               Port number to listen on (LCM only). [Default = 7667]
  --ttl arg             Multicast ttl (loopback = 0, local network = 1). 
                        [Default = 0]
  --dgs arg             Rate for sampling surface points for deformation graph.
                        [Default = 5000]
  --c arg               Surfel confidence threshold. [Default = 10]
  --d arg               Cutoff distance for depth processing (m). [Default = 
                        3m]
  --i arg               Relative ICP/RGB tracking weights. One for each input 
                        sequence [Default = 10]
  --ie arg              Local loop closure residual threshold. [Default = 
                        5e-05]
  --cv arg              Local loop closure covariance threshold. [Default = 
                        1e-05]
  --pt arg              Global loop closure photometric threshold. [Default = 
                        115]
  --ipt arg             Intermap loop closure photo threshold. [Default = 115]
  --ft arg              Fern encoding threshold. [Default =  0.3095]
  --ch arg              LCM channel to listen on (LCM only). [Default = 
                        ELASTIC_FUSION.*]
  --m arg               Underlying multicast channel for LCM (LCM only). 
                        [Default = 239.255.76.67]
  --cal arg             Loads a camera calibration file specified as fx fy cx 
                        cy. [Default = None]
  --l arg               One or more log files in klg or lcm format (see 
                        lcmtypes). [Default = None]
  --lcm                 Stream camera feed using LCM. [Default = False]
  --predict_depth       Predict depth map using NN. [Default = False]
  --hybrid_tracking     Perform Hybrid ORB + dense frame-to-model Camera 
                        tracking. [Default = False]
  --hybrid_loops        Perform Hybrid Loop closures using ORB constraint and 
                        deformation graph. [Default = False]
  --half_float          Use float 16 version of depth prediction network. 
                        [Default = False]
  --poses arg           Ground truth poses files for the input sequences
  --clusters arg        Ground truth clusters for the input sequence
  --od arg              Directory for mapping output artefacts. [Default=./]
  --ndw arg             Relative weight between depth and rgb NID scores. 
                        [Default = 0.7]
  --nid arg             NID threshold to reach in order to fuse current frame. 
                        [Default = 0.85]
  --nbi arg             Number of histogram bins for intensity image. [Default 
                        = 64]
  --nbd arg             Number of histogram bins for depth image. [Default = 
                        500]
  --nkf                 If true NID keyframing is turned off. [Default = False]
  --npl arg             Pyramid level to compute NID at. [Default = 0]
  --orb_tracking        Use ORBSLAM3 camera tracking [Default = false]
  --orb_vocab arg       DBoW vocabulary used by ORBSLAM3 [Default = ""]
  --orb_params arg      Configuration parameters used by ORBSLAM3 [Default = 
                        ""]

```

## Running the system with KITTI 
An example of running the system in monocular mode on a KITTI odometry benchmark sequence:
```
./ElasticFusion --q --f --predict_depth --orb_tracking --nkf --c 0.700000 --t 200 --ic 35000 --ie 0.000050 --pt 115.000000 --ipt 115.000000 --ft 0.309500 --dgs 5000 --d 40.000000 --od ./results/ --l sequence.lcm --cal sequence.calib.txt --orb_params sequence.orb.params.yaml --nf 1 --n 1
```
## Collaborative Session with ICL-NUIM
An example of running a collaborative 2 camera session with ICL-NUIM:
```
./ElasticFusion --l sequence_1.lcm --l sequence_2.lcm --n 2 --nf 2 --cal calib.txt
```