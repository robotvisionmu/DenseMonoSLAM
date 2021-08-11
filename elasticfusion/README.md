# ElasticFusion #

Our system builds and extends on the [ElasticFusion](https://github.com/mp3guy/ElasticFusion) dense SLAM system. 

# 1. What do I need to build it?

## 1.1. Ubuntu ##

* Ubuntu 14.04, 15.04 or 16.04 (Though many other linux distros will work fine)
* CMake
* OpenGL
* [CUDA >= 7.0](https://developer.nvidia.com/cuda-downloads)
* [OpenNI2](https://github.com/occipital/OpenNI2)
* SuiteSparse
* Eigen
* zlib
* libjpeg
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)
* [librealsense] (https://github.com/IntelRealSense/librealsense) - Optional (for Intel RealSense cameras)
* [OnnxRuntime](https://onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html#requirements) -  - Install with CUDA execution provider
* [LCM](https://lcm-proj.github.io/) - Need for handling logs.


Firstly, add [nVidia's official CUDA repository](https://developer.nvidia.com/cuda-downloads) to your apt sources, then run the following command to pull in most dependencies from the official repos:

```bash
sudo apt-get install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev openjdk-7-jdk freeglut3-dev libglew-dev cuda-7-5 libsuitesparse-dev libeigen3-dev zlib1g-dev libjpeg-dev
```

Afterwards install [OpenNI2](https://github.com/occipital/OpenNI2) and [Pangolin](https://github.com/stevenlovegrove/Pangolin) from source. Note, you may need to manually tell CMake where OpenNI2 is since Occipital's fork does not have an install option. It is important to build Pangolin last so that it can find some of the libraries it has optional dependencies on. 

When you have all of the dependencies installed, build the Core followed by the GUI. 

# 6. Datasets #

We have provided a sample dataset which you can run easily with ElasticFusion for download [here](http://www.doc.ic.ac.uk/~sleutene/datasets/elasticfusion/dyson_lab.klg). Launch it as follows:

```bash
./ElasticFusion -l dyson_lab.klg
```

# 7. License #
ElasticFusion is freely available for non-commercial use only.  Full terms and conditions which govern its use are detailed [here](http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/) and in the LICENSE.txt file.

# 8. FAQ #
***What are the hardware requirements?***

A [very fast nVidia GPU (3.5TFLOPS+)](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#GeForce_900_Series), and a fast CPU (something like an i7). If you want to use a non-nVidia GPU you can rewrite the tracking code or substitute it with something else, as the rest of the pipeline is actually written in the OpenGL Shading Language. 

***How can I get performance statistics?***

Download [Stopwatch](https://github.com/mp3guy/Stopwatch) and run *StopwatchViewer* at the same time as ElasticFusion. 

***I ran a large dataset and got assert(graph.size() / 16 < MAX_NODES) failed***

Currently there's a limit on the number of nodes in the deformation graph down to lazy coding (using a really wide texture instead of a proper 2D one). So we're bound by the maximum dimension of a texture, which is 16384 on modern cards/OpenGL. Either fix the code so this isn't a problem any more, or increase the modulo factor in *Shaders/sample.geom*. 

***I have a nice new laptop with a good GPU but it's still slow***

If your laptop is running on battery power the GPU will throttle down to save power, so that's unlikely to work (as an aside, [Kintinuous](https://github.com/mp3guy/Kintinuous) will run at 30Hz on a modern laptop on battery power these days). You can try disabling SO(3) pre-alignment, enabling fast odometry, only using either ICP or RGB tracking and not both, running in open loop mode or disabling the tracking pyramid. All of these will cost you accuracy. 

***I saved a map, how can I view it?***

Download [Meshlab](http://meshlab.sourceforge.net/). Select Render->Shaders->Splatting. 

***The map keeps getting corrupted - tracking is failing - loop closures are incorrect/not working***

Firstly, if you're running live and not processing a log file, ensure you're hitting 30Hz, this is important. Secondly, you cannot move the sensor extremely fast because this violates the assumption behind projective data association. In addition to this, you're probably using a primesense, which means you're suffering from motion blur, unsynchronised cameras and rolling shutter. All of these are aggravated by fast motion and hinder tracking performance. 

If you're not getting loop closures and expecting some, pay attention to the inlier and residual graphs in the bottom right, these are an indicator of how close you are to a local loop closure. For global loop closures, you're depending on [fern keyframe encoding](http://www.doc.ic.ac.uk/~bglocker/pdfs/glocker2015tvcg.pdf) to save you, which like all appearance-based place recognition methods, has its limitations. 

***Is there a ROS bridge/node?***

No. The system relies on an extremely fast and tight coupling between the mapping and tracking on the GPU, which I don't believe ROS supports natively in terms of message passing. 

***This doesn't seem to work like it did in the videos/papers***

A substantial amount of refactoring was carried out in order to open source this system, including rewriting a lot of functionality to avoid certain licenses and reduce dependencies. Although great care was taken during this process, it is possible that performance regressions were introduced and have not yet been discovered.
