#ifndef REALSENSEINTERFACE_H_
#define REALSENSEINTERFACE_H_

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

//#ifdef WITH_REALSENSE
#include <librealsense2/rs.hpp>
//#endif

#include "CameraInterface.h"
#include "ThreadMutexObject.h"

class RealSenseInterface : public CameraInterface {
 public:
  RealSenseInterface(int width = 640, int height = 480, int fps = 30);
  virtual ~RealSenseInterface();

  const int width, height, fps;

  bool getAutoExposure();
  bool getAutoWhiteBalance();
  virtual void setAutoExposure(bool value);
  virtual void setAutoWhiteBalance(bool value);

  virtual bool ok() { return initSuccessful; }

  virtual std::string error() { return errorText; }

  struct FrameCallback {
   public:
    FrameCallback(
        int64_t &lastFrameTime, ThreadMutexObject<int> &latestFrameIndex,
        std::pair<std::pair<uint8_t *, uint8_t *>, int64_t> *frameBuffers, rs2::align & align)
        : lastFrameTime(lastFrameTime),
          latestFrameIndex(latestFrameIndex),
          frameBuffers(frameBuffers),
          align(align)
        {
            //rs2::decimation_filter * dec_filter = new rs2::decimation_filter;
            rs2::spatial_filter * spat_filter = new rs2::spatial_filter;
            rs2::temporal_filter * temp_filter = new rs2::temporal_filter;
            rs2::disparity_transform * depth_to_disparity = new rs2::disparity_transform(true);
            rs2::disparity_transform * disparity_to_depth = new rs2::disparity_transform(false);

            //filters.push_back(dec_filter);
            filters.push_back(depth_to_disparity);
            filters.push_back(spat_filter);
            filters.push_back(temp_filter);
            filters.push_back(disparity_to_depth);
        }

    void operator()(const rs2::frame & frame) {
      if (rs2::frameset fs = frame.as<rs2::frameset>()) {
        lastFrameTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

        int bufferIndex = (latestFrameIndex.getValue() + 1) % numBuffers;

        auto aligned = align.process(fs);

        rs2::depth_frame filtered_depth = aligned.get_depth_frame();
        
        for(auto filter : filters)
        {
          filtered_depth = filter->process(filtered_depth);
        }
        
        std::memcpy(frameBuffers[bufferIndex].first.first, filtered_depth.get_data(),
                    fs.get_depth_frame().get_width() *
                        fs.get_depth_frame().get_height() * 2);

        frameBuffers[bufferIndex].second = lastFrameTime;

        std::memcpy(frameBuffers[bufferIndex].first.second,
                    fs.get_color_frame().get_data(),
                    fs.get_color_frame().get_width() *
                        fs.get_color_frame().get_height() * 3);

        latestFrameIndex++;
      }
    }

   private:
    int64_t &lastFrameTime;
    ThreadMutexObject<int> &latestFrameIndex;
    std::pair<std::pair<uint8_t *, uint8_t *>, int64_t> *frameBuffers;
    rs2::align & align;
    std::vector<rs2::filter * > filters;
  };

 private:
  rs2::context ctx;
  rs2::pipeline *pipeline;
  FrameCallback *frameCallback;
  rs2::align *align;

  bool initSuccessful;
  std::string errorText;

  //ThreadMutexObject<int> latestDepthIndex;

  int64_t lastDepthTime;
};
#endif /*REALSENSEINTERFACE_H_*/