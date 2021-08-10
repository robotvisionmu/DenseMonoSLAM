#include "RealSenseInterface.h"
#include <functional>

//#ifdef WITH_REALSENSE
RealSenseInterface::RealSenseInterface(int inWidth,int inHeight,int inFps)
  : width(inWidth),
  height(inHeight),
  fps(inFps),
  pipeline(nullptr),
  align(nullptr),
  initSuccessful(true)
{
  if(ctx.query_devices().size() == 0)
  {
    errorText = "No device connected.";
    initSuccessful = false;
    return;
  }

  pipeline = new rs2::pipeline(ctx);
  rs2::config config;
  config.enable_stream(RS2_STREAM_DEPTH, -1, width, height, RS2_FORMAT_Z16,fps);
  config.enable_stream(RS2_STREAM_COLOR, -1, width, height, RS2_FORMAT_RGB8,fps);
  
  latestDepthIndex.assign(-1);

  for(int i = 0; i < numBuffers; i++)
  {
    uint8_t * newDepth = (uint8_t *)std::calloc(width * height * 2,sizeof(uint8_t));
    uint8_t * newImage = (uint8_t *)std::calloc(width * height * 3,sizeof(uint8_t));
    frameBuffers[i] = std::pair<std::pair<uint8_t *,uint8_t *>,int64_t>(std::pair<uint8_t *,uint8_t *>(newDepth,newImage),0);
  }

  setAutoExposure(true);
  setAutoWhiteBalance(true);

  align = new rs2::align(RS2_STREAM_COLOR);

  frameCallback = new FrameCallback(lastDepthTime, latestDepthIndex, frameBuffers, *align);
  pipeline->start(config, *frameCallback);

  // auto const i = pipeline->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
  // std::cout << i.fx << " " << i.fy << " " << i.ppx << " " << i.ppy << std::endl;
}

RealSenseInterface::~RealSenseInterface()
{
  if(initSuccessful)
  {
    pipeline->stop();

    for(int i = 0; i < numBuffers; i++)
    {
      std::free(frameBuffers[i].first.first);
      std::free(frameBuffers[i].first.second);
    }

    delete frameCallback;
  }
}

void RealSenseInterface::setAutoExposure(bool value)
{
 // dev->set_option(rs2::option::color_enable_auto_exposure,value);
}

void RealSenseInterface::setAutoWhiteBalance(bool value)
{
  //dev->set_option(rs2::option::color_enable_auto_white_balance,value);
}

bool RealSenseInterface::getAutoExposure()
{
 // return dev->get_option(rs2::option::color_enable_auto_exposure);
 return false;
}

bool RealSenseInterface::getAutoWhiteBalance()
{
  //return dev->get_option(rs2::option::color_enable_auto_white_balance);
  return false;
}
