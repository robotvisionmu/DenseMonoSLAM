#ifndef MULTIUSBCAMERAMANAGER_H_
#define MULTIUSBCAMERAMANAGER_H_

#include <OpenNI.h>
#include <librealsense2/rs.hpp>

#include "MultiCameraManager.h"
#include "LiveLogReader.h"

class MultiUsbCameraManager : public MultiCameraManager
{
  public:
    MultiUsbCameraManager()
    {
        //openni::Status rc = openni::STATUS_OK;

        //rc = 
        openni::OpenNI::initialize();

        std::string errorString(openni::OpenNI::getExtendedError());

        if (errorString.length() > 0)
        {
            std::cout << "OpenNI failure:\n"
                      << errorString << "\n";
            std::exit(EXIT_FAILURE);
        }

        openni::Array<openni::DeviceInfo> deviceInfoList;
        openni::OpenNI::enumerateDevices(&deviceInfoList);

        int nc = std::min(Options::get().numSensors, deviceInfoList.getSize());
        for (int i = 0; i < nc; i++)
        {
            std::cout << deviceInfoList[i].getUri() << "\n";
            std::shared_ptr<LiveLogReader> lr(new LiveLogReader(deviceInfoList[i].getUri(), deviceInfoList[i].getUri(),
                                                                Options::get().flip, LiveLogReader::CameraType::OpenNI2));
            if (lr->cam->ok())
                m_lrs.push_back(lr);
        }

        rs2::context ctx;
        if(ctx.query_devices().size() > 0)
        {
            std::shared_ptr<LiveLogReader> lr(new LiveLogReader("deviceInfoList[i].getUri()", "deviceInfoList[i].getUri()",
                                                                Options::get().flip, LiveLogReader::CameraType::RealSense));
            if (lr->cam->ok())
                m_lrs.push_back(lr);
        }

    }
    virtual ~MultiUsbCameraManager() {}

    std::vector<std::shared_ptr<LogReader>> devices() const
    {
        return std::vector<std::shared_ptr<LogReader>>(m_lrs.begin(), m_lrs.end());
    }

    void reset()
    {
        // m_lrs.clear();

        // openni::Array<openni::DeviceInfo> deviceInfoList;
        // openni::OpenNI::enumerateDevices(&deviceInfoList);

        // int nc = std::min(Options::get().numSensors, deviceInfoList.getSize());
        // for (int i = 0; i < nc; i++)
        // {
        //     std::shared_ptr<LiveLogReader> lr(new LiveLogReader(deviceInfoList[i].getUri(),
        //                                                         Options::get().flip, LiveLogReader::CameraType::OpenNI2));
        //     if (lr->cam->ok())
        //         m_lrs.push_back(lr);
        // }
    }

  private:
    std::vector<std::shared_ptr<LiveLogReader>> m_lrs;
};

#endif /*MULTIUSBCAMERAMANAGER_H_*/