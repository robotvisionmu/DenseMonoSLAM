#ifndef MULTICAMERAMANAGERFACTORY_H_
#define MULTICAMERAMANAGERFACTORY_H_

#include "MultiLiveCameraManager.h"
#include "MultiLogCameraManager.h"
#include "MultiMixedCameraManager.h"

#include <Utils/Options.h>

class MultiCameraManagerFactory
{
	public:
		static MultiCameraManager * get()
		{
			if(Options::get().logfiles.size() >= (size_t)Options::get().numSensors) // log
			{
				if(Options::get().live)// play back live
				{
					std::cout << "live camera manager\n";
					return new MultiLiveCameraManager();
				}	 
				else//playback from raw logs
				{
					std::cout << "log camera manager\n";
					return new MultiLogCameraManager();
				}
			}
			else if(Options::get().logfiles.size() > 0 && Options::get().logfiles.size() < (size_t)Options::get().numSensors)
			{
				std::cout << "mixed camera manager\n"; 
				return new MultiMixedCameraManager();
			}
			else if (Options::get().lcm)
			{
				std::cout << "live camera manager\n";
				return new MultiLiveCameraManager();
				//return new MultiMixedCameraManager();
			}
			else
			{
				std::cout << "USB camera manager\n";
				return new MultiUsbCameraManager();
			}
		}
};


#endif /*MULTICAMERAMANAGERFACTORY_H_*/