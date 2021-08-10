#ifndef MULTILOGCAMERAMANAGER_H_
#define MULTILOGCAMERAMANAGER_H_


#include "MultiCameraManager.h"
#include "RawLcmLogReader.h"
#include "RawLogReader.h"

class MultiLogCameraManager : public MultiCameraManager
{
	public:
		MultiLogCameraManager()
		{
			std::vector<std::string> logfiles = Options::get().logfiles;
			bool flip = Options::get().flip; 

			for(auto & lf : logfiles)
			{
				// std::shared_ptr<RawLcmLogReader> device = std::shared_ptr<RawLcmLogReader>(new RawLcmLogReader(lf, flip));
				std::size_t dot_pos = lf.find_last_of(".");
				std::string ext = lf.substr(dot_pos + 1);
				std::shared_ptr<LogReader> device = ext.compare("lcm") == 0 ?
											std::shared_ptr<LogReader>(new RawLcmLogReader(lf, flip)) :
											std::shared_ptr<LogReader>(new RawLogReader(lf, flip));

				//if(device->good())
				//{
					std::cout << "log opened for reading: " << lf << std::endl;
					m_devices.push_back(device);	
				//}
				//else
				//{
				//	std::cout << "couldn't open log: " << lf << std::endl;
				//}				
			}
		}

		virtual ~MultiLogCameraManager(){}

		std::vector<std::shared_ptr<LogReader>> devices() const 
		{
			return std::vector<std::shared_ptr<LogReader>>(m_devices.begin(), m_devices.end());
		}
		
		void reset()
		{
			for(auto & d : m_devices)
			{
				d->rewind();
			}
		}

	private:
		//std::vector<std::shared_ptr<RawLcmLogReader>> m_devices;
		std::vector<std::shared_ptr<LogReader>> m_devices;
};


#endif /*MULTILOGCAMERAMANAGER_H_*/