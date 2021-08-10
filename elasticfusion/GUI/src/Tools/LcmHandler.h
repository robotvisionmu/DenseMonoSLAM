#ifndef LCMHANDLER_H_
#define LCMHANDLER_H_

#include <vector>
#include <unordered_map>

#include <lcm/lcm-cpp.hpp>

#include "LiveLcmLogReader.h"
#include "concurrent_queue.h"

class LcmHandler
{
	public:
		LcmHandler(std::unordered_map<std::string, std::shared_ptr<LiveLcmLogReader>> & demux,
					concurrent_queue<std::shared_ptr<LiveLcmLogReader>> & devices)
		:m_demux(demux),
		 m_devices(devices)
		{}

		virtual ~LcmHandler()
		{}	

		void onMessage(const lcm::ReceiveBuffer * rbuf,  const std::string & chan, const eflcm::Frame * frame)
		{
			auto device = m_demux.find(frame->senderName);

			if(device == m_demux.end())
			{	
				std::shared_ptr<LiveLcmLogReader> newDevice = std::shared_ptr<LiveLcmLogReader>(new LiveLcmLogReader(frame->senderName));
				newDevice->onFrame(frame);

				m_devices.push_back(newDevice);

				m_demux[newDevice->getFile()] = newDevice;
			}
			else
			{
				device->second->onFrame(frame);
			}
		}

	private:
		std::unordered_map<std::string, std::shared_ptr<LiveLcmLogReader>> & m_demux;
		concurrent_queue<std::shared_ptr<LiveLcmLogReader>> & m_devices;
};
#endif//LCMHANDLER