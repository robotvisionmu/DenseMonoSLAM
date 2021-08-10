#ifndef MULTILIVECAMERAMANAGER_H_
#define MULTILIVECAMERAMANAGER_H_

#include <vector>
#include <unordered_map>

#include <boost/lockfree/queue.hpp>

#include "LiveLcmLogReader.h"
#include "LcmHandler.h"
#include "concurrent_queue.h"
#include "networking/LcmReceiver.h"

#include "MultiCameraManager.h"

class MultiLiveCameraManager : public MultiCameraManager
{
	public:
		MultiLiveCameraManager()
		{
			m_lcmUrl = Options::get().lcmUrl();
			m_lcm = new lcm::LCM(m_lcmUrl);

			if(!m_lcm->good())
			{
				std::exit(EXIT_FAILURE);
			}
			
			m_channel = Options::get().lcmChannel;
			m_handler = new LcmHandler(m_demux, m_devices);
			m_lcm->subscribe(m_channel, &LcmHandler::onMessage, m_handler);

			m_receiver = new LcmReceiver(*m_lcm);
			m_receiver->start();
		}

		virtual ~MultiLiveCameraManager()
		{
			m_receiver->stop();

			delete m_receiver;
			delete m_handler;
			delete m_lcm;
		}

		std::vector<std::shared_ptr<LogReader>> devices() const 
		{
			std::vector<std::shared_ptr<LiveLcmLogReader>> devices = m_devices.snapshot();
			
			return std::vector<std::shared_ptr<LogReader>>(devices.begin(), devices.end());
		}

		void reset()
		{
			m_receiver->stop();
			
			delete m_receiver;
			delete m_handler;
			delete m_lcm;   

			m_devices.clear();
			m_demux.clear();
			
			m_lcm = new lcm::LCM(m_lcmUrl);

			if(!m_lcm->good())
			{
				std::exit(EXIT_FAILURE);
			}
			
			m_handler = new LcmHandler(m_demux, m_devices);
			m_lcm->subscribe(m_channel, &LcmHandler::onMessage, m_handler);

			m_receiver = new LcmReceiver(*m_lcm);
			m_receiver->start();
		}
	
	protected:
		LcmReceiver *m_receiver;
		LcmHandler *m_handler;

		concurrent_queue<std::shared_ptr<LiveLcmLogReader>> m_devices;
		std::unordered_map<std::string, std::shared_ptr<LiveLcmLogReader>> m_demux;

		std::string m_channel;
		std::string m_lcmUrl;
		lcm::LCM * m_lcm;
};
#endif /*MULTILIVECAMERAMANAGER_H_*/