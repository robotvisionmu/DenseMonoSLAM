#ifndef LCMRECEIVER_H_
#define LCMRECEIVER_H_

#include <lcm/lcm-cpp.hpp>

#include <boost/thread.hpp>

#include <atomic>

#include <Utils/Resolution.h>

class LcmReceiver
{
	public: 
		LcmReceiver(lcm::LCM & lcm);
		virtual ~LcmReceiver();

		void start();
		void stop();
	
	private:
		void receive();

		boost::thread * m_receiveThread; 

		lcm::LCM & m_lcm;

		std::atomic<bool> m_receive{false};
};
#endif //LCMRECEIVER_H_