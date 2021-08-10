#include "LcmReceiver.h"

LcmReceiver::LcmReceiver(lcm::LCM & lcm)
:m_receiveThread(0),
 m_lcm(lcm)
{
	if(!m_lcm.good())
	{
		std::exit(EXIT_FAILURE);
	}
}

LcmReceiver::~LcmReceiver()
{}

void LcmReceiver::receive()
{
	 while(m_receive && m_lcm.handle() == 0);
}

void LcmReceiver::start()
{
	m_receive = true;
	
	m_receiveThread = new boost::thread(boost::bind(&LcmReceiver::receive, this));
}

void LcmReceiver::stop()
{
	m_receive = false;

	m_receiveThread->join();

	m_receiveThread = 0;
}