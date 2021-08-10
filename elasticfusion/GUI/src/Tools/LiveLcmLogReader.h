#ifndef LIVELCMLOGREADER_H_
#define LIVELCMLOGREADER_H_

#include <atomic>

//#include <opencv2/opencv.hpp>

#include "LogReader.h"
#include "CircularBuffer.h"
#include "lcmtypes/eflcm/Frame.hpp"

class LiveLcmLogReader : public LogReader
{
	public:
		LiveLcmLogReader(std::string name);
		virtual ~LiveLcmLogReader();
     	
     	void getNext();

        int getNumFrames()
        {
			return std::numeric_limits<int>::max();
        }

        bool hasMore()
        {
        	return !receivedLast || !frameBuffers.empty();
        }

        bool rewound()
        {
        	return false;
        }

        void rewind(){}

        void getBack(){}

        void fastForward(int frame){}

        const std::string getFile();

        void setAuto(bool value){}
		
		void onFrame(const eflcm::Frame * frame);

	private:
		std::string name;

        bool receivedLast;

		CircularBuffer frameBuffers;
};
#endif // LIVELCMLOGREADER_H_
