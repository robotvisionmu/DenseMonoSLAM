#ifndef RAWLCMLOGREADER_H_
#define RAWLCMLOGREADER_H_


#include <lcm/lcm-cpp.hpp>

class RawLcmLogReader : public LogReader
{
	public:
        RawLcmLogReader(std::string file, bool flipColors)
         : LogReader(file, flipColors),
         m_logfile(file, "r"),
         m_rewound(false),
         m_done(false)
		{
			currentFrame = 0;

			depthReadBuffer = new unsigned char[Resolution::getInstance().numPixels() * 2];
			imageReadBuffer = new unsigned char[Resolution::getInstance().numPixels() * 3];
			decompressionBufferDepth = new Bytef[Resolution::getInstance().numPixels() * 2];
			decompressionBufferImage =  new Bytef[Resolution::getInstance().numPixels() * 3];

			m_depth = std::shared_ptr<unsigned short>(new unsigned short[Resolution::getInstance().numPixels() * 2]);
			m_rgb = std::shared_ptr<unsigned char>(new unsigned char[Resolution::getInstance().numPixels() * 3]);
			std::cout << "log file cstr: " << file << std::endl;
			std::cout << "log file cstr: " << m_logfile.good() << std::endl;
		}

        ~RawLcmLogReader()
        {
			delete [] depthReadBuffer;
			delete [] imageReadBuffer;
			delete [] decompressionBufferDepth;
			delete [] decompressionBufferImage;
        }

        void getNext()
        {
        	const lcm::LogEvent * le = m_logfile.readNextEvent();

			if(le == NULL)
			{
				m_done = true;
				//std::cout << "done null" << std::endl;
				return;
			}

			eflcm::Frame f;
			if(f.decode(le->data, 0, le->datalen) < 0)
			{
				return;
			}

			if(f.last)
			{
				//std::cout << "done last" << std::endl;
				m_done=true;
			}
			if(f.compressed)
			{
				unsigned long decompressedDepthSize = Resolution::getInstance().numPixels() * 2;
				uncompress(&decompressionBufferDepth[0], (unsigned long*)&decompressedDepthSize, (const Bytef*)(f.depth.data()), f.depthSize);		

				jpeg.readData((unsigned char *)f.image.data(), f.imageSize, (unsigned char *)&decompressionBufferImage[0]);

				memcpy(m_depth.get(), (unsigned short *)decompressionBufferDepth, Resolution::getInstance().numPixels() * 2);
				memcpy(m_rgb.get(), (unsigned char *)decompressionBufferImage, Resolution::getInstance().numPixels() * 3);
			}
			else
			{
				memcpy(m_depth.get(), f.depth.data(), f.depthSize);
				memcpy(m_rgb.get(), f.image.data(), f.imageSize);
			}

			if(flipColors)
    		{
        		for(int i = 0; i < Resolution::getInstance().numPixels() * 3; i += 3)
        		{
            		std::swap(m_rgb.get()[i + 0], m_rgb.get()[i + 2]);
        		}
    		}

			timestamp = f.timestamp;
			currentFrame++;
        }

        int getNumFrames()
        {
        	return std::numeric_limits<int>::max();
        }

        bool hasMore()
        {
        	return !m_done;
        }

        bool rewound()
        {
        	return m_rewound;
        }

        void rewind()
        {
			std::cout << "log file rewind: " << m_logfile.good() << std::endl;
        	m_logfile.seekToTimestamp(0);
        	m_rewound = true;
        	m_done = false;

        }

        void getBack()
        {

        }

        void fastForward(int frame)
        {

        }

        const std::string getFile()
        {
        	return file;
        }

        void setAuto(bool value)
        {

        }

        bool good()
        {
        	return m_logfile.good();
        }

	private:
		lcm::LogFile m_logfile;
		bool m_rewound;
		bool m_done;
};

#endif /*RAWLCMLOGREADER_H_*/