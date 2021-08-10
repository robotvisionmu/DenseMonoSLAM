#ifndef LogReader_H_
#define LogReader_H_

#ifdef WIN32
#  include <cstdint>
#endif
#include <string>
#include <memory>
#if (defined WIN32) && (defined FAR)
#  undef FAR
#endif
#include <zlib.h>
#ifndef WIN32
#  include <poll.h>
#endif
#include <Utils/Img.h>
#include <Utils/Resolution.h>

#include "JPEGLoader.h"

class LogReader
{
    public:
        LogReader(std::string file, bool flipColors)
         : flipColors(flipColors),
           timestamp(0),
           currentFrame(0),
           decompressionBufferDepth(0),
           decompressionBufferImage(0),
           file(file),
           width(Resolution::getInstance().width()),
           height(Resolution::getInstance().height()),
           numPixels(width * height)
        {}

        virtual ~LogReader()
        {}

        virtual void getNext() = 0;

        virtual int getNumFrames() = 0;

        virtual bool hasMore() = 0;

        virtual bool rewound() = 0;

        virtual void rewind() = 0;

        virtual void getBack() = 0;

        virtual void fastForward(int frame) = 0;

        virtual const std::string getFile() = 0;

        virtual void setAuto(bool value) = 0;

        virtual std::shared_ptr<unsigned short> depth()
        {
          return m_depth;
        }

        virtual std::shared_ptr<unsigned char> rgb()
        {
          return m_rgb;
        }

        bool flipColors;
        bool trackOnly;
        int64_t timestamp;

        int currentFrame;

    protected:
        Bytef * decompressionBufferDepth;
        Bytef * decompressionBufferImage;
        unsigned char * depthReadBuffer;
        unsigned char * imageReadBuffer;
        int32_t depthSize;
        int32_t imageSize;

        std::shared_ptr<unsigned short> m_depth;
        std::shared_ptr<unsigned char> m_rgb;

        const std::string file;
        FILE * fp;
        int32_t numFrames;
        int width;
        int height;
        int numPixels;

        JPEGLoader jpeg;
};
#endif //LogReader_H_