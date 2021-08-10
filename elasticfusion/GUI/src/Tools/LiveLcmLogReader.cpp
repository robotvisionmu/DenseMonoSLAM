#include "LiveLcmLogReader.h"

LiveLcmLogReader::LiveLcmLogReader(std::string name)
:LogReader("", false),
 name(name),
 frameBuffers(10)
{
	depthReadBuffer = new unsigned char[numPixels * 2];
	imageReadBuffer =  new unsigned char[numPixels * 3];
	decompressionBufferDepth = new Bytef[Resolution::getInstance().numPixels() * 2];
	decompressionBufferImage =  new Bytef[Resolution::getInstance().numPixels() * 3];

	m_depth = std::shared_ptr<unsigned short>(new unsigned short[Resolution::getInstance().numPixels() * 2]);
	m_rgb = std::shared_ptr<unsigned char>(new unsigned char[Resolution::getInstance().numPixels() * 3]);
}

LiveLcmLogReader::~LiveLcmLogReader()
{
	delete [] depthReadBuffer;
	delete [] imageReadBuffer;
	delete [] decompressionBufferDepth;
	delete [] decompressionBufferImage;
}

void LiveLcmLogReader::getNext()
{
	if(!hasMore() || frameBuffers.empty()){ return; }

	imageSize = Resolution::getInstance().numPixels() * 3;
	depthSize = Resolution::getInstance().numPixels() * 2;

	frameBuffers.pop(m_rgb, m_depth, timestamp);
}

void LiveLcmLogReader::onFrame(const eflcm::Frame * frame)
{
	if((receivedLast = frame->last) )return;

	trackOnly = frame->trackOnly;
	/*std::unique_ptr<unsigned short[]> dp;
	std::unique_ptr<unsigned char[]> im; */

	/*decompressionBufferDepth = new Bytef[numPixels * 2];
	decompressionBufferImage = new Bytef[numPixels * 3];*/

	if(frame->compressed)
	{
		std::vector<std::future<void>> tasks;

		unsigned long decompressedDepthSize = Resolution::getInstance().numPixels() * 2;
		uncompress(&decompressionBufferDepth[0], (unsigned long*)&decompressedDepthSize, (const Bytef*)(frame->depth.data()), frame->depthSize);		

		jpeg.readData((unsigned char *)frame->image.data(), frame->imageSize, (unsigned char *)&decompressionBufferImage[0]);
/*
		dp = std::unique_ptr<unsigned short[]>((unsigned short *)&decompressionBufferDepth[0]);
		im = std::unique_ptr<unsigned char[]>((unsigned char *)&decompressionBufferImage[0]);*/
	}
	else
	{
		/*dp = std::unique_ptr<unsigned short[]>(new unsigned short[frame->depthSize]);
		im = std::unique_ptr<unsigned char[]>(new unsigned char[frame->imageSize]);*/

		memcpy(&decompressionBufferDepth[0]/*dp.get()*/, frame->depth.data(), frame->depthSize);
		memcpy(&decompressionBufferImage[0]/*im.get()*/, frame->image.data(), frame->imageSize);
	}

	frameBuffers.push((unsigned char *)decompressionBufferImage/*std::move(im)*/, (unsigned short *)decompressionBufferDepth/*std::move(dp)*/, frame->timestamp);
}

const std::string LiveLcmLogReader::getFile()
{
	return name;
}