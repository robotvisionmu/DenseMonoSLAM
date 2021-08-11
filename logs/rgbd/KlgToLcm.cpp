/**
 * Simple programme to convert raw Freiburg data files
 * to klg format
 * @param argc
 * @param argv
 * @return
 */

#include <zlib.h>
#include <boost/thread.hpp>
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <lcm/lcm-cpp.hpp>
#include "lcmtypes/eflcm/Frame.hpp"

#include "RawLogReader.h"

CvMat * encodedImage = 0;

void encodeJpeg(cv::Vec<unsigned char, 3> * rgb_data)
{
    cv::Mat3b rgb(480, 640, rgb_data, 1920);

    IplImage * img = new IplImage(rgb);

    int jpeg_params[] = {CV_IMWRITE_JPEG_QUALITY, 90, 0};

    if(encodedImage != 0)
    {
        cvReleaseMat(&encodedImage);
    }

    encodedImage = cvEncodeImage(".jpg", img, jpeg_params);

    delete img;
}

void tokenize(const std::string & str, std::vector<std::string> & tokens, std::string delimiters = " ")
{
    tokens.clear();

    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}

int main(int argc, char ** argv)
{
    for(int i = 0; i < argc; i++)
    {
        if(argc < 3 || strcmp(argv[i], "-help") == 0)
        {
            std::cout << "KlgToLcm, A tool for converting Klg logs to lcm logs. Usage:" << std::endl;
            std::cout << argv[0] << " log.klg -ch <lcm_channel> [ -split <num_of_chunks_to_split_log_into> [-trackOnly]]" << std::endl;
            std::exit(EXIT_SUCCESS);
        }
    }

    std::string klg_log(argv[1]);
    std::string channel(argv[3]);

    bool trackOnly = argc >= 5 && strcmp(argv[6], "-trackOnly") == 0 ? true : false;
    int split = argc >= 8 && strcmp(argv[4], "-split") == 0 ? atoi(argv[5]) : 1;
    std::cout << "split " << split << std::endl;

    RawLogReader rlr(klg_log, false);

    const int64_t rgbFrameSize = 640 * 480 * 3 * sizeof(unsigned char);
    const int64_t depthRawSize = 640 * 480 * sizeof(double);

    int depth_compress_buf_size = 640 * 480 * sizeof(int16_t) * 4;
    uint8_t * depth_compress_buf = (uint8_t*)malloc(depth_compress_buf_size);

    cv::Mat3b rgbImg(480, 640);
    cv::Mat1w depthImg(480, 640);
    int64_t timestamp;

    std::string klg_log_filename = klg_log.substr(klg_log.find_last_of("/"), klg_log.find_last_of("."));
    std::string directory = klg_log.substr(0, klg_log.find_last_of("/"));

    int segment_size = rlr.getNumFrames() /split;
    for(int j = 0; j < split; j++)
    {
        std::string logFilename = directory;
        logFilename.append(klg_log_filename);
        logFilename.append("." + std::to_string(j));
        logFilename.append(".lcm");

        lcm::LogFile lcm_log(logFilename, "w");

        for(int i = j * segment_size; i < (j + 1) * segment_size; i++)
        {
            rlr.getNext();
            rgbImg = cv::Mat3b(480, 640, (cv::Vec3b*)rlr.rgb().get());

            depthImg = cv::Mat1w(480, 640, rlr.depth().get());
            timestamp = rlr.timestamp;

            unsigned long compressed_size = depth_compress_buf_size;
            boost::thread_group threads;

            threads.add_thread(new boost::thread(&encodeJpeg, (cv::Vec<unsigned char, 3> *)rgbImg.data));

            // for(unsigned int i = 0; i < 480; i++)
            // {
            //     for(unsigned int j = 0; j < 640; j++)
            //     {
            //         depthImg.at<unsigned short>(i, j) *= 1000.0;
            //     }
            // }

            threads.add_thread(new boost::thread(compress2,
                                                depth_compress_buf,
                                                &compressed_size,
                                                (const Bytef*)depthImg.data,
                                                640 * 480 * sizeof(short),
                                                Z_BEST_SPEED));


            threads.join_all();

            int32_t depthSize = compressed_size;
            int32_t imageSize = encodedImage->width;

            eflcm::Frame f;
            f.trackOnly = trackOnly;
            f.compressed = true;
            f.last = i == rlr.getNumFrames() - 1;
            f.depthSize =depthSize;
            f.imageSize = imageSize;
            f.depth.assign(depth_compress_buf, depth_compress_buf + depthSize);
            f.image.assign(encodedImage->data.ptr, encodedImage->data.ptr + imageSize);
            f.timestamp = timestamp;
            f.frameNumber = i;
            f.senderName = logFilename;

            unsigned char * buf = new unsigned char[f.getEncodedSize() / sizeof(unsigned char)]; 
            int bytesEncoded = f.encode((void *)buf, 0, f.getEncodedSize());

            lcm::LogEvent e;
            e.channel = channel;
            e.data = (void *) buf;
            e.datalen = f.getEncodedSize();
            e.timestamp = i * 33000;//i++ * 33000;

            lcm_log.writeEvent(&e);

            delete [] buf;

            std::cout << "Frame number: " << i << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Done!" << std::endl;

    free(depth_compress_buf);

    if(encodedImage != 0)
    {
        cvReleaseMat(&encodedImage);
    }
}
