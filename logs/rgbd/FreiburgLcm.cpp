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
            std::cout << "FreiburgLcm, A tool for converting TUM-RGBD compatible datasets to lcm logs. Usage:" << std::endl;
            std::cout << argv[0] << " <associations.txt> <outfile_name> -ch <lcm_channel> <parent_directory>[ -split <num_of_chunks_to_split_log_into> [-trackOnly]]" << std::endl;
            std::exit(EXIT_SUCCESS);
        }
    }

    std::string directory(argv[1]);
    std::string channel(argv[3]);
    std::string parent_directory(argv[4]);

    bool trackOnly = argc >= 9 && strcmp(argv[8], "-trackOnly") == 0 ? true : false;
    int split = argc >= 8 && strcmp(argv[6], "-split") == 0 ? atoi(argv[7]) : 1;
    std::cout << "split " << split << std::endl;
    if(directory.at(directory.size() - 1) != '/')
    {
        directory.append("/");
    }

    std::string associationFile = directory;
    associationFile.append("associations.txt");

    std::cout << "associations file: " << associationFile << std::endl;

    std::ifstream asFile;
    asFile.open(associationFile.c_str());

    std::string currentLine;
    std::vector<std::string> tokens;
    std::vector<std::string> timeTokens;
    std::vector<std::pair<int64_t, std::pair<std::string, std::string>>> associations;

    const int64_t rgbFrameSize = 640 * 480 * 3 * sizeof(unsigned char);
    const int64_t depthRawSize = 640 * 480 * sizeof(double);

    int depth_compress_buf_size = 640 * 480 * sizeof(int16_t) * 4;
    uint8_t * depth_compress_buf = (uint8_t*)malloc(depth_compress_buf_size);

    cv::Mat3b rgbImg(480, 640);
    cv::Mat1w depthImg(480, 640);
    int64_t timestamp;

    // std::string logFilename = directory;
    // logFilename.append(argv[2]);
    // logFilename.append(".lcm");

    // lcm::LogFile lcm_log(logFilename, "w");

    int numFrames = 0;

    while(!asFile.eof())
    {
        getline(asFile, currentLine);
        tokenize(currentLine, tokens);

        if(tokens.size() == 0)
            break;

        // std::string imageLoc = directory;
        // imageLoc.append(tokens[3]);
        // std::cout << imageLoc << std::endl;

        // std::string depthLoc = directory;
        // depthLoc.append(tokens[1]);

        std::string imageLoc = directory;
        std::string depthLoc = directory;

        imageLoc.append(tokens[1].compare(0, 3, "rgb") == 0 ? tokens[1] : tokens[3]);
        depthLoc.append(tokens[3].compare(0, 5, "depth") == 0 ? tokens[3] : tokens[1]);

        std::cout << depthLoc << std::endl;
        tokenize(tokens[0], timeTokens, ".");

        std::string timeString = timeTokens[0];
        timeString.append(timeTokens[1]);

        unsigned long long int time;
        std::istringstream(timeString) >> time;

        //std::cout << time << std::endl;
        std::pair<int64_t, std::pair<std::string, std::string>> association = {time, {imageLoc, depthLoc}};

        associations.push_back(association);        
    }

    int segment_size = associations.size() /split;
    for(int j = 0; j < split; j++)
    {
        std::string logFilename = directory;
        logFilename.append(argv[2]);
        logFilename.append("." + std::to_string(j));
        logFilename.append(".lcm");

        lcm::LogFile lcm_log(logFilename, "w");

        for(int i = j * segment_size; i < (j + 1) * segment_size; i++)
        {
            //std::cout << "time: " << associations[i].first << ", depth location: " <<  associations[i].second.second << ", rgb location: "<< associations[i].second.first << std::endl;
            rgbImg = cv::imread(associations[i].second.first);
            cv::imshow("colour", rgbImg);
            depthImg = cv::imread(associations[i].second.second, CV_LOAD_IMAGE_ANYDEPTH);
            timestamp = associations[i].first;

            unsigned long compressed_size = depth_compress_buf_size;
            boost::thread_group threads;

            threads.add_thread(new boost::thread(&encodeJpeg, (cv::Vec<unsigned char, 3> *)rgbImg.data));

            for(unsigned int i = 0; i < 480; i++)
            {
                for(unsigned int j = 0; j < 640; j++)
                {
                    depthImg.at<unsigned short>(i, j) /= 5;
                }
            }

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
            f.last = i == associations.size() - 1;
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

            std::cout << "\rWritten frame " << numFrames << "      "; std::cout.flush();

            numFrames++;

            delete [] buf;
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
