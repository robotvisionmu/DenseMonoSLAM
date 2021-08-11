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
    assert(argc == 3 && "Please supply the association file dir as the first argument and outfile name as second");

    std::string directory(argv[1]);

    if(directory.at(directory.size() - 1) != '/')
    {
        directory.append("/");
    }

    std::string associationFile = directory;
    associationFile.append("association.txt");

    std::ifstream asFile;
    asFile.open(associationFile.c_str());

    std::string currentLine;
    std::vector<std::string> tokens;
    std::vector<std::string> timeTokens;

    const int64_t rgbFrameSize = 640 * 480 * 3 * sizeof(unsigned char);
    const int64_t depthRawSize = 640 * 480 * sizeof(double);

    int depth_compress_buf_size = 640 * 480 * sizeof(int16_t) * 4;
    uint8_t * depth_compress_buf = (uint8_t*)malloc(depth_compress_buf_size);

    cv::Mat3b rgbImg(480, 640);
    cv::Mat1w depthImg(480, 640);

    std::string logFilename = directory;
    logFilename.append(argv[2]);
    logFilename.append(".klg");

    FILE * logFile = fopen(logFilename.c_str(), "wb+");

    int numFrames = 0;
    fwrite(&numFrames, sizeof(int32_t), 1, logFile);

    while(!asFile.eof())
    {
        getline(asFile, currentLine);
        tokenize(currentLine, tokens);

        if(tokens.size() == 0)
            break;

        std::string imageLoc = directory;
        imageLoc.append(tokens[1]);
        rgbImg = cv::imread(imageLoc);

        std::string depthLoc = directory;
        depthLoc.append(tokens[3]);
        depthImg = cv::imread(depthLoc, CV_LOAD_IMAGE_ANYDEPTH);

        tokenize(tokens[0], timeTokens, ".");

        std::string timeString = timeTokens[0];
        timeString.append(timeTokens[1]);

        unsigned long long int time;
        std::istringstream(timeString) >> time;

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

        /**
         * Format is:
         * int64_t: timestamp
         * int32_t: depthSize
         * int32_t: imageSize
         * depthSize * unsigned char: depth_compress_buf
         * imageSize * unsigned char: encodedImage->data.ptr
         */

        fwrite(&time, sizeof(int64_t), 1, logFile);
        fwrite(&depthSize, sizeof(int32_t), 1, logFile);
        fwrite(&imageSize, sizeof(int32_t), 1, logFile);
        fwrite(depth_compress_buf, depthSize, 1, logFile);
        fwrite(encodedImage->data.ptr, imageSize, 1, logFile);

        std::cout << "\rWritten frame " << numFrames << "      "; std::cout.flush();

        numFrames++;
    }

    fseek(logFile, 0, SEEK_SET);
    fwrite(&numFrames, sizeof(int32_t), 1, logFile);

    std::cout << std::endl;
    std::cout << "Done!" << std::endl;

    fclose(logFile);
    free(depth_compress_buf);

    if(encodedImage != 0)
    {
        cvReleaseMat(&encodedImage);
    }
}
