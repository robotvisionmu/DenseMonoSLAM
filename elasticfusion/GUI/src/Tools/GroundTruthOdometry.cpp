/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "GroundTruthOdometry.h"

GroundTruthOdometry::GroundTruthOdometry(const std::string & filename)
 : last_utime(0)
{
    loadTrajectory(filename);
}

GroundTruthOdometry::~GroundTruthOdometry()
{

}

void GroundTruthOdometry::loadTrajectory(const std::string & filename)
{
    std::ifstream file;
    std::string line;
    file.open(filename.c_str());
    while (!file.eof())
    {
        unsigned long long int utime;
        float x, y, z, qx, qy, qz, qw;
        float r11, r12, r13, tx;
        float r21, r22, r23, ty;
        float r31, r32, r33, tz;
        std::getline(file, line);
        //int n = sscanf(line.c_str(), "%llu,%f,%f,%f,%f,%f,%f,%f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);
        int n = sscanf(line.c_str(), "%llu,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &utime, &r11, &r12, &r13, &tx, &r21, &r22, &r23, &ty, &r31, &r32, &r33, &tz);
            //n += sscanf(line.c_str(), "%f %f %f %f", &r21, &r22, &r23, &ty);
            // n += sscanf(line.c_str(), "%f %f %f %f", &r31, &r32, &r33, &tz);
        Eigen::Matrix4f pose;
        pose << r11, r12, r13, tx,
                r21, r22, r23, ty,
                r31, r32, r33, tz,
                0, 0, 0, 1;
        // /std::cout << utime << "\n" << tz << "\n";
        if(file.eof())
            break;

        // assert(n == 8);

        // Eigen::Quaternionf q(qw, qx, qy, qz);
        // Eigen::Vector3f t(x, y, z);

        // Eigen::Isometry3f T;
        // T.setIdentity();
        // T.pretranslate(t).rotate(q);
        // camera_trajectory[utime] = T;
        camera_trajectory[utime] = pose;
        // std::cout << "=========================" << "\n";
        // std::cout << utime << "\n" << pose << "\n";
        // std::cout << "=========================" << "\n";
    }
}

Eigen::Matrix4f GroundTruthOdometry::getTransformation(uint64_t timestamp)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    if(last_utime != 0)
    {
        std::map<uint64_t, Eigen::Isometry3f>::const_iterator it = camera_trajectory.find(last_utime);
        if (it == camera_trajectory.end())
        {
            last_utime = timestamp;
            return pose;
        }

        //Poses are stored in the file in iSAM basis, undo it
        Eigen::Matrix4f M;
        M <<  0,  0, 1, 0,
             -1,  0, 0, 0,
              0, -1, 0, 0,
              0,  0, 0, 1;
        M = Eigen::Matrix4f::Identity();
        pose = M.inverse() * camera_trajectory[timestamp] * M;
        //pose = camera_trajectory[timestamp];
    }
    else
    {
        std::map<uint64_t, Eigen::Isometry3f>::const_iterator it = camera_trajectory.find(timestamp);
        Eigen::Isometry3f ident = it->second;
        pose = Eigen::Matrix4f::Identity();
        camera_trajectory[last_utime] = ident;
    }

    last_utime = timestamp;
    // std::cout << "=========================" << "\n";
    // std::cout << last_utime << "\n" << pose << "\n";
    // std::cout << "=========================" << "\n";
    return pose;
}

Eigen::MatrixXd GroundTruthOdometry::getCovariance()
{
    Eigen::MatrixXd cov(6, 6);
    cov.setIdentity();
    cov(0, 0) = 0.1;
    cov(1, 1) = 0.1;
    cov(2, 2) = 0.1;
    cov(3, 3) = 0.5;
    cov(4, 4) = 0.5;
    cov(5, 5) = 0.5;
    return cov;
}
