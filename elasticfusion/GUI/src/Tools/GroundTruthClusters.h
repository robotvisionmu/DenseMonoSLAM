#ifndef GROUNDTRUTHCLUSTERS_H_
#define GROUNDTRUTHCLUSTERS_H_

#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <tuple>
#include <algorithm>
#include <math.h>

#include <Eigen/Core>
class GroundTruthClusters
{

    public:
        GroundTruthClusters(const std::string & filename);

        virtual ~GroundTruthClusters();

        int getCluster(uint64_t timestamp);
        std::tuple<float,float,float> getClusterColor(int cluster);
    private:
        std::map<uint64_t, int> clusters;
        std::map<uint64_t, std::tuple<float, float, float>> cluster_colors;
        uint64_t last_utime;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif