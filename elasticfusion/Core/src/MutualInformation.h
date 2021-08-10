#ifndef MUTUALINFORMATION_H_
#define MUTUALINFORMATION_H_

#include <Eigen/LU>

#include "Cuda/cudafuncs.cuh"
#include "GPUTexture.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include "KeyFrame.h"

class MutualInformation
{
public:
  MutualInformation(int num_bins_depth = 500, int num_bins_img = 64, int num_pyr_levels = 3);
  virtual ~MutualInformation();

  // float mi(GPUTexture &curr_img,
  //          KeyFrame &kf,
  //          Eigen::Matrix4f &curr_pose);
  // float mi(DeviceArray2D<unsigned char> &curr_img,
  //          Eigen::Matrix4f &curr_pose,
  //          DeviceArray2D<float> &kf_vert,
  //          DeviceArray2D<unsigned char> &kf_img,
  //          Eigen::Matrix4f &kf_pose);
  float nid(DeviceArray2D<unsigned char> &curr_img,
            Eigen::Matrix4f &curr_pose,
            DeviceArray2D<float> &kf_vert,
            DeviceArray2D<unsigned char> &kf_img,
            Eigen::Matrix4f &kf_pose);
  float nidImg(DeviceArray2D<unsigned char> &curr_img,
               KeyFrame &kf,
               Eigen::Matrix4f &curr_pose,
               int pyr_level = 0,
               bool save = false);
  float nidDepth(DeviceArray2D<float> & curr_depth,
                 KeyFrame &kf,
                 Eigen::Matrix4f &curr_pose,
                 const float &max_depth,
                 int pyr_level = 0,
                 bool save = false);

  int &numBinsDepth() { return num_bins_depth; }
  int &numBinsImg() { return num_bins_img; }

  float lastMIScore;
  float lastNIDImgScore;
  float lastNIDDepthScore;

private:
  const int neighborhood_size;
  int num_bins_img;
  int num_bins_depth;
  const float max_depth_rgb;
  int height;
  int width;

  // DeviceArray2D<float> vmap_kf;
  // DeviceArray2D<unsigned char> img_kf;
  // DeviceArray2D<unsigned char> img_curr;
  // DeviceArray<float> vmap_tmp;
  // DeviceArray<unsigned char> img_tmp;
  // DeviceArray2D<float> dmap_curr;
  Eigen::Matrix4f pose_kf;
  CameraModel intrinsics;

  std::vector<DeviceArray2D<unsigned char>> resized_imgs;//(curr_img.rows(), curr_img.cols());
  std::vector<DeviceArray2D<float>> resized_depths;//(curr_img.rows(), curr_img.cols());
  std::vector<DeviceArray2D<unsigned char>> resized_imgs_old;//(curr_img.rows(), curr_img.cols());
  std::vector<DeviceArray2D<float>> resized_depths_old;//(curr_img.rows(), curr_img.cols());

  // void init_currentframe(GPUTexture &img);
  // void init_currentdepth(GPUTexture &depth);
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
#endif /*MUTUALINFORMATION_H_*/