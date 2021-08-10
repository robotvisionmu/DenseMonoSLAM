#include "MutualInformation.h"

MutualInformation::MutualInformation(int num_bins_depth, int num_bins_img, int num_pyr_levels)
    : lastMIScore(0),
      lastNIDImgScore(0),
      lastNIDDepthScore(0),
      neighborhood_size(4),
      num_bins_img(num_bins_img),
      num_bins_depth(num_bins_depth),
      max_depth_rgb(6.0),
      intrinsics(Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(),
                 Intrinsics::getInstance().cx(),
                 Intrinsics::getInstance().cy()) {
  height = Resolution::getInstance().height();
  width = Resolution::getInstance().width();

  resized_imgs.resize(num_pyr_levels);
  resized_depths.resize(num_pyr_levels);
  resized_imgs_old.resize(num_pyr_levels);
  resized_depths_old.resize(num_pyr_levels);

  for(int i = 0; i < num_pyr_levels; i++)
  {
    int h = height >> i;
    int w = width >> i;

    resized_imgs[i].create(h, w);
    resized_depths[i].create(h, w);
    resized_imgs_old[i].create(h, w);
    resized_depths_old[i].create(h, w);
  }

  // img_curr.create(height, width);
  // img_tmp.create(height * width);
  // dmap_curr.create(height, width);
}

MutualInformation::~MutualInformation() {
  // img_curr.release();
  // img_tmp.release();
  // dmap_curr.release();
}

// void MutualInformation::init_currentframe(GPUTexture &img) {
//   cudaArray *textPtr;

//   cudaGraphicsMapResources(1, &(img.cudaRes));
//   cudaGraphicsSubResourceGetMappedArray(&textPtr, img.cudaRes, 0, 0);
//   imageBGRToIntensity(textPtr, img_curr);
//   cudaGraphicsUnmapResources(1, &img.cudaRes);

//   cudaDeviceSynchronize();
// }

// void MutualInformation::init_currentdepth(GPUTexture &depth_tex) {
//   cudaArray *textPtr;

//   cudaGraphicsMapResources(1, &(depth_tex.cudaRes));
//   cudaGraphicsSubResourceGetMappedArray(&textPtr, depth_tex.cudaRes, 0, 0);
//   cudaMemcpy2DFromArray(dmap_curr.ptr(), dmap_curr.step(), textPtr, 0, 0,
//                         dmap_curr.colsBytes(), dmap_curr.rows(),
//                         cudaMemcpyDeviceToDevice);
//   cudaGraphicsUnmapResources(1, &depth_tex.cudaRes);

//   cudaDeviceSynchronize();
// }

// float MutualInformation::mi(GPUTexture &curr_img, KeyFrame &kf,
//                             Eigen::Matrix4f &curr_pose) {
//   init_currentframe(curr_img);

//   float mutual_info = 0.0f;

//   Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_kf =
//       kf.pose().topLeftCorner(3, 3);
//   float3 t_kf =
//       *reinterpret_cast<float3 *>(kf.pose().topRightCorner(3, 1).data());

//   Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_curr =
//       curr_pose.inverse().topLeftCorner(3, 3);
//   float3 t_curr =
//       *reinterpret_cast<float3 *>(curr_pose.topRightCorner(3, 1).data());

//   mat33 device_Rcam_curr = R_curr;
//   mat33 device_Rcam_kf = R_kf;

//   computeMutualInfo(kf.vmap(), kf.img(), device_Rcam_kf, t_kf, img_curr,
//                     device_Rcam_curr, t_curr, mutual_info, num_bins_img,
//                     intrinsics);

//   lastMIScore = mutual_info;
//   return mutual_info;
// }

// float MutualInformation::mi(DeviceArray2D<unsigned char> &curr_img,
//                             Eigen::Matrix4f &curr_pose,
//                             DeviceArray2D<float> &kf_vert,
//                             DeviceArray2D<unsigned char> &kf_img,
//                             Eigen::Matrix4f &kf_pose) {
//   // init_currentframe(curr_img);

//   float mutual_info = 0.0f;

//   Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_kf =
//       kf_pose.topLeftCorner(3, 3);
//   float3 t_kf =
//       *reinterpret_cast<float3 *>(kf_pose.topRightCorner(3, 1).data());

//   Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_curr =
//       curr_pose.inverse().topLeftCorner(3, 3);
//   float3 t_curr =
//       *reinterpret_cast<float3 *>(curr_pose.topRightCorner(3, 1).data());

//   mat33 device_Rcam_curr = R_curr;
//   mat33 device_Rcam_kf = R_kf;

//   computeMutualInfo(kf_vert, kf_img, device_Rcam_kf, t_kf, curr_img,
//                     device_Rcam_curr, t_curr, mutual_info, num_bins_img,
//                     intrinsics);

//   lastMIScore = mutual_info;
//   return mutual_info;
// }

float MutualInformation::nid(DeviceArray2D<unsigned char> &curr_img,
                             Eigen::Matrix4f &curr_pose,
                             DeviceArray2D<float> &kf_vert,
                             DeviceArray2D<unsigned char> &kf_img,
                             Eigen::Matrix4f &kf_pose) {
  float normalised_info_dist = 0.0f;

  Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_kf =
      kf_pose.topLeftCorner(3, 3);
  float3 t_kf =
      *reinterpret_cast<float3 *>(kf_pose.topRightCorner(3, 1).data());

  Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_curr_inverse =
      curr_pose.inverse().topLeftCorner(3, 3);
  float3 t_curr =
      *reinterpret_cast<float3 *>(curr_pose.topRightCorner(3, 1).data());

  mat33 device_Rcam_curr_inverse = R_curr_inverse;
  mat33 device_Rcam_kf = R_kf;

  // computeNIDRGB(kf_vert, kf_img,
  //               device_Rcam_kf, t_kf,
  //               curr_img, device_Rcam_curr_inverse, t_curr,
  //               normalised_info_dist, num_bins, intrinsics);

  lastNIDImgScore = normalised_info_dist;
  return normalised_info_dist;
}

float MutualInformation::nidImg(DeviceArray2D<unsigned char> &curr_img,
                                KeyFrame &kf, Eigen::Matrix4f &curr_pose,
                                int pyr_level,
                                bool save) {
  float normalised_info_dist = 0.0f;

  Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_kf =
      kf.pose().topLeftCorner(3, 3);
  float3 t_kf =
      *reinterpret_cast<float3 *>(kf.pose().topRightCorner(3, 1).data());

  Eigen::Matrix<float, 3, 3, Eigen::RowMajorBit> R_curr_inverse =
      curr_pose.inverse().topLeftCorner(3, 3);
  float3 t_curr =
      *reinterpret_cast<float3 *>(curr_pose.topRightCorner(3, 1).data());

  mat33 device_Rcam_curr_inverse = R_curr_inverse;
  mat33 device_Rcam_kf = R_kf;

  kf.img().copyTo(resized_imgs[0]);
  kf.dmap().copyTo(resized_depths[0]);
  kf.old_img().copyTo(resized_imgs_old[0]);
  kf.old_dmap().copyTo(resized_depths_old[0]);

  for(int i = 1; i <= pyr_level; i++){
    pyrDownUcharGauss(resized_imgs[i - 1], resized_imgs[i]);
    pyrDownGaussF(resized_depths[i - 1], resized_depths[i]);
    pyrDownUcharGauss(resized_imgs_old[i - 1], resized_imgs_old[i]);
    pyrDownGaussF(resized_depths_old[i - 1], resized_depths_old[i]);
  }

  // computeNIDImg(resized_imgs[pyr_level], resized_imgs_old[pyr_level], resized_depths[pyr_level], resized_depths_old[pyr_level], curr_img, normalised_info_dist, num_bins_img, save);
  computeNIDImg(resized_imgs[pyr_level], resized_imgs_old[pyr_level], resized_depths[pyr_level], resized_depths_old[pyr_level], curr_img, normalised_info_dist, num_bins_img, save);

 
  lastNIDImgScore = normalised_info_dist;
  return normalised_info_dist;
}

//flaky! assume NID img has been called first to fill buffers
float MutualInformation::nidDepth(DeviceArray2D<float> &curr_depth,
                                  KeyFrame &kf, Eigen::Matrix4f &curr_pose,
                                  const float &max_depth, int pyr_level, bool save) {
  // init_currentdepth(curr_depth);
  float normalised_info_dist = 0.0f;

  // DeviceArray2D<float> resized_depth(curr_depth.rows(), curr_depth.cols());
  // DeviceArray2D<float> resized_depth_old(curr_depth.rows(), curr_depth.cols());

  // pyrDownGaussF(kf.dmap(), resized_depth);
  // pyrDownGaussF(kf.old_dmap(), resized_depth_old);

  // computeNIDDepthSmem(resized_depths[pyr_level], resized_depths_old[pyr_level], curr_depth, normalised_info_dist,
  //                 num_bins_depth, max_depth * 1000, save);
  computeNIDDepth(resized_depths[pyr_level], resized_depths_old[pyr_level], curr_depth, normalised_info_dist,
                  num_bins_depth, max_depth * 1000, save);

  lastNIDDepthScore = normalised_info_dist;
  return normalised_info_dist;
}