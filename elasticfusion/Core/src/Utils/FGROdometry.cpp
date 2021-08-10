// ----------------------------------------------------------------------------
// -                       Fast Global Registration                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) Intel Corporation 2016
// Qianyi Zhou <Qianyi.Zhou@gmail.com>
// Jaesik Park <syncle@gmail.com>
// Vladlen Koltun <vkoltun@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "FGROdometry.h"

using namespace Eigen;
using namespace std;

void FGROdometry::ReadFeature(const char* filepath) {
  Points pts;
  Feature feat;
  ReadFeature(filepath, pts, feat);
  // LoadFeature(pts, feat);
}

void FGROdometry::LoadFeature(const Points& pts, const Points& unnormalised_pts,
                              const Feature& feat, Eigen::Vector3f& mean,
                              KDTree& feature_tree) {
  pointcloud_.push_back(pts);
  unnormalised_pointcloud_.push_back(unnormalised_pts);
  features_.push_back(feat);
  feature_trees_.push_back(feature_tree);
  Means.push_back(mean);
}

void FGROdometry::LoadFeature(const Points& pts, const Feature& feat,
                              KDTree& feature_tree) {
  pointcloud_.push_back(pts);
  unnormalised_pointcloud_.push_back(pts);
  features_.push_back(feat);
  feature_trees_.push_back(feature_tree);
}

void FGROdometry::LoadFeature(const Points& pts, const BriskFeature& feat) {
  pointcloud_.push_back(pts);
  unnormalised_pointcloud_.push_back(pts);
  brisk_features_.push_back(feat);
}
void FGROdometry::ClearFeature() {
  pointcloud_.clear();
  unnormalised_pointcloud_.clear();
  mean_shifted_pointclouds_.clear();
  normals_.clear();
  features_.clear();
  feature_trees_.clear();
  brisk_features_.clear();
  corres_.clear();
  line_processes_.clear();
  Means.clear();
  scales_.clear();
  GlobalScale = 1.0f;
  StartScale = 1.0f;

  TransOutput_ = Eigen::Matrix4f::Identity();
}

void FGROdometry::ReadFeature(const char* filepath, Points& pts,
                              Feature& feat) {
  printf("ReadFeature ... ");
  FILE* fid = fopen(filepath, "rb");
  int nvertex;
  fread(&nvertex, sizeof(int), 1, fid);
  int ndim;
  fread(&ndim, sizeof(int), 1, fid);

  // read from feature file and fill out pts and feat
  for (int v = 0; v < nvertex; v++) {
    Vector3f pts_v;
    fread(&pts_v(0), sizeof(float), 3, fid);

    VectorXf feat_v(ndim);
    fread(&feat_v(0), sizeof(float), ndim, fid);

    pts.push_back(pts_v);
    feat.push_back(feat_v);
  }
  fclose(fid);
  printf("%d points with %d feature dimensions.\n", nvertex, ndim);
}

void FGROdometry::computeFeaturesGPU(std::vector<float>& points,
                                     std::vector<float>& normals, int height,
                                     int width) {
  TICK("FGROdom::FPFH::Format");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_normals(
      new pcl::PointCloud<pcl::PointXYZ>);

  std::cout << "======================================\n";
  int num_pixels = height * width;
  std::cout << "num points: " << points.size() << std::endl;
  std::cout << "num normals: " << normals.size()<< std::endl;
  for (int x = 0; x < width; x++) {
    for(int y = 0; y < height; y++)
    {
      
      float px = points[y * width * 4 + (x * 4) + 0];
      float py = points[y * width * 4 + (x * 4) + 1];
      float pz = points[y * width * 4 + (x * 4) + 2];
      float nx = normals[y * width * 4 + (x * 4) + 0];
      float ny = normals[y * width * 4 + (x * 4) + 1];
      float nz = normals[y * width * 4 + (x * 4) + 2];
      if (std::isnan(px) || std::isnan(py) || std::isnan(pz) || std::isinf(px) ||
          std::isinf(py) || std::isinf(pz) || std::isnan(nx) || std::isnan(ny) ||
          std::isnan(nz) || std::isinf(nx) || std::isinf(ny) || std::isinf(nz))
        continue;

      cloud->push_back(pcl::PointXYZ(px, py, pz));
      cloud_normals->push_back(pcl::PointXYZ(nx, ny, nz));
    }
  }
  std::cout << "cloud size: " << cloud->size()<< std::endl;

  cloud->is_dense = true;
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud_normals->is_dense = true;
  cloud_normals->width = cloud_normals->points.size();
  cloud_normals->height = 1;
  TOCK("FGROdom::FPFH::Format");

  TICK("FGROdom::FPFH::KDTree");
  // Note copied from pcl gpu/features/test/data_source.hpp
  // std::vector<std::vector<int>> neighbors_all;
  // std::vector<int> data;
  // std::vector<int> sizes;
  // std::vector<float> dists;
  // int max_nn_size;

  // pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(
  //     new pcl::KdTreeFLANN<pcl::PointXYZ>);
  // kdtree->setInputCloud(cloud);
  // size_t cloud_size = cloud->points.size();

  // neighbors_all.resize(cloud_size);
  // int k = 128;
  // for (size_t i = 0; i < cloud_size; ++i) {
  //   kdtree->nearestKSearch(cloud->points[i], k, neighbors_all[i], dists);
  //   sizes.push_back((int)neighbors_all[i].size());
  // }

  // max_nn_size = *max_element(sizes.begin(), sizes.end());

  // data.resize(max_nn_size * neighbors_all.size());
  // pcl::gpu::PtrStep<int> ps(&data[0], max_nn_size * sizeof(int));
  // for (size_t i = 0; i < neighbors_all.size(); ++i)
  //   std::copy(neighbors_all[i].begin(), neighbors_all[i].end(), ps.ptr(i));
  TOCK("FGROdom::FPFH::KDTree");

  TICK("FGROdom::FPFH::Upload");
  pcl::gpu::FPFHEstimation::PointCloud cloud_gpu;
  cloud_gpu.upload(cloud->points);

  pcl::gpu::FPFHEstimation::Normals normals_gpu;
  normals_gpu.upload(cloud_normals->points);

  // pcl::gpu::NeighborIndices indices;
  // indices.upload(data, sizes, max_nn_size);
  TOCK("FGROdom::FPFH::Upload");

  pcl::gpu::DeviceArray2D<pcl::FPFHSignature33> fpfh33_features;

  TICK("FGROdom::FPFH::Compute");
  pcl::gpu::FPFHEstimation fpfh_gpu;
  fpfh_gpu.setInputCloud(cloud_gpu);
  fpfh_gpu.setInputNormals(normals_gpu);
  fpfh_gpu.setRadiusSearch(0.50, 512);
  fpfh_gpu.compute(fpfh33_features);
  // fpfh_gpu.compute(cloud_gpu, normals_gpu, indices, fpfh33_features);

  std::vector<pcl::FPFHSignature33> downloaded_features;
  int cols;
  fpfh33_features.download(downloaded_features, cols);

  std::cout << "num features: " << downloaded_features.size()<< std::endl;

  assert(downloaded_features.size() == cloud->points.size());

  Points pts;
  Points nrms;
  Feature features;
  for (int i = 0; (size_t)i < cloud->points.size(); i++) {
    Eigen::Vector3f p(cloud->at(i).x, cloud->at(i).y, cloud->at(i).z);
    Eigen::Vector3f n(cloud_normals->at(i).x, cloud_normals->at(i).y,
                      cloud_normals->at(i).z);
    
    Eigen::VectorXf f(33);
    memcpy(&f(0), downloaded_features.at(i).histogram, sizeof(float) * 33);
    if(std::isnan(f(0)))continue;
    pts.push_back(p);
    nrms.push_back(n);
    features.push_back(f);
  }

  TOCK("FGROdom::FPFH::Compute");

  TICK("FGROdom::FPFH::Finish");
  normals_.push_back(nrms);
  KDTree feature_tree(flann::KDTreeSingleIndexParams(15));
  BuildKDTree(features, &feature_tree);
  LoadFeature(pts, features, feature_tree);
  NormalizePoints();
  TOCK("FGROdom::FPFH::Finish");
  cloud_gpu.release();
  normals_gpu.release();
  fpfh33_features.release();
  std::cout << "num point clouds: " << pointcloud_.size()<< std::endl;
  std::cout << "======================================\n";
}

void FGROdometry::computeFeaturesSIFT2D(
    std::vector<float>& points, const std::shared_ptr<unsigned char>& rgb,
    std::vector<float>& normals, const int& height, const int& width) {
  TICK("FGROdom::SIFT::Format");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>(width, height));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_normals(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (int yImg = 0; yImg < height; yImg++) {
    for (int xImg = 0; xImg < width; xImg++) {
      int i = (yImg * Resolution::getInstance().width()) + xImg;
      float px = points[i];
      float py = points[Resolution::getInstance().numPixels() + i];
      float pz = points[(Resolution::getInstance().numPixels() * 2) + i];
      float nx = normals[i];
      float ny = normals[Resolution::getInstance().numPixels() + i];
      float nz = normals[(Resolution::getInstance().numPixels() * 2) + i];
      uint8_t r = rgb.get()[i * 3 + 0];
      uint8_t g = rgb.get()[i * 3 + 1];
      uint8_t b = rgb.get()[i * 3 + 2];

      if (std::isnan(px) || std::isnan(py) || std::isnan(pz) ||
          std::isinf(px) || std::isinf(py) || std::isinf(pz) ||
          std::isnan(nx) || std::isnan(ny) || std::isnan(nz) ||
          std::isinf(nx) || std::isinf(ny) || std::isinf(nz))
        continue;

      pcl::PointXYZRGB p(r, g, b);
      pcl::PointXYZ p_xyz(px, py, pz);
      p.x = px;
      p.y = py;
      p.z = pz;
      cloud->at(xImg, yImg) = p;
      cloud_xyz->push_back(pcl::PointXYZ(px, py, pz));
      cloud_normals->push_back(pcl::PointXYZ(nx, ny, nz));
    }
  }
  TOCK("FGROdom::SIFT::Format");

  TICK("FGROdom::SIFT::SIFT");
  const float min_scale = 0.08f;
  const int n_octaves = 3;
  const int n_scales_per_octave = 8;
  const float min_contrast = 0.01f;
  pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale>::Ptr cloud_keypoints(
      new pcl::PointCloud<pcl::PointWithScale>);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr sift_tree(
      new pcl::search::KdTree<pcl::PointXYZRGB>());
  sift.setSearchMethod(sift_tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(cloud);
  sift.compute(*cloud_keypoints);
  TOCK("FGROdom::SIFT::SIFT");

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypoints_xyz(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; (size_t)i < cloud_keypoints->size(); i++) {
    pcl::PointXYZ p;
    p.x = cloud_keypoints->points[i].x;
    p.y = cloud_keypoints->points[i].y;
    p.z = cloud_keypoints->points[i].z;
    cloud_keypoints_xyz->push_back(p);
  }

  TICK("FGROdom::SIFT::FPFH");
  pcl::gpu::FPFHEstimation::PointCloud cloud_gpu;
  cloud_gpu.upload(cloud_xyz->points);
  pcl::gpu::FPFHEstimation::PointCloud keypoints_gpu;
  keypoints_gpu.upload(cloud_keypoints_xyz->points);
  pcl::gpu::FPFHEstimation::Normals normals_gpu;
  normals_gpu.upload(cloud_normals->points);

  pcl::gpu::DeviceArray2D<pcl::FPFHSignature33> fpfh33_features;

  pcl::gpu::FPFHEstimation fpfh_gpu;
  fpfh_gpu.setInputCloud(keypoints_gpu);
  fpfh_gpu.setInputNormals(normals_gpu);
  fpfh_gpu.setSearchSurface(cloud_gpu);
  fpfh_gpu.setRadiusSearch(0.50, 128);
  fpfh_gpu.compute(fpfh33_features);

  std::vector<pcl::FPFHSignature33> downloaded_features;
  int cols;
  fpfh33_features.download(downloaded_features, cols);
  TOCK("FGROdom::SIFT::FPFH");

  TICK("FGROdom::SIFT::Finish");
  Points pts;
  Points nrms;
  Feature features;
  for (int i = 0; (size_t)i < cloud_keypoints->points.size(); i++) {
    Eigen::Vector3f p(cloud_keypoints->at(i).x, cloud_keypoints->at(i).y,
                      cloud_keypoints->at(i).z);
    Eigen::Vector3f n(cloud_normals->at(i).x, cloud_normals->at(i).y,
                      cloud_normals->at(i).z);
    pts.push_back(p);
    nrms.push_back(n);

    Eigen::VectorXf f(33);
    pcl::FPFHSignature33 descriptor = downloaded_features[i];
    memcpy(&f(0), descriptor.histogram, sizeof(float) * 33);
    features.push_back(f);
  }

  normals_.push_back(nrms);
  KDTree feature_tree(flann::KDTreeSingleIndexParams(15));
  BuildKDTree(features, &feature_tree);
  LoadFeature(pts, features, feature_tree);
  NormalizePoints();
  cloud_gpu.release();
  keypoints_gpu.release();
  normals_gpu.release();
  fpfh33_features.release();
  TOCK("FGROdom::SIFT::Finish");
}

void FGROdometry::computeFeatures(std::vector<float>& points,
                                  std::vector<float>& normals) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
      new pcl::PointCloud<pcl::Normal>);

  for (int i = 0; i < Resolution::getInstance().numPixels(); i++) {
    float px = points[i];
    float py = points[Resolution::getInstance().numPixels() + i];
    float pz = points[(Resolution::getInstance().numPixels() * 2) + i];
    float nx = normals[i];
    float ny = normals[Resolution::getInstance().numPixels() + i];
    float nz = normals[(Resolution::getInstance().numPixels() * 2) + i];
    if (std::isnan(px) || std::isnan(py) || std::isnan(pz) || std::isinf(px) ||
        std::isinf(py) || std::isinf(pz) || std::isnan(nx) || std::isnan(ny) ||
        std::isnan(nz) || std::isinf(nx) || std::isinf(ny) || std::isinf(nz))
      continue;

    cloud->push_back(pcl::PointXYZ(px, py, pz));
    cloud_normals->push_back(pcl::Normal(nx, ny, nz));
  }
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud(cloud);
  fpfh.setInputNormals(cloud_normals);

  // Create an empty kdtree representation, and pass it to the FPFH estimation
  // object.
  // Its content will be filled inside the object, based on the given input
  // dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>);
  fpfh.setSearchMethod(tree);

  // Output datasets
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(
      new pcl::PointCloud<pcl::FPFHSignature33>());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to
  // estimate the surface normals!!!
  fpfh.setRadiusSearch(0.12);

  // Compute the features
  fpfh.compute(*fpfhs);

  Points pts;
  Points nrms;
  Feature features;
  for (int i = 0; i < cloud->size(); i++) {
    Eigen::Vector3f p(cloud->at(i).x, cloud->at(i).y, cloud->at(i).z);
    Eigen::Vector3f n(cloud_normals->at(i).normal_x,
                      cloud_normals->at(i).normal_y,
                      cloud_normals->at(i).normal_z);
    pts.push_back(p);
    nrms.push_back(n);
    Eigen::VectorXf f(33);
    memcpy(&f(0), fpfhs->at(i).histogram, sizeof(float) * 33);
    features.push_back(f);
  }
  normals_.push_back(nrms);
  KDTree feature_tree(flann::KDTreeSingleIndexParams(15));
  BuildKDTree(features, &feature_tree);
  LoadFeature(pts, features, feature_tree);
}

template <typename T, typename A>
void FGROdometry::BuildKDTree(const vector<T, A>& data, KDTree* tree) {
  int rows, dim;
  rows = (int)data.size();
  dim = (int)data[0].size();
  std::vector<float> dataset(rows * dim);
  flann::Matrix<float> dataset_mat(&dataset[0], rows, dim);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < dim; j++) dataset[i * dim + j] = data[i][j];
  KDTree temp_tree(dataset_mat, flann::KDTreeSingleIndexParams(15));
  temp_tree.buildIndex();
  *tree = temp_tree;
}

template <typename T>
void FGROdometry::SearchKDTree(KDTree* tree, const T& input,
                               std::vector<int>& indices,
                               std::vector<float>& dists, int nn) {
  int rows_t = 1;
  int dim = input.size();

  std::vector<float> query;
  query.resize(rows_t * dim);
  for (int i = 0; i < dim; i++) query[i] = input(i);
  flann::Matrix<float> query_mat(&query[0], rows_t, dim);

  indices.resize(rows_t * nn);
  dists.resize(rows_t * nn);
  flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
  flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);

  tree->knnSearch(query_mat, indices_mat, dists_mat, nn,
                  flann::SearchParams(256));
}

template <typename T, typename A>
void FGROdometry::BuildLshIndex(const std::vector<T, A>& data,
                                LshIndex* index) {
  int rows, dim;
  rows = (int)data.size();
  dim = (int)data[0].size();
  std::vector<unsigned char> dataset(rows * dim);
  flann::Matrix<unsigned char> dataset_mat(&dataset[0], rows, dim);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < dim; j++) dataset[i * dim + j] = data[i][j];
  LshIndex temp_index(dataset_mat, flann::LshIndexParams());
  temp_index.buildIndex();
  *index = temp_index;
}

template <typename T>
void FGROdometry::SearchLshIndex(LshIndex* index, const T& input,
                                 std::vector<int>& indices,
                                 std::vector<unsigned int>& dists, int nn) {
  int rows_t = 1;
  int dim = input.size();

  std::vector<uint8_t> query;
  query.resize(rows_t * dim);
  for (int i = 0; i < dim; i++) query[i] = input(i);
  flann::Matrix<uint8_t> query_mat(&query[0], rows_t, dim);

  indices.resize(rows_t * nn);
  dists.resize(rows_t * nn);
  flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
  flann::Matrix<unsigned int> dists_mat(&dists[0], rows_t, nn);

  index->knnSearch(query_mat, indices_mat, dists_mat, nn,
                   flann::SearchParams(128));
}

void FGROdometry::AdvancedMatching() {
  int fi = 0;
  int fj = 1;

  // printf("Advanced matching : [%d - %d]\n", fi, fj);
  bool swapped = false;

  if (pointcloud_[fj].size() > pointcloud_[fi].size()) {
    int temp = fi;
    fi = fj;
    fj = temp;
    swapped = true;
  }

  int nPti = pointcloud_[fi].size();
  int nPtj = pointcloud_[fj].size();
  std::cout << "nPti: " << nPti << " , Nptj: " << nPtj << std::endl;
  bool crosscheck = true;
  bool tuple = true;

  std::vector<int> corres_K, corres_K2;
  std::vector<float> dis;
  std::vector<unsigned int> hamming_dis;
  std::vector<int> ind;

  std::vector<std::pair<int, int>> corres;
  std::vector<std::pair<int, int>> corres_cross;
  std::vector<std::pair<int, int>> corres_ij;
  std::vector<std::pair<int, int>> corres_ji;

  std::vector<int> i_to_j(nPti, -1);

  ///////////////////////////
  /// BUILD FLANNTREE
  ///////////////////////////
  if (features_.size() > 0) {
    // KDTree feature_tree_i(flann::KDTreeSingleIndexParams(15));
    // BuildKDTree(features_[fi], &feature_tree_i);
    KDTree feature_tree_i = feature_trees_[fi];

    // KDTree feature_tree_j(flann::KDTreeSingleIndexParams(15));
    // BuildKDTree(features_[fj], &feature_tree_j);
    KDTree feature_tree_j = feature_trees_[fj];
    ///////////////////////////
    /// INITIAL MATCHING
    ///////////////////////////

    for (int j = 0; j < nPtj; j += 1) {
      SearchKDTree(&feature_tree_i, features_[fj][j], corres_K, dis, 1);
      int i = corres_K[0];
      if(i < 0 || i > features_[fi].size())
      {
        std::cout << "i out of range j = " << j << " -> i = " << i << std::endl;
        for(int x = 0; x < features_[fj].size(); x++ )
        {
          if(std::isnan(features_[fj][x](0)))
          {
            std::cout << "=======================" << std::endl;
            std::cout << "feature " << x << " = " << features_[fj][x] << std::endl;
          }
        }
        std::cout << "point j = " << pointcloud_[fj][j] << std::endl;
      }
      if (i_to_j[i] == -1) {
        SearchKDTree(&feature_tree_j, features_[fi][i], corres_K, dis, 1);
        int ij = corres_K[0];
        i_to_j[i] = ij;
      }
      corres_ji.push_back(std::pair<int, int>(i, j));
    }
  } else {
    LshIndex feature_index_i(flann::KDTreeSingleIndexParams(15));
    BuildLshIndex(brisk_features_[fi], &feature_index_i);

    LshIndex feature_index_j(flann::KDTreeSingleIndexParams(15));
    BuildLshIndex(brisk_features_[fj], &feature_index_j);
    ///////////////////////////
    /// INITIAL MATCHING
    ///////////////////////////

    for (int j = 0; j < nPtj; j += 1) {
      SearchLshIndex(&feature_index_i, brisk_features_[fj][j], corres_K,
                     hamming_dis, 1);
      int i = corres_K[0];

      if (i_to_j[i] == -1) {
        SearchLshIndex(&feature_index_j, brisk_features_[fi][i], corres_K,
                       hamming_dis, 1);
        int ij = corres_K[0];
        i_to_j[i] = ij;
      }
      corres_ji.push_back(std::pair<int, int>(i, j));
    }
  }

  for (int i = 0; i < nPti; i++) {
    if (i_to_j[i] != -1) corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
  }

  int ncorres_ij = corres_ij.size();
  int ncorres_ji = corres_ji.size();

  // corres = corres_ij + corres_ji;
  for (int i = 0; i < ncorres_ij; ++i)
    corres.push_back(
        std::pair<int, int>(corres_ij[i].first, corres_ij[i].second));
  for (int j = 0; j < ncorres_ji; ++j)
    corres.push_back(
        std::pair<int, int>(corres_ji[j].first, corres_ji[j].second));

  // printf("\t[initial matching] Number of points that remain: %d\n",
  // (int)corres.size());

  ///////////////////////////
  /// CROSS CHECK
  /// input : corres_ij, corres_ji
  /// output : corres
  ///////////////////////////
  if (crosscheck) {
    // printf("\t[cross check] ");

    // build data structure for cross check
    corres.clear();
    corres_cross.clear();
    std::vector<std::vector<int>> Mi(nPti);
    std::vector<std::vector<int>> Mj(nPtj);

    int ci, cj;
    for (int i = 0; i < ncorres_ij; ++i) {
      ci = corres_ij[i].first;
      cj = corres_ij[i].second;
      Mi[ci].push_back(cj);
    }
    for (int j = 0; j < ncorres_ji; ++j) {
      ci = corres_ji[j].first;
      cj = corres_ji[j].second;
      Mj[cj].push_back(ci);
    }

    // cross check
    for (int i = 0; i < nPti; ++i) {
      for (int ii = 0; ii < Mi[i].size(); ++ii) {
        int j = Mi[i][ii];
        for (int jj = 0; jj < Mj[j].size(); ++jj) {
          if (Mj[j][jj] == i) {
            corres.push_back(std::pair<int, int>(i, j));
            corres_cross.push_back(std::pair<int, int>(i, j));
          }
        }
      }
    }
    // printf("Number of points that remain after cross-check: %d\n",
    //       (int)corres.size());
  }

  ///////////////////////////
  /// TUPLE CONSTRAINT
  /// input : corres
  /// output : corres
  ///////////////////////////
  if (tuple) {
    srand(time(NULL));

    // printf("\t[tuple constraint] ");
    int rand0, rand1, rand2;
    int idi0, idi1, idi2;
    int idj0, idj1, idj2;
    float scale = tuple_scale_;
    int ncorr = corres.size();
    int number_of_trial = ncorr * 100;
    std::vector<std::pair<int, int>> corres_tuple;

    int cnt = 0;
    int i;
    for (i = 0; i < number_of_trial; i++) {
      rand0 = rand() % ncorr;
      rand1 = rand() % ncorr;
      rand2 = rand() % ncorr;

      idi0 = corres[rand0].first;
      idj0 = corres[rand0].second;
      idi1 = corres[rand1].first;
      idj1 = corres[rand1].second;
      idi2 = corres[rand2].first;
      idj2 = corres[rand2].second;

      // collect 3 points from i-th fragment
      Eigen::Vector3f pti0 = pointcloud_[fi][idi0];
      Eigen::Vector3f pti1 = pointcloud_[fi][idi1];
      Eigen::Vector3f pti2 = pointcloud_[fi][idi2];

      float li0 = (pti0 - pti1).norm();
      float li1 = (pti1 - pti2).norm();
      float li2 = (pti2 - pti0).norm();

      // collect 3 points from j-th fragment
      Eigen::Vector3f ptj0 = pointcloud_[fj][idj0];
      Eigen::Vector3f ptj1 = pointcloud_[fj][idj1];
      Eigen::Vector3f ptj2 = pointcloud_[fj][idj2];

      float lj0 = (ptj0 - ptj1).norm();
      float lj1 = (ptj1 - ptj2).norm();
      float lj2 = (ptj2 - ptj0).norm();

      if ((li0 * scale < lj0) && (lj0 < li0 / scale) && (li1 * scale < lj1) &&
          (lj1 < li1 / scale) && (li2 * scale < lj2) && (lj2 < li2 / scale)) {
        corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
        corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
        corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
        cnt++;
      }

      if (cnt >= tuple_max_cnt_) break;
    }

    // printf("%d tuples (%d trial, %d actual).\n", cnt, number_of_trial, i);
    corres.clear();

    for (int i = 0; i < corres_tuple.size(); ++i)
      corres.push_back(
          std::pair<int, int>(corres_tuple[i].first, corres_tuple[i].second));
  }

  if (swapped) {
    std::vector<std::pair<int, int>> temp;
    for (int i = 0; i < corres.size(); i++)
      temp.push_back(std::pair<int, int>(corres[i].second, corres[i].first));
    corres.clear();
    corres = temp;
  }

  // printf("\t[final] matches %d.\n", (int)corres.size());
  corres_ = corres;
}

void FGROdometry::NormalizePoints() {
  float max_scale = 0;

  Vector3f mean;
  mean.setZero();

  int npti = pointcloud_.back().size();
  for (int ii = 0; ii < npti; ++ii) {
    Eigen::Vector3f p(pointcloud_.back()[ii](0), pointcloud_.back()[ii](1),
                      pointcloud_.back()[ii](2));
    mean = mean + p;
  }

  mean = mean / npti;
  Means.push_back(mean);

  // printf("normalize points :: mean[%d] = [%f %f %f]\n", i, mean(0), mean(1),
  //       mean(2));

  for (int ii = 0; ii < npti; ++ii) {
    pointcloud_.back()[ii](0) -= mean(0);
    pointcloud_.back()[ii](1) -= mean(1);
    pointcloud_.back()[ii](2) -= mean(2);

    float temp = pointcloud_.back()[ii]
                     .norm();  // because we extract mean in the previous stage.
    if (temp > max_scale) max_scale = temp;
  }

  scales_.push_back(max_scale);
  mean_shifted_pointclouds_.push_back(pointcloud_.back());
}

void FGROdometry::ScalePoints() {
  float scale = 0;
  for (const auto& s : scales_) {
    if (s > scale) scale = s;
  }

  //// mean of the scale variation
  if (use_absolute_scale_) {
    GlobalScale = 1.0f;
    StartScale = scale;
  } else {
    GlobalScale = scale;  // second choice: we keep the maximum scale.
    StartScale = 1.0f;
  }
  // printf("normalize points :: global scale : %f\n", GlobalScale);

  for (int i = 0; i < 2; ++i) {
    int npti = pointcloud_[i].size();
    for (int ii = 0; ii < npti; ++ii) {
      pointcloud_[i][ii](0) /= GlobalScale;
      pointcloud_[i][ii](1) /= GlobalScale;
      pointcloud_[i][ii](2) /= GlobalScale;
    }
  }
}

// Normalize scale of points.
// X' = (X-\mu)/scale
// void FGROdometry::NormalizePoints() {
//   int num = 2;
//   float scale = 0;

//   Means.clear();

//   for (int i = 0; i < num; ++i) {
//     float max_scale = 0;

//     // compute mean
//     Vector3f mean;
//     mean.setZero();

//     int npti = pointcloud_[i].size();
//     for (int ii = 0; ii < npti; ++ii) {
//       Eigen::Vector3f p(pointcloud_[i][ii](0), pointcloud_[i][ii](1),
//                         pointcloud_[i][ii](2));
//       mean = mean + p;
//     }
//     mean = mean / npti;
//     Means.push_back(mean);

//     //printf("normalize points :: mean[%d] = [%f %f %f]\n", i, mean(0),
//     mean(1),
//     //       mean(2));

//     for (int ii = 0; ii < npti; ++ii) {
//       pointcloud_[i][ii](0) -= mean(0);
//       pointcloud_[i][ii](1) -= mean(1);
//       pointcloud_[i][ii](2) -= mean(2);
//     }

//     // compute scale
//     for (int ii = 0; ii < npti; ++ii) {
//       Eigen::Vector3f p(pointcloud_[i][ii](0), pointcloud_[i][ii](1),
//                         pointcloud_[i][ii](2));
//       float temp = p.norm();  // because we extract mean in the previous
//       stage.
//       if (temp > max_scale) max_scale = temp;
//     }

//     if (max_scale > scale) scale = max_scale;
//   }

//   //// mean of the scale variation
//   if (use_absolute_scale_) {
//     GlobalScale = 1.0f;
//     StartScale = scale;
//   } else {
//     GlobalScale = scale;  // second choice: we keep the maximum scale.
//     StartScale = 1.0f;
//   }
//   //printf("normalize points :: global scale : %f\n", GlobalScale);

//   for (int i = 0; i < num; ++i) {
//     int npti = pointcloud_[i].size();
//     for (int ii = 0; ii < npti; ++ii) {
//       pointcloud_[i][ii](0) /= GlobalScale;
//       pointcloud_[i][ii](1) /= GlobalScale;
//       pointcloud_[i][ii](2) /= GlobalScale;
//     }
//   }
// }

double FGROdometry::OptimizePairwise(bool decrease_mu_) {
  printf("Pairwise rigid pose optimization\n");

  double par;
  int numIter = iteration_number_;
  TransOutput_ = Eigen::Matrix4f::Identity();

  par = StartScale;

  int i = 0;
  int j = 1;

  // make another copy of pointcloud_[j].
  Points pcj_copy;
  int npcj = pointcloud_[j].size();
  pcj_copy.resize(npcj);
  for (int cnt = 0; cnt < npcj; cnt++) pcj_copy[cnt] = pointcloud_[j][cnt];

  if (corres_.size() < 10) return -1;

  std::vector<double> s(corres_.size(), 1.0);

  Eigen::Matrix4f trans;
  trans.setIdentity();

  for (int itr = 0; itr < numIter; itr++) {
    // graduated non-convexity.
    // std::cout << "iter: " << itr << std::endl;
    if (decrease_mu_) {
      if (itr % 4 == 0 && par > max_corr_dist_) {
        // std::cout << "\t" << "par: " << par << std::endl;
        par /= div_factor_;
      }
    }

    const int nvariable = 6;  // 3 for rotation and 3 for translation
    Eigen::MatrixXd JTJ(nvariable, nvariable);
    Eigen::MatrixXd JTr(nvariable, 1);
    Eigen::MatrixXd J(nvariable, 1);
    JTJ.setZero();
    JTr.setZero();

    double r;
    double r2 = 0.0;

    for (int c = 0; c < corres_.size(); c++) {
      int ii = corres_[c].first;
      int jj = corres_[c].second;
      Eigen::Vector3f p, q;
      p = pointcloud_[i][ii];
      q = pcj_copy[jj];
      Eigen::Vector3f rpq = p - q;

      int c2 = c;

      float temp = par / (rpq.dot(rpq) + par);
      s[c2] = temp * temp;

      J.setZero();
      J(1) = -q(2);
      J(2) = q(1);
      J(3) = -1;
      r = rpq(0);
      JTJ += J * J.transpose() * s[c2];
      JTr += J * r * s[c2];
      r2 += r * r * s[c2];

      J.setZero();
      J(2) = -q(0);
      J(0) = q(2);
      J(4) = -1;
      r = rpq(1);
      JTJ += J * J.transpose() * s[c2];
      JTr += J * r * s[c2];
      r2 += r * r * s[c2];

      J.setZero();
      J(0) = -q(1);
      J(1) = q(0);
      J(5) = -1;
      r = rpq(2);
      JTJ += J * J.transpose() * s[c2];
      JTr += J * r * s[c2];
      r2 += r * r * s[c2];

      r2 += (par * (1.0 - sqrt(s[c2])) * (1.0 - sqrt(s[c2])));
    }

    Eigen::MatrixXd result(nvariable, 1);
    result = -JTJ.llt().solve(JTr);

    Eigen::Affine3d aff_mat;
    aff_mat.linear() = (Eigen::Matrix3d)Eigen::AngleAxisd(
                           result(2), Eigen::Vector3d::UnitZ()) *
                       Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
    aff_mat.translation() = Eigen::Vector3d(result(3), result(4), result(5));

    Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();

    trans = delta * trans;
    TransformPoints(pcj_copy, delta);
    // std::cout << "\t" << "residual: " << r2 << std::endl;
  }

  TransOutput_ = trans * TransOutput_;
  return par;
}

double FGROdometry::OptimizePairwiseGPU(bool decrease_mu_,
                                        Eigen::Matrix4f initialisation) {
  // printf("Pairwise rigid pose optimization\n");

  double par;
  int numIter = iteration_number_;
  TransOutput_ = Eigen::Matrix4f::Identity();

  par = StartScale;
  std::cout << "start scale: " << StartScale << std::endl;
  int i = 0;
  int j = 1;

  // copy pcj and pci to gpu
  DeviceArray<float> pci_gpu;
  DeviceArray<float> pcj_gpu;
  pci_gpu.create(pointcloud_[i].size() * 3);
  pcj_gpu.create(pointcloud_[j].size() * 3);
  pci_gpu.upload((float*)pointcloud_[i].data(), pointcloud_[i].size() * 3);
  pcj_gpu.upload((float*)pointcloud_[j].data(), pointcloud_[j].size() * 3);

  if (corres_.size() < 10) return -1;

  std::vector<double> s(corres_.size(), 1.0);
  std::vector<FGRDataTerm> fgr_corres;
  for (const auto& c : corres_) {
    FGRDataTerm fgr_data_term;
    fgr_data_term.zero = c.first;
    fgr_data_term.one = c.second;
    fgr_data_term.lp = 1.0;
    fgr_data_term.valid = true;
    fgr_corres.push_back(fgr_data_term);
  }
  DeviceArray<FGRDataTerm> corres_gpu;
  corres_gpu.create(corres_.size());
  corres_gpu.upload(fgr_corres.data(), corres_.size());

  Eigen::Matrix4f trans = initialisation;
  // trans.setIdentity();

  DeviceArray<JtJJtrSE3> sumDataSE3;
  DeviceArray<JtJJtrSE3> outDataSE3;
  sumDataSE3.create(MAX_THREADS);
  outDataSE3.create(1);

  for (int itr = 0; itr < numIter; itr++) {
    // graduated non-convexity.
    // std::cout << "iter: " << itr << std::endl;
    if (decrease_mu_) {
      if (itr % 4 == 0 && par > max_corr_dist_) {
        // std::cout << "\t" << "par: " << par << std::endl;
        par /= div_factor_;
      }
    }
    float residual[2];

    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_fgr;
    Eigen::Matrix<float, 6, 1> b_fgr;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr =
        trans.topLeftCorner(3, 3);
    Eigen::Vector3f tcurr = Eigen::Vector3f(trans.topRightCorner(3, 1));
    mat33 device_Rcurr = Rcurr;
    float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

    fgrStep(device_Rcurr, device_tcurr, pci_gpu, pcj_gpu, corres_gpu, par,
            sumDataSE3, outDataSE3, A_fgr.data(), b_fgr.data(), residual, 256,
            112);

    last_residual = sqrt(residual[0]) / residual[1];
    Eigen::Matrix<double, 6, 1> result;
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_fgr = A_fgr.cast<double>();
    Eigen::Matrix<double, 6, 1> db_fgr = b_fgr.cast<double>();

    result = -dA_fgr.llt().solve(db_fgr);

    Eigen::Affine3d aff_mat;
    aff_mat.linear() = (Eigen::Matrix3d)Eigen::AngleAxisd(
                           result(2), Eigen::Vector3d::UnitZ()) *
                       Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
    aff_mat.translation() = Eigen::Vector3d(result(3), result(4), result(5));

    Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();

    trans = delta * trans;

    // std::cout << "\t" << "residual: " << sqrt(residual[0]) / residual[1] <<
    // std::endl;
  }

  corres_gpu.download(line_processes_);

  pci_gpu.release();
  pcj_gpu.release();
  corres_gpu.release();
  sumDataSE3.release();
  outDataSE3.release();

  TransOutput_ = trans * TransOutput_;
  std::cout << "end mu: " << par << std::endl;
  return par;
}

Eigen::Matrix4f SE3(Eigen::Quaternionf r, Eigen::Vector3f t) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

  Eigen::Matrix3f rot = r.normalized().toRotationMatrix();

  T.topLeftCorner(3, 3) = rot;
  T.topRightCorner(3, 1) = t;

  return T;
}

double FGROdometry::OptimizePairwiseGPUPDA(
    std::vector<DeviceArray2D<float>>& vmap_curr_pyr,
    std::vector<DeviceArray2D<float>>& nmap_curr_pyr,
    std::vector<DeviceArray2D<unsigned char>>& rgb_curr_pyr,
    Eigen::Matrix4f& pose_curr,
    std::vector<DeviceArray2D<float>>& vmap_prev_pyr,
    std::vector<DeviceArray2D<float>>& nmap_prev_pyr,
    std::vector<DeviceArray2D<unsigned char>>& rgb_prev_pyr,
    Eigen::Matrix4f& pose_prev, Eigen::Matrix4f& last_pose,
    Eigen::Matrix4f& pre_last_pose, CameraModel& intr, bool decrease_mu_,
    bool rgb, bool geom, Eigen::Matrix4f initialisation) {
  // printf("Pairwise rigid pose optimization\n");

  double par;
  int numIter = iteration_number_;
  TransOutput_ = Eigen::Matrix4f::Identity();
  const float distance_thresh = 0.1;  // 0.6;//0.1;//0.10f;
  const float angle_thresh = sin(20.f * 3.14159254f / 180.f);  // sin(0.6);
  const int blocks = 112;
  const int threads = 256;
  int compute_corres_freq = 6;
  int pyr_levels = vmap_curr_pyr.size() - 2;

  // Do everything in kf local
  Eigen::Matrix4f last_2_pre_last_diff = pre_last_pose.inverse() * last_pose;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
      initialisations;

  Eigen::Matrix4f ident = Eigen::Matrix4f::Identity();

  // //pose curr is in keyframes frame of reference not the global maps frame of
  // reference.
  initialisations.push_back(ident);
  initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr);
  initialisations.push_back(last_2_pre_last_diff.inverse() *
                            last_2_pre_last_diff.inverse() * pose_curr);
  initialisations.push_back(pose_curr);

  for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta += 0.02) {
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, rotDelta, 0, 0), Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, 0, rotDelta, 0), Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, 0, 0, rotDelta), Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, 0, 0), Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, 0, -rotDelta, 0), Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, 0, 0, -rotDelta), Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, rotDelta, rotDelta, 0),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, 0, rotDelta, rotDelta),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, rotDelta, 0, rotDelta),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, -rotDelta, rotDelta, 0),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, 0, -rotDelta, rotDelta),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, -rotDelta, 0, rotDelta),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, rotDelta, -rotDelta, 0),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, 0, rotDelta, -rotDelta),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(last_2_pre_last_diff.inverse() * pose_curr *
                              SE3(Eigen::Quaternionf(1, rotDelta, 0, -rotDelta),
                                  Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, -rotDelta, 0),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, 0, -rotDelta, -rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, 0, -rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, -rotDelta, -rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, -rotDelta, rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, rotDelta, -rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, rotDelta, rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, rotDelta, -rotDelta, -rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, rotDelta, -rotDelta, rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, -rotDelta, -rotDelta, -rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, rotDelta, rotDelta, -rotDelta),
            Eigen::Vector3f(0, 0, 0)));
    initialisations.push_back(
        last_2_pre_last_diff.inverse() * pose_curr *
        SE3(Eigen::Quaternionf(1, rotDelta, rotDelta, rotDelta),
            Eigen::Vector3f(0, 0, 0)));
  }
  float best_residual = std::numeric_limits<float>::infinity();

  if (geom) {
    for (const auto& init : initialisations) {
      Eigen::Matrix4f trans = init;

      par = 10.0f;  // 5.0f;//10.0f;//5.0f;//StartScale;
      float mu = 1.8;
      float max_corres_d = 0.0001;         // 0.000001f;
      residual_threshold_ = max_corres_d;  // 5.0e-06;
      int num_iter = 64;
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv =
          pose_prev.topLeftCorner(3, 3).inverse();
      Eigen::Vector3f tprev = pose_prev.topRightCorner(3, 1);
      mat33 device_Rprev_inv = Rprev_inv;
      float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Corres_Rcurr =
          init.topLeftCorner(3, 3);  // pose_curr.topLeftCorner(3,3);
      Eigen::Vector3f corres_tcurr =
          init.topRightCorner(3, 1);  // pose_curr.topRightCorner(3,1);
      mat33 device_Rcurr = Corres_Rcurr;
      float3 device_tcurr = *reinterpret_cast<float3*>(corres_tcurr.data());

      DeviceArray2D<FGRPDADataTerm> corres_gpu;
      int pyr_level = vmap_curr_pyr.size() - 1;
      // computeCorrespondences(device_Rcurr, device_tcurr, vmap_curr_pyr.back(),
      //                        nmap_curr_pyr.back(), device_Rprev_inv,
      //                        device_tprev, vmap_prev_pyr.back(),
      //                        nmap_prev_pyr.back(), corres_gpu, intr(pyr_level),
      //                        distance_thresh, angle_thresh, blocks, threads);

      DeviceArray<JtJJtrSE3> sumDataSE3;
      DeviceArray<JtJJtrSE3> outDataSE3;
      sumDataSE3.create(MAX_THREADS);
      outDataSE3.create(1);

      for (int itr = 0; itr < num_iter; itr++) {
        // graduated non-convexity.
        // std::cout << "iter: " << itr << std::endl;
        if (decrease_mu_) {
          if (itr % 4 == 0 && par > max_corres_d) {
            // std::cout << "\t" << "par: " << par << std::endl;
            par /= mu;
          }
        }

        if (itr % compute_corres_freq == 0) {
          Corres_Rcurr =
              trans.topLeftCorner(3, 3);  // pose_curr.topLeftCorner(3,3);
          corres_tcurr =
              trans.topRightCorner(3, 1);  // pose_curr.topRightCorner(3,1);
          mat33 device_Rcurr = Corres_Rcurr;
          float3 device_tcurr = *reinterpret_cast<float3*>(corres_tcurr.data());
          int pyr_level = vmap_curr_pyr.size() - 1;
          // computeCorrespondences(
          //     device_Rcurr, device_tcurr, vmap_curr_pyr.back(),
          //     nmap_curr_pyr.back(), device_Rprev_inv, device_tprev,
          //     vmap_prev_pyr.back(), nmap_prev_pyr.back(), corres_gpu,
          //     intr(pyr_level), distance_thresh, angle_thresh, blocks, threads);
        }

        float residual[2];

        Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_fgr;
        Eigen::Matrix<float, 6, 1> b_fgr;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr =
            trans.topLeftCorner(3, 3);
        Eigen::Vector3f tcurr = Eigen::Vector3f(trans.topRightCorner(3, 1));
        mat33 device_Rcurr = Rcurr;
        float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

        // fgrPDAStep(device_Rcurr, device_tcurr, vmap_prev_pyr.back(),
        //            vmap_curr_pyr.back(), intr, corres_gpu, par, sumDataSE3,
        //            outDataSE3, A_fgr.data(), b_fgr.data(), residual, 256, 112);

        last_residual = sqrt(residual[0]) / residual[1];
        Eigen::Matrix<double, 6, 1> result;
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_fgr =
            A_fgr.cast<double>();
        Eigen::Matrix<double, 6, 1> db_fgr = b_fgr.cast<double>();
        result = -dA_fgr.llt().solve(db_fgr);

        Eigen::Affine3d aff_mat;
        aff_mat.linear() =
            (Eigen::Matrix3d)Eigen::AngleAxisd(result(2),
                                               Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
        aff_mat.translation() =
            Eigen::Vector3d(result(3), result(4), result(5));

        Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();

        trans = delta * trans;
      }

      if (last_residual < best_residual) {
        TransOutput_ = Eigen::Matrix4f::Identity();
        TransOutput_ = trans * TransOutput_;
        best_residual = last_residual;
        pda_line_processes_.clear();
        std::vector<FGRPDADataTerm> corres;
        int elem_step;
        corres_gpu.download(corres, elem_step);
        std::cout << "num corres: " << corres.size() << std::endl;
        int valid_count = 0;
        for (const auto& c : corres) {
          if (c.valid) pda_line_processes_.push_back(c);
        }
        std::cout << "num valid corres: " << pda_line_processes_.size()
                  << std::endl;
      }
      corres_gpu.release();
      sumDataSE3.release();
      outDataSE3.release();
      if (best_residual < residual_threshold_) {
        // return par;
        break;
      }
      // std::cout << "end mu: " << par << std::endl;
    }

    // return par;

    Eigen::Matrix4f trans = TransOutput_;
    par = 10.0f;  // 5.0f;//10.0f;//5.0f;//StartScale;
    float mu = 1.8;
    float max_corres_d = 0.0001f;
    int num_iter = 64;
    for (int i = pyr_levels; i >= 0; i--) {
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv =
          pose_prev.topLeftCorner(3, 3).inverse();
      Eigen::Vector3f tprev = pose_prev.topRightCorner(3, 1);
      mat33 device_Rprev_inv = Rprev_inv;
      float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Corres_Rcurr =
          trans.topLeftCorner(3, 3);  // pose_curr.topLeftCorner(3,3);
      Eigen::Vector3f corres_tcurr =
          trans.topRightCorner(3, 1);  // pose_curr.topRightCorner(3,1);
      mat33 device_Rcurr = Corres_Rcurr;
      float3 device_tcurr = *reinterpret_cast<float3*>(corres_tcurr.data());

      DeviceArray2D<FGRPDADataTerm> corres_gpu;
      // computeCorrespondences(
      //     device_Rcurr, device_tcurr, vmap_curr_pyr[i], nmap_curr_pyr[i],
      //     device_Rprev_inv, device_tprev, vmap_prev_pyr[i], nmap_prev_pyr[i],
      //     corres_gpu, intr(i), distance_thresh, angle_thresh, blocks, threads);

      DeviceArray<JtJJtrSE3> sumDataSE3;
      DeviceArray<JtJJtrSE3> outDataSE3;
      sumDataSE3.create(MAX_THREADS);
      outDataSE3.create(1);

      for (int itr = 0; itr < num_iter; itr++) {
        // graduated non-convexity.
        // std::cout << "iter: " << itr << std::endl;
        if (decrease_mu_) {
          if (itr % 4 == 0 && par > max_corr_dist_) {
            // std::cout << "\t" << "par: " << par << std::endl;
            par /= div_factor_;
          }
        }

        if (itr % compute_corres_freq == 0) {
          Corres_Rcurr =
              trans.topLeftCorner(3, 3);  // pose_curr.topLeftCorner(3,3);
          corres_tcurr =
              trans.topRightCorner(3, 1);  // pose_curr.topRightCorner(3,1);
          mat33 device_Rcurr = Corres_Rcurr;
          float3 device_tcurr = *reinterpret_cast<float3*>(corres_tcurr.data());
          int pyr_level = vmap_curr_pyr.size() - 1;
          // computeCorrespondences(
          //     device_Rcurr, device_tcurr, vmap_curr_pyr[i], nmap_curr_pyr[i],
          //     device_Rprev_inv, device_tprev, vmap_prev_pyr[i],
          //     nmap_prev_pyr[i], corres_gpu, intr(i), distance_thresh,
          //     angle_thresh, blocks, threads);
        }

        float residual[2];

        Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_fgr;
        Eigen::Matrix<float, 6, 1> b_fgr;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr =
            trans.topLeftCorner(3, 3);
        Eigen::Vector3f tcurr = Eigen::Vector3f(trans.topRightCorner(3, 1));
        mat33 device_Rcurr = Rcurr;
        float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

        // fgrPDAStep(device_Rcurr, device_tcurr, vmap_prev_pyr[i],
        //            vmap_curr_pyr[i], intr, corres_gpu, par, sumDataSE3,
        //            outDataSE3, A_fgr.data(), b_fgr.data(), residual, 256, 112);

        last_residual = sqrt(residual[0]) / residual[1];
        Eigen::Matrix<double, 6, 1> result;
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_fgr =
            A_fgr.cast<double>();
        Eigen::Matrix<double, 6, 1> db_fgr = b_fgr.cast<double>();
        result = -dA_fgr.llt().solve(db_fgr);

        Eigen::Affine3d aff_mat;
        aff_mat.linear() =
            (Eigen::Matrix3d)Eigen::AngleAxisd(result(2),
                                               Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
        aff_mat.translation() =
            Eigen::Vector3d(result(3), result(4), result(5));

        Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();

        trans = delta * trans;
      }

      best_residual = last_residual;
      pda_line_processes_.clear();
      std::vector<FGRPDADataTerm> corres;
      int elem_step;
      corres_gpu.download(corres, elem_step);
      int valid_count = 0;
      for (const auto& c : corres) {
        if (c.valid) pda_line_processes_.push_back(c);
      }
      corres_gpu.release();
      sumDataSE3.release();
      outDataSE3.release();
    }
  }

  Eigen::Matrix4f trans = TransOutput_;
  TransOutput_ = Eigen::Matrix4f::Identity();
  TransOutput_ = trans * TransOutput_;

  if (rgb) {
    std::vector<DeviceArray2D<short>> nextdIdx;
    std::vector<DeviceArray2D<short>> nextdIdy;
    std::vector<DeviceArray2D<float3>> pointClouds;
    std::vector<DeviceArray2D<float>> dmap_prev_pyr;
    std::vector<DeviceArray2D<float>> dmap_curr_pyr;
    std::vector<DeviceArray2D<DataTerm>> corres_img;

    DeviceArray<int2> sumResidualRGB(MAX_THREADS);
    DeviceArray<JtJJtrSE3> sumDataSE3(MAX_THREADS);
    DeviceArray<JtJJtrSE3> outDataSE3(1);

    int sobelSize = 3;
    float sobelScale = 1.0 / pow(2.0, sobelSize);
    float maxDepthDeltaRGB = 0.07f;

    std::vector<float> minimumGradientMagnitudes;
    minimumGradientMagnitudes.resize(pyr_levels);
    minimumGradientMagnitudes[0] = 5;
    minimumGradientMagnitudes[1] = 3;
    minimumGradientMagnitudes[2] = 1;
    minimumGradientMagnitudes[3] = 1;

    for (int i = 0; i < pyr_levels; i++) {
      DeviceArray2D<DataTerm> c_img(rgb_curr_pyr[i].rows(),
                                    rgb_curr_pyr[i].cols());
      corres_img.push_back(c_img);
    }
    for (int i = 0; i < pyr_levels; i++) {
      DeviceArray2D<short> idx(rgb_curr_pyr[i].rows(), rgb_curr_pyr[i].cols());
      DeviceArray2D<short> idy(rgb_curr_pyr[i].rows(), rgb_curr_pyr[i].cols());
      nextdIdx.push_back(idx);
      nextdIdy.push_back(idy);
      computeDerivativeImages(rgb_curr_pyr[i], nextdIdx[i], nextdIdy[i]);
    }

    DeviceArray2D<float> dmap_p(vmap_prev_pyr[0].rows() / 3,
                                vmap_prev_pyr[1].cols());
    verticesToDepth(vmap_prev_pyr[0], dmap_p, 6.0f);  // 6.0 in ef
    dmap_prev_pyr.push_back(dmap_p);

    DeviceArray2D<float> dmap_c(vmap_curr_pyr[0].rows() / 3,
                                vmap_curr_pyr[1].cols());
    verticesToDepth(vmap_curr_pyr[0], dmap_c, 6.0f);  // 6.0 in ef
    dmap_curr_pyr.push_back(dmap_c);

    for (int i = 0; i < pyr_levels; i++) {
      DeviceArray2D<float> d_map_p(vmap_prev_pyr[i + 1].rows() / 3,
                                   vmap_prev_pyr[i + 1].cols());
      pyrDownGaussF(dmap_prev_pyr[i], d_map_p);
      dmap_prev_pyr.push_back(d_map_p);

      DeviceArray2D<float> d_map_c(vmap_curr_pyr[i + 1].rows() / 3,
                                   vmap_curr_pyr[i + 1].cols());
      pyrDownGaussF(dmap_curr_pyr[i], d_map_c);
      dmap_curr_pyr.push_back(d_map_c);
    }

    for (int i = 0; i < pyr_levels; i++) {
      DeviceArray2D<float3> p_cloud(dmap_prev_pyr[i].rows(),
                                    dmap_prev_pyr[i].cols());
      projectToPointCloud(dmap_prev_pyr[i], p_cloud, intr, i);
      pointClouds.push_back(p_cloud);
    }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt =
        trans.cast<double>();

    for (int i = pyr_levels - 1; i >= 0; i--) {
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K =
          Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

      K(0, 0) = intr(i).fx;
      K(1, 1) = intr(i).fy;
      K(0, 2) = intr(i).cx;
      K(1, 2) = intr(i).cy;
      K(2, 2) = 1;

      float lastRGBError = std::numeric_limits<float>::max();
      float lastRGBCount = rgb_curr_pyr[i].cols() * rgb_curr_pyr[i].rows();

      for (int jIter = 0; jIter < 10; jIter++) {
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv =
            K * R * K.inverse();
        mat33 krkInv;
        memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(),
               sizeof(mat33));

        Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
        Kt = K * Kt;
        float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

        int sigma = 0;
        int rgbSize = 0;

        computeRgbResidual(
            pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
            nextdIdx[i], nextdIdy[i], dmap_prev_pyr[i], dmap_curr_pyr[i],
            rgb_prev_pyr[i], rgb_curr_pyr[i], corres_img[i], sumResidualRGB,
            maxDepthDeltaRGB, kt, krkInv, sigma, rgbSize, 256, 112);
        if (sqrt(sigma) / rgbSize > lastRGBError) {
          break;
        }
        lastRGBError = sqrt(sigma) / rgbSize;
        lastRGBCount = rgbSize;

        Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
        Eigen::Matrix<float, 6, 1> b_rgbd;

        rgbStep(corres_img[i], -1, pointClouds[i], intr(i).fx, intr(i).fy,
                nextdIdx[i], nextdIdy[i], sobelScale, sumDataSE3, outDataSE3,
                A_rgbd.data(), b_rgbd.data(), 256, 112);

        Eigen::Matrix<double, 6, 1> result;
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd =
            A_rgbd.cast<double>();
        Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();

        result = dA_rgbd.ldlt().solve(db_rgbd);

        Eigen::Affine3d aff_mat;
        aff_mat.linear() =
            (Eigen::Matrix3d)Eigen::AngleAxisd(result(2),
                                               Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
        aff_mat.translation() =
            Eigen::Vector3d(result(3), result(4), result(5));

        Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();

        trans = delta * trans;
        resultRt = trans.cast<double>();
      }
    }

    for (int i = 0; i < pyr_levels; i++) {
      nextdIdx[i].release();
      nextdIdy[i].release();
      pointClouds[i].release();
      dmap_prev_pyr[i].release();
      dmap_curr_pyr[i].release();
      corres_img[i].release();
    }
  }

  TransOutput_ = Eigen::Matrix4f::Identity();
  TransOutput_ = trans * TransOutput_;

  return par;
}

Eigen::Matrix4f FGROdometry::OptimizePhotometricCost(
    std::vector<DeviceArray2D<float>>& vmap_curr_pyr,
    std::vector<DeviceArray2D<float>>& nmap_curr_pyr,
    std::vector<DeviceArray2D<unsigned char>>& rgb_curr_pyr,
    std::vector<DeviceArray2D<float>>& vmap_prev_pyr,
    std::vector<DeviceArray2D<float>>& nmap_prev_pyr,
    std::vector<DeviceArray2D<unsigned char>>& rgb_prev_pyr,
    CameraModel& intr, Eigen::Matrix4f initialisation) {
  std::vector<DeviceArray2D<short>> nextdIdx;
  std::vector<DeviceArray2D<short>> nextdIdy;
  std::vector<DeviceArray2D<float3>> pointClouds;
  std::vector<DeviceArray2D<float>> dmap_prev_pyr;
  std::vector<DeviceArray2D<float>> dmap_curr_pyr;
  std::vector<DeviceArray2D<DataTerm>> corres_img;

  DeviceArray<int2> sumResidualRGB(MAX_THREADS);
  DeviceArray<JtJJtrSE3> sumDataSE3(MAX_THREADS);
  DeviceArray<JtJJtrSE3> outDataSE3(1);

  //TransOutput_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f trans = initialisation;

  int pyr_levels = vmap_curr_pyr.size() - 1;
  int sobelSize = 3;
  float sobelScale = 1.0 / pow(2.0, sobelSize);
  float maxDepthDeltaRGB = 0.07f;

  std::vector<float> minimumGradientMagnitudes;
  minimumGradientMagnitudes.resize(pyr_levels);
  minimumGradientMagnitudes[0] = 5;
  minimumGradientMagnitudes[1] = 3;
  minimumGradientMagnitudes[2] = 1;
  minimumGradientMagnitudes[3] = 1;

  for (int i = 0; i < pyr_levels; i++) {
    DeviceArray2D<DataTerm> c_img(rgb_curr_pyr[i].rows(),
                                  rgb_curr_pyr[i].cols());
    corres_img.push_back(c_img);
  }
  for (int i = 0; i < pyr_levels; i++) {
    DeviceArray2D<short> idx(rgb_curr_pyr[i].rows(), rgb_curr_pyr[i].cols());
    DeviceArray2D<short> idy(rgb_curr_pyr[i].rows(), rgb_curr_pyr[i].cols());
    nextdIdx.push_back(idx);
    nextdIdy.push_back(idy);
    computeDerivativeImages(rgb_curr_pyr[i], nextdIdx[i], nextdIdy[i]);
  }

  DeviceArray2D<float> dmap_p(vmap_prev_pyr[0].rows() / 3,
                              vmap_prev_pyr[1].cols());
  verticesToDepth(vmap_prev_pyr[0], dmap_p, 6.0f);  // 6.0 in ef
  dmap_prev_pyr.push_back(dmap_p);

  DeviceArray2D<float> dmap_c(vmap_curr_pyr[0].rows() / 3,
                              vmap_curr_pyr[1].cols());
  verticesToDepth(vmap_curr_pyr[0], dmap_c, 6.0f);  // 6.0 in ef
  dmap_curr_pyr.push_back(dmap_c);

  for (int i = 0; i < pyr_levels; i++) {
    DeviceArray2D<float> d_map_p(vmap_prev_pyr[i + 1].rows() / 3,
                                 vmap_prev_pyr[i + 1].cols());
    pyrDownGaussF(dmap_prev_pyr[i], d_map_p);
    dmap_prev_pyr.push_back(d_map_p);

    DeviceArray2D<float> d_map_c(vmap_curr_pyr[i + 1].rows() / 3,
                                 vmap_curr_pyr[i + 1].cols());
    pyrDownGaussF(dmap_curr_pyr[i], d_map_c);
    dmap_curr_pyr.push_back(d_map_c);
  }

  for (int i = 0; i < pyr_levels; i++) {
    DeviceArray2D<float3> p_cloud(dmap_prev_pyr[i].rows(),
                                  dmap_prev_pyr[i].cols());
    projectToPointCloud(dmap_prev_pyr[i], p_cloud, intr, i);
    pointClouds.push_back(p_cloud);
  }

  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = trans.cast<double>();

  for (int i = pyr_levels - 1; i >= 0; i--) {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K =
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

    K(0, 0) = intr(i).fx;
    K(1, 1) = intr(i).fy;
    K(0, 2) = intr(i).cx;
    K(1, 2) = intr(i).cy;
    K(2, 2) = 1;

    float lastRGBError = std::numeric_limits<float>::max();
    float lastRGBCount = rgb_curr_pyr[i].cols() * rgb_curr_pyr[i].rows();

    for (int jIter = 0; jIter < 100; jIter++) {
      Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv =
          K * R * K.inverse();
      mat33 krkInv;
      memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(),
             sizeof(mat33));

      Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
      Kt = K * Kt;
      float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

      int sigma = 0;
      int rgbSize = 0;

      computeRgbResidual(
          pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
          nextdIdx[i], nextdIdy[i], dmap_prev_pyr[i], dmap_curr_pyr[i],
          rgb_prev_pyr[i], rgb_curr_pyr[i], corres_img[i], sumResidualRGB,
          maxDepthDeltaRGB, kt, krkInv, sigma, rgbSize, 256, 112);
      if (sqrt(sigma) / rgbSize > lastRGBError) {
        std::cout << "error increasing: " << rgbSize << std::endl;
        break;
      }
      std::cout << "num corres: " << rgbSize << std::endl;
      lastRGBError = sqrt(sigma) / rgbSize;
      lastRGBCount = rgbSize;

      Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
      Eigen::Matrix<float, 6, 1> b_rgbd;

      rgbStep(corres_img[i], -1, pointClouds[i], intr(i).fx, intr(i).fy,
              nextdIdx[i], nextdIdy[i], sobelScale, sumDataSE3, outDataSE3,
              A_rgbd.data(), b_rgbd.data(), 256, 112);

      Eigen::Matrix<double, 6, 1> result;
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd =
          A_rgbd.cast<double>();
      Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();

      // result = dA_rgbd.ldlt().solve(db_rgbd);

      // Eigen::Affine3d aff_mat;
      // aff_mat.linear() =
      //     (Eigen::Matrix3d)Eigen::AngleAxisd(result(2),
      //                                        Eigen::Vector3d::UnitZ()) *
      //     Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY()) *
      //     Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
      // aff_mat.translation() = Eigen::Vector3d(result(3), result(4), result(5));

      // Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();
      // trans = delta * trans;
      // trans = delta * trans;
      //resultRt = trans.cast<double>();
      
      result = dA_rgbd.ldlt().solve(db_rgbd);

     Eigen::Isometry3f rgbOdom;

     //odometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);

     Eigen::Isometry3f currentT;
     currentT.setIdentity();
    //  currentT.rotate(Rprev);
    //  currentT.translation() = tprev;

     currentT = currentT * rgbOdom.inverse();

     trans.topRightCorner(3,1) = currentT.translation();
     trans.topLeftCorner(3,3) = currentT.rotation();
  
    }
  }

  for (int i = 0; i < pyr_levels; i++) {
    nextdIdx[i].release();
    nextdIdy[i].release();
    pointClouds[i].release();
    dmap_prev_pyr[i].release();
    dmap_curr_pyr[i].release();
    corres_img[i].release();
  }
  // TransOutput_ = Eigen::Matrix4f::Identity();
  // TransOutput_ = trans * TransOutput_;

  return trans;
}

void FGROdometry::TransformPoints(Points& points,
                                  const Eigen::Matrix4f& Trans) {
  int npc = (int)points.size();
  Matrix3f R = Trans.block<3, 3>(0, 0);
  Vector3f t = Trans.block<3, 1>(0, 3);
  Vector3f temp;
  for (int cnt = 0; cnt < npc; cnt++) {
    temp = R * points[cnt] + t;
    points[cnt] = temp;
  }
}

Eigen::Matrix4f FGROdometry::GetOutputTrans() {
  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  R = TransOutput_.block<3, 3>(0, 0);
  t = TransOutput_.block<3, 1>(0, 3);

  Eigen::Matrix4f transtemp;
  transtemp.fill(0.0f);

  transtemp.block<3, 3>(0, 0) = R;
  transtemp.block<3, 1>(0, 3) = -R * Means[1] + t * GlobalScale + Means[0];
  transtemp(3, 3) = 1;

  return transtemp;
}

void FGROdometry::WriteTrans(const char* filepath) {
  FILE* fid = fopen(filepath, "w");

  // Below line indicates how the transformation matrix aligns two point clouds
  // e.g. T * pointcloud_[1] is aligned with pointcloud_[0].
  // '2' indicates that there are two point cloud fragments.
  fprintf(fid, "0 1 2\n");

  Eigen::Matrix4f transtemp = GetOutputTrans();

  fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(0, 0), transtemp(0, 1),
          transtemp(0, 2), transtemp(0, 3));
  fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(1, 0), transtemp(1, 1),
          transtemp(1, 2), transtemp(1, 3));
  fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(2, 0), transtemp(2, 1),
          transtemp(2, 2), transtemp(2, 3));
  fprintf(fid, "%.10f %.10f %.10f %.10f\n", 0.0f, 0.0f, 0.0f, 1.0f);

  fclose(fid);
}

Eigen::Matrix4f FGROdometry::ReadTrans(const char* filename) {
  Eigen::Matrix4f temp;
  temp.fill(0);
  int temp0, temp1, temp2, cnt = 0;
  FILE* fid = fopen(filename, "r");
  while (fscanf(fid, "%d %d %d", &temp0, &temp1, &temp2) == 3) {
    for (int j = 0; j < 4; j++) {
      float a, b, c, d;
      fscanf(fid, "%f %f %f %f", &a, &b, &c, &d);
      temp(j, 0) = a;
      temp(j, 1) = b;
      temp(j, 2) = c;
      temp(j, 3) = d;
    }
  }
  return temp;
}

void FGROdometry::BuildDenseCorrespondence(const Eigen::Matrix4f& trans,
                                           Correspondences& corres) {
  int fi = 0;
  int fj = 1;
  Points pci = pointcloud_[fi];
  Points pcj = pointcloud_[fj];
  TransformPoints(pcj, trans);

  KDTree feature_tree_i(flann::KDTreeSingleIndexParams(15));
  BuildKDTree(pci, &feature_tree_i);
  std::vector<int> ind;
  std::vector<float> dist;
  corres.clear();
  for (int j = 0; j < pcj.size(); ++j) {
    SearchKDTree(&feature_tree_i, pcj[j], ind, dist, 1);
    float dist_j = sqrt(dist[0]);
    if (dist_j / GlobalScale < max_corr_dist_ / 2.0)
      corres.push_back(std::pair<int, int>(ind[0], j));
  }
}

void FGROdometry::Evaluation(const char* gth, const char* estimation,
                             const char* output) {
  float inlier_ratio = -1.0f;
  float overlapping_ratio = -1.0f;

  int fi = 0;
  int fj = 1;

  std::vector<std::pair<int, int>> corres;
  Eigen::Matrix4f gth_trans = ReadTrans(gth);
  BuildDenseCorrespondence(gth_trans, corres);
  printf("Groundtruth correspondences [%d-%d] : %d\n", fi, fj,
         (int)corres.size());

  int ncorres = corres.size();
  float err_mean = 0.0f;

  Points pci = pointcloud_[fi];
  Points pcj = pointcloud_[fj];
  Eigen::Matrix4f est_trans = ReadTrans(estimation);
  std::vector<float> error;
  for (int i = 0; i < ncorres; ++i) {
    int idi = corres[i].first;
    int idj = corres[i].second;
    Eigen::Vector4f pi(pci[idi](0), pci[idi](1), pci[idi](2), 1);
    Eigen::Vector4f pj(pcj[idj](0), pcj[idj](1), pcj[idj](2), 1);
    Eigen::Vector4f pjt = est_trans * pj;
    float errtemp = (pi - pjt).norm();
    error.push_back(errtemp);
    // this is based on the RMSE defined in
    // https://en.wikipedia.org/wiki/Root-mean-square_deviation
    errtemp = errtemp * errtemp;
    err_mean += errtemp;
  }
  err_mean /= ncorres;        // this is MSE = mean(d^2)
  err_mean = sqrt(err_mean);  // this is RMSE = sqrt(MSE)
  printf("mean error : %0.4e\n", err_mean);

  // overlapping_ratio = (float)ncorres / min(
  //		pointcloud_[fj].size(), pointcloud_[fj].size());
  overlapping_ratio = (float)ncorres / pointcloud_[fj].size();

  // write errors
  FILE* fid = fopen(output, "w");
  fprintf(fid, "%d %d %e %e %e\n", fi, fj, err_mean, inlier_ratio,
          overlapping_ratio);
  fclose(fid);
}

std::vector<Feature>& FGROdometry::Features() { return features_; }

std::vector<BriskFeature>& FGROdometry::BriskFeatures() {
  return brisk_features_;
}

std::vector<Points>& FGROdometry::PointClouds() { return pointcloud_; }

std::vector<Points>& FGROdometry::Normals() { return normals_; }

std::vector<Points>& FGROdometry::UnNormalisedPointClouds() {
  return unnormalised_pointcloud_;
}

std::vector<std::pair<int, int>>& FGROdometry::PointCorrespondences() {
  return corres_;
}

std::vector<FGRDataTerm>& FGROdometry::LineProcesses() {
  return line_processes_;
}

std::vector<KDTree>& FGROdometry::FeatureTrees() { return feature_trees_; }

const float& FGROdometry::LastResidual() { return last_residual; }

int FGROdometry::GetInliers(const float & angle_thresh, const float & dist_thresh)
{

}

int FGROdometry::GetInliers(const float inlier_lp_thresh)
{
  int num_inliers = 0;

  for(const auto & fgr_data_term : line_processes_)
  {
    if(fgr_data_term.lp > inlier_lp_thresh)
    {
      num_inliers++;
    }
  }
  return num_inliers;
}

std::vector<Points>& FGROdometry::MeanShiftedPointClouds() {
  return mean_shifted_pointclouds_;
}

Points& FGROdometry::PointCloudMeans() { return Means; }

std::vector<FGRPDADataTerm>& FGROdometry::PDALineProcesses() {
  return pda_line_processes_;
}

int FGROdometry::NumLpBelowThresh(float thresh) {
  int num_valid_corres = 0;

  for (const auto& t : pda_line_processes_) {
    if (t.valid && t.lp < thresh) {
      num_valid_corres++;
    }
  }

  return num_valid_corres;
}