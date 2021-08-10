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
#include <vector>
#include <flann/flann.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/impl/sift_keypoint.hpp>
#include <pcl/features/brisk_2d.h>
#include <pcl/keypoints/brisk_2d.h>
#include <pcl/gpu/features/features.hpp>

#include "Resolution.h"

#include "Stopwatch.h"

#include "../Cuda/cudafuncs.cuh"


#define USE_ABSOLUTE_SCALE	1		// Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
#define DIV_FACTOR			1.4	// Division factor used for graduated non-convexity
#define MAX_CORR_DIST		0.00001//0.001//0.0006//0.005//0.0000125	// Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
#define ITERATION_NUMBER	64//512		// Maximum number of iteration
#define TUPLE_SCALE			0.98//0.98	// Similarity measure used for tuples of feature points.
#define TUPLE_MAX_CNT		3000//1000	// Maximum tuple numbers.


typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> Points;
typedef std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> Feature;
typedef std::vector<Eigen::Matrix<unsigned char, 64, 1>, Eigen::aligned_allocator<Eigen::Matrix<unsigned char, 64, 1>>> BriskFeature;
typedef flann::Index<flann::L2<float> > KDTree;
typedef flann::Index<flann::Hamming<unsigned char>> LshIndex;
typedef std::vector<std::pair<int, int> > Correspondences;

class FGROdometry{
public:
	FGROdometry(double div_factor         = DIV_FACTOR,
	    bool    use_absolute_scale = USE_ABSOLUTE_SCALE,
	    double  max_corr_dist      = MAX_CORR_DIST,
	    int     iteration_number   = ITERATION_NUMBER,
	    float   tuple_scale        = TUPLE_SCALE,
	    int     tuple_max_cnt      = TUPLE_MAX_CNT):
		div_factor_(div_factor),
		use_absolute_scale_(use_absolute_scale),
		max_corr_dist_(max_corr_dist),
		iteration_number_(iteration_number),
		tuple_scale_(tuple_scale),
		tuple_max_cnt_(tuple_max_cnt){}
	void LoadFeature(const Points& pts, const Feature& feat, KDTree & feature_tree);
	void LoadFeature(const Points& pts, const Points& unnormalised_pts, const Feature& feat, Eigen::Vector3f & mean, KDTree & feature_tree);
	void LoadFeature(const Points& pts, const BriskFeature& feat);
	void ReadFeature(const char* filepath);
    void computeFeatures(std::vector<float> & points, std::vector<float> & normals);
	void computeFeaturesSIFT2D(std::vector<float> & points, const std::shared_ptr<unsigned char> & rgb, std::vector<float> & normals,  const int & height = 480, const int & width = 640);
	void computeFeaturesGPU(std::vector<float> & points, std::vector<float> & normals, int height = 640, int width = 480);
	void NormalizePoints();
	void ScalePoints();
	void AdvancedMatching();
	Eigen::Matrix4f ReadTrans(const char* filepath);
	void WriteTrans(const char* filepath);
	Eigen::Matrix4f GetOutputTrans();
	double OptimizePairwise(bool decrease_mu_);
	double OptimizePairwiseGPU(bool decrease_mu_, Eigen::Matrix4f initialisation = Eigen::Matrix4f::Identity());
	double OptimizePairwiseGPUPDA(std::vector<DeviceArray2D<float>> & vmap_curr_pyr, std::vector<DeviceArray2D<float>> & nmap_curr_pyr, std::vector<DeviceArray2D<unsigned char>> & rgb_curr_pyr, Eigen::Matrix4f & pose_curr, std::vector<DeviceArray2D<float>> & vmap_prev_pyr, std::vector<DeviceArray2D<float>> & nmap_prev_pyr, std::vector<DeviceArray2D<unsigned char>> & rgb_prev_pyr, Eigen::Matrix4f & pose_prev, Eigen::Matrix4f & last_pose, Eigen::Matrix4f & pre_last_pose, CameraModel & intr, bool decrease_mu_, bool rgb = false, bool geom = true, Eigen::Matrix4f initialisation = Eigen::Matrix4f::Identity());
	Eigen::Matrix4f OptimizePhotometricCost(std::vector<DeviceArray2D<float>> & vmap_curr_pyr, std::vector<DeviceArray2D<float>> & nmap_curr_pyr, std::vector<DeviceArray2D<unsigned char>> & rgb_curr_pyr, std::vector<DeviceArray2D<float>> & vmap_prev_pyr, std::vector<DeviceArray2D<float>> & nmap_prev_pyr, std::vector<DeviceArray2D<unsigned char>> & rgb_prev_pyr, CameraModel & intr, Eigen::Matrix4f initialisation = Eigen::Matrix4f::Identity());
	void Evaluation(const char* gth, const char* estimation, const char *output);
	void ClearFeature();

	std::vector<Feature> & Features();
	std::vector<KDTree> & FeatureTrees();
	std::vector<BriskFeature> & BriskFeatures();
	std::vector<Points> & PointClouds();
	Points & PointCloudMeans();
	std::vector<Points> & UnNormalisedPointClouds();
	std::vector<Points> & MeanShiftedPointClouds();
	std::vector<Points> & Normals();
	std::vector<std::pair<int, int> > & PointCorrespondences();
	std::vector<FGRDataTerm> & LineProcesses();
	std::vector<FGRPDADataTerm> & PDALineProcesses();
	int NumLpBelowThresh(float thresh = 0.5f);
	const float & LastResidual(); 
	int GetInliers(const float & angle_thresh, const float & dist_thresh);
	int GetInliers(const float inlier_lp_thresh);
private:
	// containers
	std::vector<Points> pointcloud_;
	std::vector<Points> unnormalised_pointcloud_;
	std::vector<Points> mean_shifted_pointclouds_;
	std::vector<Points> normals_;
	std::vector<Feature> features_;
	std::vector<KDTree> feature_trees_;
	std::vector<BriskFeature> brisk_features_;
	Eigen::Matrix4f TransOutput_;
	std::vector<std::pair<int, int> > corres_;
	std::vector<FGRDataTerm> line_processes_;
	std::vector<FGRPDADataTerm> pda_line_processes_;

	// for normalization
	Points Means;
	std::vector<float> scales_;
	float GlobalScale = 1.0f;
	float StartScale = 1.0f;
	
	float last_residual = 0.0f;

	// some internal functions
	void ReadFeature(const char* filepath, Points& pts, Feature& feat);
	void TransformPoints(Points& points, const Eigen::Matrix4f& Trans);
	void BuildDenseCorrespondence(const Eigen::Matrix4f& gth, 
			Correspondences& corres);
	
	template <typename T, typename A>
	void BuildKDTree(const std::vector<T, A>& data, KDTree* tree);
	template <typename T>
	void SearchKDTree(KDTree* tree,
		const T& input,
		std::vector<int>& indices,
		std::vector<float>& dists,
		int nn);

	template <typename T, typename A>
	void BuildLshIndex(const std::vector<T, A>& data, LshIndex* index);
	template <typename T>
	void SearchLshIndex(LshIndex* index,
		const T& input,
		std::vector<int>& indices,
		std::vector<unsigned int>& dists,
		int nn);

	double div_factor_;
	bool   use_absolute_scale_;
	double max_corr_dist_;
	int    iteration_number_;
	float  tuple_scale_;
	int    tuple_max_cnt_;

	float residual_threshold_;

  public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};