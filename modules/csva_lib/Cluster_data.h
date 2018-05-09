#pragma once
#include <stdio.h>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;


enum TransformType { HOMOGENOUS_TRANSFORM, AFFINE_TRANSFORM, 
	SIMILARITY_TRANSFORM, SIMILARITY_RANSAC, OPENCV_HOMOGRAPHY_RANSAC,
	OPENCV_HOMOGRAPHY_LMEDS, OPENCV_AFFINE, OPENCV_SIMILARITY};

class Cluster_data
{
public:
	std::vector<cv::DMatch> matches;
	Cluster_data(const Cluster_data& target);
	Cluster_data operator=(const Cluster_data& other);
	Cluster_data() {position = cv::Point(0,0);};
	int init_size;
	~Cluster_data();
	int scaleBin;
	int orBin;
	int XBin;
	int YBin;
	Mat transfMat;

	void exclude(DMatch match);
	Mat Cluster_data::fitModelParams(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
		TransformType transfType, int extended_output,
		Mat image1 = Mat(), Mat image2 = Mat(), 
		double inlier_dist = 0, int iter = 0, int hip_check = 0, double deleteThresh = 0.2);;

	Mat fitModelParamsSimilarityRansac(const vector<KeyPoint>& keypoints1,
		const vector<KeyPoint>& keypoints2,
		const Mat& image1, const Mat& image2,
		int iterations, int hip_check, double model_distT, double AngleThresh, double ScaleThresh);
	
	vector<DMatch> eliminateOutliers(vector<KeyPoint> points1, vector<KeyPoint> points2, double distThreshPercent, double distThreshModelPercent, double rotationThresh, double ScaleThresh, Mat image1 = Mat(), Mat image2 = Mat(), int extendedOutPut = 0, int inv = 0);
	cv::Point2f position;
	Point2f ClusterPosition(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2);

};

bool compare(Cluster_data a, Cluster_data b);