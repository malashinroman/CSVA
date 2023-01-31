/*
Copyright (C) 2014  Roman Malashin
Copyright (C) 2018  Roman Malashin



This is the Author's implementation of CSVA: "Core" structural verification algorithm [1]. There are few differences with the paper. 

[1] Malashin R.O. Core algorithm for structural verification of keypoint matches. Intelligent Systems Reference Library. Computer Vision in Control Systems-3. 2018. P. 251-286


Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 
 *The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once
#include <stdio.h>
#include <opencv2/core.hpp>

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
    ~Cluster_data() = default;
	int scaleBin;
	int orBin;
	int XBin;
	int YBin;
    cv::Mat transfMat;

    void exclude(cv::DMatch match);
    cv::Mat fitModelParams(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
		TransformType transfType, int extended_output,
        cv::Mat image1 = cv::Mat(), cv::Mat image2 = cv::Mat(),
		double inlier_dist = 0, int iter = 0, int hip_check = 0, double deleteThresh = 0.2);;

    cv::Mat fitModelParamsSimilarityRansac(const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const cv::Mat& image1, const cv::Mat& image2,
		int iterations, int hip_check, double model_distT, double AngleThresh, double ScaleThresh);
	
    std::vector<cv::DMatch> eliminateOutliers(std::vector<cv::KeyPoint> points1, std::vector<cv::KeyPoint> points2, double distThreshPercent, double distThreshModelPercent, double rotationThresh, double ScaleThresh, cv::Mat image1 = cv::Mat(), cv::Mat image2 = cv::Mat(), int extendedOutPut = 0, int inv = 0);
	cv::Point2f position;
    cv::Point2f ClusterPosition(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2);

};

bool compare(Cluster_data a, Cluster_data b);
