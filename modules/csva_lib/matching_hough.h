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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "misc_functions.h"
#include "Cluster_data.h"
#include "HoughList.h"

class Hough_Transform
{
private:
    std::vector<cv::DMatch> UseTransfForCluster(Cluster_data& NewCData, double initPointPercTh, double initModelPercTh,
		double InitialRotationThresh, double InitialScaleThresh, 
		TransformType transfType, 
		double deleteClusterThresh = 0.2, int hip_check = 0, 
		int RANSAC_iter = 100, double RANSAC_inlierDistT = 0.1);
public:
	HoughListCpp houghListCpp;
	void excludeDuplicates();
	void removeSmallClusters(int threshold, bool independent);
	void ExcludeOne2ManyFromClusters();
	void ExcludeMany2OneFromClusters();
	
    Hough_Transform(double BinOrSize, double BinXSize, double BinYSize, double BinScFactor, cv::Mat image1, cv::Mat image2/*, HoughAccHashType hashtype = HashCpp11*/);
	~Hough_Transform();
    void FillAcc(std::vector<cv::KeyPoint> points1, std::vector<cv::KeyPoint> points2, std::vector<cv::DMatch> matches);
    void FillAccNewBoundary(std::vector<cv::KeyPoint> points1, std::vector<cv::KeyPoint> points2, std::vector<cv::DMatch> matches, int shift);
    void FillAccNewBoundary2(std::vector<cv::KeyPoint> points1, std::vector<cv::KeyPoint> points2, std::vector<cv::DMatch> matches, int shift);
	void FindClusters(int voteThresh);
	Cluster_data MaxCluster();
	
    std::vector<cv::DMatch> getAllClusterMatches();
    std::vector<Cluster_data> clusters;
    std::vector<cv::DMatch> allmatches;

	void sortClusters();

	void UseTransformConstraint(double initPointPercTh, double initModelPercTh, double InitialRotationThresh, 
		double InitialScaleThresh, 
		TransformType transfType, 
		double delClusterThresh = 0.2, int hip_check = 0, int RANSAC_iter = 100, double RANSAC_inlierDistT = 0.1);

	double BinXSize;
	double BinYSize;
	double BinOrSize;
	double BinScFactor;

	int NumOrBin;
	int NumScaleBin;
private:
	int totalSizeofAcc;
    cv::Mat image1;
    cv::Mat image2;
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
	int NumXBin;
	int NumYBin;

	double ImWidth;
	double ImHeight;
};
