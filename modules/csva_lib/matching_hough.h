#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "misc_functions.h"
#include "Cluster_data.h"
#include "HoughList.h"
using namespace cv;

//enum HoughAccHashType { HashCpp11, HashManual, HashManualND};
class Hough_Transform
{
private:
	vector<DMatch> UseTransfForCluster(Cluster_data& NewCData, double initPointPercTh, double initModelPercTh, 
		double InitialRotationThresh, double InitialScaleThresh, 
		TransformType transfType, 
		double deleteClusterThresh = 0.2, int hip_check = 0, 
		int RANSAC_iter = 100, double RANSAC_inlierDistT = 0.1);
public:
	HoughListCpp houghListCpp;
	int extended_output[4];
	void excludeDuplicates();
	void removeSmallClusters(int threshold, bool independent);
	void ExcludeOne2ManyFromClusters();
	void ExcludeMany2OneFromClusters();
	
	Hough_Transform(double BinOrSize, double BinXSize, double BinYSize, double BinScFactor, Mat image1, Mat image2/*, HoughAccHashType hashtype = HashCpp11*/);
	~Hough_Transform();
	void FillAcc(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<DMatch> matches);
	void FillAccNewBoundary(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<DMatch> matches, int shift);
	void FillAccNewBoundary2(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<DMatch> matches, int shift);
	void FindClusters(int voteThresh);
	Cluster_data MaxCluster();
	
	//void enrichClusters(double DistanceThreshProj, double DistanceThreshModel, double RotationThresh, double ScaleThresh, int transfType, vector<DMatch> matches = vector<DMatch>());

	vector<DMatch> getAllClusterMatches();
	vector<Cluster_data> clusters;
	vector<DMatch> allmatches;

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
	Mat image1;
	Mat image2;
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	//HoughAccHashType hashtype = HashCpp11;
	int NumXBin;
	int NumYBin;

	double ImWidth;
	double ImHeight;
};
