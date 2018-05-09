#pragma once

#include <stdio.h>
#include <opencv2/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

#define SIMPLEBLOB_DETECTOR_TYPE 27
#define GOODFEAT_DETECTOR_TYPE -1
#define HOG_DESCRIPTOR_TYPE 1111
#define OPPONENT_SURF_DESCRIPTOR 111
#define STAR_DETECTOR_TYPE 111

#define STRUCTURAL_ELEMENTS 1703
#define AGAST_DETECTOR_TYPE 0
#define SIFT_DETECTOR_TYPE 1
#define SURF_DETECTOR_TYPE 2
#define VGG_DETECTOR_TYPE 3
#define AKAZE_DETECTOR_TYPE 32324 //!
#define FAST_DETECTOR_TYPE 4
#define MSER_DETECTOR_TYPE 5
#define DAISY_DETECTOR_TYPE 6
#define BRISK_DETECTOR_TYPE 7
#define DENSE_DETECTOR_TYPE 8
#define ORB_DETECTOR_TYPE 9

#define LUCID_DESCRIPTOR_TYPE 0
#define SIFT_DESCRIPTOR_TYPE 10
#define SURF_DESCRIPTOR_TYPE 20
#define AKAZE_DESCRIPTOR_TYPE 30
#define BRIEF_DESCRIPTOR_TYPE 40
#define BRISK_DESCRIPTOR_TYPE 50
#define ORB_DESCRIPTOR_TYPE 60
#define FREAK_DESCRIPTOR_TYPE 3232//!
#define VGG_DESCRIPTOR_TYPE 70
#define DAISY_DESCRIPTOR_TYPE 80 

#define LATCH_DESCRIPTOR_TYPE 90


#define FLANN_MATCHER_TYPE 100
#define BF_MATCHER_TYPE 200
#define HAMMING_MATCHER 300
#define KNN_MATCHER 400
#define CROSSCHECK_MATCHER_TYPE 500
#define BOW_MATCHING 600

class OpenCVfeatures
{
public:
	bool globalInitializationDescriptors;
	Ptr<SiftFeatureDetector> detectorSift;
	Ptr<SurfFeatureDetector> detectorSurf;
	Ptr<FastFeatureDetector> detectorFast; // detectorFast(20)
	Ptr<VGG> vggfeatures;
	Ptr<cv::MSER> detectorMser;
	Ptr<StarDetector> detectorStar;
	Ptr<SimpleBlobDetector> detectorSimpleBlob;
	Ptr<AKAZE> detectorAkaze;
	Ptr<AgastFeatureDetector> detectorAgast;
	//Ptr<GoodFeaturesToTrackDetector> detectorGoodFeat;
	Ptr<GFTTDetector> detectorGoodFeat;
	Ptr<LATCH> detectorLatch;
	Ptr<DAISY> detectorDaisy;

	//cv::xfeatures2d::
	Ptr<cv::ORB> detectorOrb;
	//BRISK detectorBrisk(10, 4);
	Ptr<cv::BRISK> detectorBrisk;

	Ptr<SiftDescriptorExtractor> extractorSift;
	Ptr<SurfDescriptorExtractor> extractorSurf;
	//Ptr<OpponentColorDescriptorExtractor> extractorSURFOpponent(new SurfDescriptorExtractor);

	Ptr<BriefDescriptorExtractor> extractorBrief;
	Ptr<BRISK> extractorBrisk;
	Ptr<cv::ORB> extractorOrb;
	Ptr<FREAK> extractorFreak;
	Ptr<LATCH> extractorLatch;
	Ptr<AKAZE> extractorAkaze;
	Ptr<LUCID> extractorLucid;
	Ptr<DAISY> extractorDaisy;

	//GfemFeatureDetectorExtractor detectorExtractorGfem;
	FlannBasedMatcher matcher1;
	BFMatcher matcher2;

	Ptr<DescriptorMatcher> matcher3;
	void initDetectorDescriptors();

	void releaseDetectorDescriptors();
	OpenCVfeatures();
	~OpenCVfeatures();
	vector<DMatch> getLocalPatchMatches2(Mat image1, Mat image2,
		vector<KeyPoint>& points1, vector<KeyPoint>& points2,
		int type, int* detectionTime, int* descriptionTime,
		int* matchingTime, int ConsoleOutput);

	vector<DMatch> getMatches(Mat descs1, Mat descs2, int type, int ConsoleOutput);
	Mat getDescriptors(Mat image, vector<KeyPoint>& points, int type, int ConsoleOutput);
	vector<KeyPoint> getKeyPoints(Mat image, int type, int ConsoleOutput);
	vector<KeyPoint> refineNotUniqueKeypoints(vector<KeyPoint> keypoints, Mat image, int k, double nnr_thresh, int type);
};