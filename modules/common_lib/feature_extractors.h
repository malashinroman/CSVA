#pragma once

#include <opencv2/features2d.hpp>
#include <stdio.h>
#include <opencv2/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>


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

	cv::Ptr<cv::SiftFeatureDetector> detectorSift;
	// cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detectorSift;
	cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> detectorSurf;
	cv::Ptr<cv::FastFeatureDetector> detectorFast; // detectorFast(20)
	cv::Ptr<cv::xfeatures2d::VGG> vggfeatures;
	cv::Ptr<cv::MSER> detectorMser;
	cv::Ptr<cv::xfeatures2d::StarDetector> detectorStar;
	cv::Ptr<cv::SimpleBlobDetector> detectorSimpleBlob;
	cv::Ptr<cv::AKAZE> detectorAkaze;
	cv::Ptr<cv::AgastFeatureDetector> detectorAgast;
	//Ptr<GoodFeaturesToTrackDetector> detectorGoodFeat;
	cv::Ptr<cv::GFTTDetector> detectorGoodFeat;
	cv::Ptr<cv::xfeatures2d::LATCH> detectorLatch;
	cv::Ptr<cv::xfeatures2d::DAISY> detectorDaisy;

	//cv::xfeatures2d::
	cv::Ptr<cv::ORB> detectorOrb;
	//BRISK detectorBrisk(10, 4);
	cv::Ptr<cv::BRISK> detectorBrisk;
  cv::Ptr<cv::SiftFeatureDetector> extractorSift;

	// cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractorSift;
	cv::Ptr<cv::xfeatures2d::SurfDescriptorExtractor> extractorSurf;
	//Ptr<OpponentColorDescriptorExtractor> extractorSURFOpponent(new SurfDescriptorExtractor);

	cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractorBrief;
	cv::Ptr<cv::BRISK> extractorBrisk;
	cv::Ptr<cv::ORB> extractorOrb;
	cv::Ptr<cv::xfeatures2d::FREAK> extractorFreak;
	cv::Ptr<cv::xfeatures2d::LATCH> extractorLatch;
	cv::Ptr<cv::AKAZE> extractorAkaze;
	cv::Ptr<cv::xfeatures2d::LUCID> extractorLucid;
	cv::Ptr<cv::xfeatures2d::DAISY> extractorDaisy;

	//GfemFeatureDetectorExtractor detectorExtractorGfem;
	cv::FlannBasedMatcher matcher1;
	cv::BFMatcher matcher2;

	cv::Ptr<cv::DescriptorMatcher> matcher3;
	void initDetectorDescriptors();

	void releaseDetectorDescriptors();
	OpenCVfeatures();
	~OpenCVfeatures();
	std::vector<cv::DMatch> getLocalPatchMatches2(cv::Mat image1, cv::Mat image2,
		std::vector<cv::KeyPoint>& points1, std::vector<cv::KeyPoint>& points2,
		int type, int ConsoleOutput);

	std::vector<cv::DMatch> getMatches(cv::Mat descs1, cv::Mat descs2, int type, int ConsoleOutput);
	cv::Mat getDescriptors(cv::Mat image, std::vector<cv::KeyPoint>& points, int type, int ConsoleOutput);
	std::vector<cv::KeyPoint> getKeyPoints(cv::Mat image, int type, int ConsoleOutput);
	std::vector<cv::KeyPoint> refineNotUniqueKeypoints(std::vector<cv::KeyPoint> keypoints, cv::Mat image, int k, double nnr_thresh, int type);
};
