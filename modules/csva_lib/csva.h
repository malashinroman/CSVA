/*
Copyright (C) 2014  Roman Malashin
Copyright (C) 2018  Roman Malashin

All rights reserved.

This is the Author's implementation of CSVA: "Core" structural verification algorithm [1]. Few unpublished modifications extensions are provided.

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

#pragma once;
#include <opencv2/core.hpp>



#if defined(_MSC_VER)
	#ifdef csva_lib_EXPORTS
		#define CSVA_LIB_API __declspec(dllexport) 
	#else
		#define CSVA_LIB_API __declspec(dllimport) 
	#endif
#elif defined(__GNUC__)
	#ifdef aero_lib_EXPORTS
		#define CSVA_LIB_API __attribute__((visibility("default")))
	#else
		#define CSVA_LIB_API
	#endif
#else
//  do nothing and hope for the best?
		#define CSVA_LIB_API
	#pragma warning Unknown dynamic link import/export semantics.
#endif


#include <opencv2/core/core.hpp>
#include <array>
using namespace cv;
using namespace std;
namespace csva
{
	CSVA_LIB_API cv::Mat filter_matches(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const vector<DMatch>& matches,
		const Mat& im1, const  Mat& im2, int mode,
		int type, vector<DMatch> &inliers,
		double* confidence, double LoweProb);
	
	CSVA_LIB_API void primary_filtering(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2,
		const vector<DMatch>& matches, float NNthresh, vector<DMatch> &inliers);

	CSVA_LIB_API std::array<double, 6> confidence_estimation(vector<DMatch>& inliers, const Mat& PT, vector<KeyPoint> kpts1, vector<KeyPoint> kpts2,
		const vector<DMatch> &excludedMatches, Mat im1, Mat im2, int mode, int type, double LoweProb);
	
	CSVA_LIB_API void verify_clusters(const vector< vector<DMatch> >& clusters, vector< vector<DMatch> >& filtered,
		vector<Mat>& transforms, const vector<KeyPoint>& kpts1,
		const vector<KeyPoint>& kpts2, const Mat& image1, const Mat& image2);

	//CSVA_LIB_API void hough_transform(vector<KeyPoint> kpts1, vector<KeyPoint> kpts2, vector<DMatch> matches,
	//	Mat image1, Mat image2, vector<vector<DMatch> >& clusters, int vote_thresh = 0);
	//
	CSVA_LIB_API void hough_transform(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2,
		const vector<DMatch>& matches, const Mat& image1, const Mat& image2,
		vector<vector<DMatch> >& clusters, int vote_thresh=0);

	//CSVA_LIB_API void hough_transform2(vector<KeyPoint> kpts1, vector<KeyPoint> kpts2, vector<DMatch> matches,
	//	Mat image1, Mat image2, vector<vector<DMatch> >& clusters, int vote_thresh = 0);
}