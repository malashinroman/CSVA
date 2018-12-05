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

namespace csva
{
	enum geometry_mode {  AEROSPACE = 0, THREEDIM_SCENE = 1};
	CSVA_LIB_API cv::Mat filter_matches(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, const std::vector<cv::DMatch>& matches,
		const cv::Mat& im1, const  cv::Mat& im2, geometry_mode mode,
		int type, std::vector<cv::DMatch> &inliers,
		double* confidence, double LoweProb);

	CSVA_LIB_API void primary_filtering(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2,
		const std::vector<cv::DMatch>& matches, float NNthresh, std::vector<cv::DMatch> &inliers);

	CSVA_LIB_API std::array<double, 6> confidence_estimation(std::vector<cv::DMatch>& inliers, const cv::Mat& PT, std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2,
		const std::vector<cv::DMatch> &excludedMatches, cv::Mat im1, cv::Mat im2, int mode, int type, double LoweProb);

	CSVA_LIB_API void verify_clusters(const std::vector< std::vector<cv::DMatch> >& clusters, std::vector< std::vector<cv::DMatch> >& filtered,
		std::vector<cv::Mat>& transforms, const std::vector<cv::KeyPoint>& kpts1,
		const std::vector<cv::KeyPoint>& kpts2, const cv::Mat& image1, const cv::Mat& image2);

	CSVA_LIB_API void hough_transform(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2,
		const std::vector<cv::DMatch>& matches, const cv::Mat& image1, const cv::Mat& image2,
		std::vector<std::vector<cv::DMatch> >& clusters, int vote_thresh = 0);

}