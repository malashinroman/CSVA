/*
Copyright (C) 2014  Roman Malashin
Copyright (C) 2018  Roman Malashin

*

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <set>

#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>


#include "common_lib/feature_extractors.h"
#include "csva_lib/csva.h"
#include <opencv2/opencv.hpp>

#define SAVE_JPG_QUALITY 100

///
cv::Mat printMatches(std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, std::vector<cv::DMatch> matches, cv::Mat image1, cv::Mat image2, cv::DrawMatchesFlags flags)
{
    cv::Mat outImage;
    drawMatches(image1, pts1, image2, pts2, matches, outImage, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0), std::vector<char>(), flags);
	return outImage;
}

///
void getMatchedKeypoints(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>&  keypoints1, const std::vector<cv::KeyPoint>&  keypoints2,
    std::vector<cv::KeyPoint>& matchedkeypoints1, std::vector<cv::KeyPoint>& matchedkeypoints2)
{
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		matchedkeypoints1.push_back(keypoints1.at(matches.at(i).queryIdx));
		matchedkeypoints2.push_back(keypoints2.at(matches.at(i).trainIdx));
	}
}

///
cv::Mat homographyOpencv(std::vector<cv::DMatch> inliers, std::vector<cv::KeyPoint> kpts1_, std::vector<cv::KeyPoint> kpts2_)
{
    std::vector<cv::KeyPoint> kpts1;
    std::vector<cv::KeyPoint> kpts2;

    getMatchedKeypoints(inliers, kpts1_, kpts2_, kpts1, kpts2);
    std::vector<cv::Point2f> points2D1;
    std::vector<cv::Point2f> points2D2;
    cv::Mat P1 = cv::Mat(inliers.size(), 2, CV_64F, 0.);
    cv::Mat P2 = cv::Mat(inliers.size(), 2, CV_64F, 0.);
    cv::Mat PT;
    for (int i = 0; i < kpts1.size(); i++)
    {
        P1.at<double>(i, 0) = kpts1.at(i).pt.x;
        P1.at<double>(i, 1) = kpts1.at(i).pt.y;
        P2.at<double>(i, 0) = kpts2.at(i).pt.x;
        P2.at<double>(i, 1) = kpts2.at(i).pt.y;
        points2D1.push_back(kpts1.at(i).pt);
        points2D2.push_back(kpts2.at(i).pt);
    }
    if (points2D1.size() > 3)
        PT = findHomography(points2D1, points2D2, cv::LMEDS);

    return PT;
}

///
cv::Mat mergeImages(cv::Mat img1, cv::Mat img2, int topdown)
{
	if (topdown)
	{
		int newWidth = img1.size().width > img2.size().width ? img1.size().width : img2.size().width;
        cv::Mat outImage(cv::Size(newWidth, img1.size().height + img2.size().height), CV_8UC3);
        cv::Mat left_roi(outImage, cv::Rect(0, 0, img1.size().width, img1.size().height));
		img1.copyTo(left_roi);
        cv::Mat down_roi(outImage, cv::Rect(0, img1.size().height, img2.size().width, img2.size().height));
		img2.copyTo(down_roi);
		return outImage;
	}
	else
	{
		int newWidth = img1.size().width + img2.size().width;
		int newHeight = img1.size().height > img2.size().height ? img1.size().height : img2.size().height;
        cv::Mat outImage(cv::Size(newWidth, newHeight), CV_8UC3);
        cv::Mat left_roi(outImage, cv::Rect(0, 0, img1.size().width, img1.size().height));
		img1.copyTo(left_roi);
        cv::Mat right_roi(outImage, cv::Rect(img1.size().width, 0, img2.size().width, img2.size().height));
		img2.copyTo(right_roi);
		return outImage;
	}
}

///
std::vector<std::string> splitpath(const std::string& str, const std::set<char> delimiters)
{
	std::vector<std::string> result;

	char const* pch = str.c_str();
	char const* start = pch;
	for (; *pch; ++pch)
	{
		if (delimiters.find(*pch) != delimiters.end())
		{
			if (start != pch)
			{
				std::string str(start, pch);
				result.push_back(str);
			}
			else
			{
				result.push_back("");
			}
			start = pch + 1;
		}
	}
	result.push_back(start);

	return result;
}

///
int main (int argc, char* argv[])
{
	if (argc < 6)
	{
        std::cout << "match_aero image1 image2 scale1 scale2 type \n" <<
            "image1: object photo" << std::endl <<
            "image1: plan photo" << std::endl <<
            "scale1: coeff to resize image1" << std::endl <<
            "scale2: coeff to resize image2" << std::endl <<
            "type: 1 or 2 (1 for fast, 2 - robust)" << std::endl <<
            "example: match_aero im1.jpg im2.jpg 0.5 0.5 1" << std::endl;
		return -1;
	}
    cv::Mat im1_ = cv::imread(argv[1]);
    cv::Mat im2_ = cv::imread(argv[2]);
	float scale1 = atof(argv[3]);
	float scale2 = atof(argv[4]);

	int type = 352;
	if (atoi(argv[5]) == 2)
	{
        std::cout << "robust regime" << std::endl;
		type = 212;
	}
	else if(atoi(argv[5]) == 1)
	{
        std::cout << "fast regime" << std::endl;
		type = 352;
	}
	else if (atoi(argv[5]) > 100)
	{
		type = atoi(argv[5]);
	}
	else
	{
        std::cout << "unknown regime" << std::endl;
		return 0;
	}

    std::cout << "image 1: " << argv[1] << std::endl;
    std::cout << "image 2: " << argv[2] << std::endl;
    std::cout << "scale coeff 1: " << scale1 << std::endl;
    std::cout << "scale coeff 2: " << scale2 << std::endl;

	assert(!im1_.empty());
	assert(!im2_.empty());
    std::vector<cv::KeyPoint> kpts1;
    std::vector<cv::KeyPoint> kpts2;
    cv::Mat image1, image2, image1c, image2c, image1_process, image2_process;

	if (im1_.channels() == 3)
	{
        cvtColor(im1_, image1, cv::COLOR_BGR2GRAY);
		image1c = im1_.clone();
	}
	else
	{
		image1 = im1_.clone();
        cvtColor(im1_, image1c, cv::COLOR_GRAY2BGR);
	}

	if (im2_.channels() == 3)
	{
        cvtColor(im2_, image2, cv::COLOR_BGR2GRAY);
		image2c = im2_.clone();
	}
	else
	{
		image2 = im2_.clone();
        cvtColor(im2_, image2c, cv::COLOR_GRAY2BGR);
	}

    resize(image1, image1_process, cv::Size(), scale1, scale1);
    resize(image2, image2_process, cv::Size(), scale2, scale2);
	//double scalef = scale2 / scale1;

    cv::Mat preT1, preT2;
	double tmp[9] = { scale1, 0, 0, 0, scale1, 0, 0, 0, 1 };
    cv::Mat(3, 3, CV_64F, tmp).copyTo(preT1);
	double tmp2[9] = { scale2, 0, 0, 0, scale2, 0, 0, 0, 1 };
    cv::Mat(3, 3, CV_64F, tmp2).copyTo(preT2);


	OpenCVfeatures feat;
    std::vector<cv::DMatch> matches = feat.getLocalPatchMatches2(image1_process, image2_process, kpts1, kpts2, type, 1);
	
    cv::Mat image1_proj, image1_projc;
    cv::Mat outim;
    cv::Mat imageMask, xoredImage, alignedImages;
    cv::Mat imageMaskc, xoredImagec, alignedImagesc;
	//char* res_file_name = "shalalla.jpg";
    std::vector<cv::DMatch> inliers;
	double confide[6];
    std::cout << "number of generated matches: " << matches.size() << std::endl;
	clock_t start = clock();
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	clahe->apply(image1, image1);
	clahe->apply(image2, image2);

    cv::Mat PT = csva::filter_matches(kpts1, kpts2, matches, image1_process, image2_process, csva::geometry_mode::AEROSPACE, 352, inliers, confide, 0.01);
	printf("CSVA time = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock(); 
    cv::Mat res = printMatches(kpts1, kpts2, inliers, image2_process, image2_process, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
    std::string resultFolder = std::string(argv[1]) + "_res";
    std::filesystem::create_directory(resultFolder);
    cv::imwrite(resultFolder + "/matches.jpg", res);
    cv::Mat result;
	if (!PT.empty())
	{
		//we will aply transformation to the images of the original size
		PT = preT2.inv() * PT * preT1;
		cv::warpPerspective(image1, image1_proj, PT, image2.size());
		cv::warpPerspective(image1c, image1_projc, PT, image2.size());

        std::vector<cv::Mat> channels;
        cv::Mat g = cv::Mat::zeros(image2.size(), CV_8UC1);
		channels.push_back(g);
		channels.push_back(image2);
		channels.push_back(image1_proj);
		merge(channels, outim);
        cv::threshold(image1_proj, imageMask, 1, 255, cv::THRESH_BINARY);
		cv::subtract(image2, imageMask, xoredImage);
		cv::add(xoredImage, image1_proj, alignedImages);
        std::vector<cv::Mat> xoredLayers;
		xoredLayers.push_back(imageMask);
		xoredLayers.push_back(imageMask);
		xoredLayers.push_back(imageMask);
		merge(xoredLayers, imageMaskc);
		cv::subtract(image2c, imageMaskc, xoredImagec);
		cv::add(xoredImagec, image1_projc, alignedImagesc);
		alignedImagesc.copyTo(result);


		alignedImagesc.copyTo(result);
		//!imwrite(DebugInfoPath + "/kpts_m.jpg", res);
        std::vector<int> compression_params;
#if (CV_MAJOR_VERSION >= 4)
		compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
#else
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
#endif
		compression_params.push_back(SAVE_JPG_QUALITY);

		imwrite(resultFolder + "/reg_green.jpg", outim, compression_params);
		imwrite(resultFolder + "/alignedc.jpg", alignedImagesc, compression_params);
		imwrite(resultFolder + "/obj.jpg", image1c, compression_params);
		imwrite(resultFolder + "/scene.jpg", image2c, compression_params);
		imwrite(resultFolder + "/transf_obj.jpg", image1_projc, compression_params);
        std::cout << "confidence: " << confide[0] << std::endl;
	}
	else
	{
		printf("object is not detected!\n");
        cv::Mat m = mergeImages(image1, image2, 1);
		m.copyTo(result);
	}

    return 0;
}  
