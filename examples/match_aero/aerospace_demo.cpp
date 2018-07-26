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
#include <opencv2/core.hpp>

#include <iostream>
#include "common_lib/feature_extractors.h"
#include "csva_lib/csva.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <boost/filesystem.hpp>
#include <set>
using namespace std;
using namespace cv;
#define SAVE_JPG_QUALITY 100
#include <direct.h>
int makeDirectory(string folder)
{
	//alternative is	
	mkdir(folder.c_str());
	//boost::filesystem::create_directory(folder);

//#ifdef WIN32
//	mkdir(folder.c_str());
//#else
//	mkdir(folder.c_str(), 0700);
//#endif
	return 0;
}

Mat printMatches(vector<KeyPoint> pts1, vector<KeyPoint> pts2, vector<DMatch> matches, Mat image1, Mat image2, int flags)
{
	Mat outImage;
	drawMatches(image1, pts1, image2, pts2, matches, outImage, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0), vector<char>(), flags);
	return outImage;
}
void getMatchedKeypoints(const vector<DMatch>& matches, 
	const vector<KeyPoint>&  keypoints1, const vector<KeyPoint>&  keypoints2, 
	vector<KeyPoint>& matchedkeypoints1, vector<KeyPoint>& matchedkeypoints2)
{
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		matchedkeypoints1.push_back(keypoints1.at(matches.at(i).queryIdx));
		matchedkeypoints2.push_back(keypoints2.at(matches.at(i).trainIdx));
	}
}
Mat homographyOpencv(vector<DMatch> inliers, vector<KeyPoint> kpts1_, vector<KeyPoint> kpts2_)
{
	vector<KeyPoint> kpts1;
	vector<KeyPoint> kpts2;

	getMatchedKeypoints(inliers, kpts1_, kpts2_, kpts1, kpts2);
	vector<Point2f> points2D1;
	vector<Point2f> points2D2;
	Mat P1 = Mat(inliers.size(), 2, CV_64F, 0.);
	Mat P2 = Mat(inliers.size(), 2, CV_64F, 0.);
	Mat PT;
	for (int i = 0; i < kpts1.size(); i++)
	{
		P1.at<double>(i, 0) = kpts1.at(i).pt.x;
		P1.at<double>(i, 1) = kpts1.at(i).pt.y;
		P2.at<double>(i, 0) = kpts2.at(i).pt.x;
		P2.at<double>(i, 1) = kpts2.at(i).pt.y;
		points2D1.push_back(kpts1.at(i).pt);
		points2D2.push_back(kpts2.at(i).pt);
	}
	if (points2D1.size() < 4)
	{
		PT = Mat();
	}
	else
	{
		PT = findHomography(points2D1, points2D2, CV_LMEDS);
	}
	return PT;
}
Mat mergeImages(Mat img1, Mat img2, int topdown)
{
	if (topdown)
	{
		int newWidth = img1.size().width > img2.size().width ? img1.size().width : img2.size().width;
		Mat outImage(Size(newWidth, img1.size().height + img2.size().height), CV_8UC3);
		Mat left_roi(outImage, Rect(0, 0, img1.size().width, img1.size().height));
		img1.copyTo(left_roi);
		Mat down_roi(outImage, Rect(0, img1.size().height, img2.size().width, img2.size().height));
		img2.copyTo(down_roi);
		return outImage;
	}
	else
	{
		int newWidth = img1.size().width + img2.size().width;
		int newHeight = img1.size().height > img2.size().height ? img1.size().height : img2.size().height;
		Mat outImage(Size(newWidth, newHeight), CV_8UC3);
		Mat left_roi(outImage, Rect(0, 0, img1.size().width, img1.size().height));
		img1.copyTo(left_roi);
		Mat right_roi(outImage, Rect(img1.size().width, 0, img2.size().width, img2.size().height));
		img2.copyTo(right_roi);
		return outImage;
	}
}
std::vector<std::string> splitpath(
	const std::string& str
	, const std::set<char> delimiters)
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


int main (int argc, char* argv[])
{
	if (argc < 6)
	{
		cout << "match_aero image1 image2 scale1 scale2 type /*type = 1 -fast, type = 2 - robust*/" << endl;
		return -1;
	}
	Mat im1 = imread(argv[1]);
	Mat im2 = imread(argv[2]);
	float scale1 = atof(argv[3]);
	float scale2 = atof(argv[4]);

	int type = 352;
	if (atoi(argv[5]) == 2)
	{
		cout<<"robust regime" <<endl;
		type = 212;
	}
	else if(atoi(argv[5]) == 1)
	{
		cout << "fast regime" <<endl;
		type = 352;
	}
	else if (atoi(argv[3]) > 100)
	{
		type = atoi(argv[3]);
	}
	else
	{
		cout << "unknown regime" <<endl;
		return 0;
	}

	cout << "image 1: " << argv[1] << endl;
	cout << "image 2: " << argv[2] << endl;
	cout << "scale coeff 1: " << scale1 << endl;
	cout << "scale coeff 2: " << scale2 << endl;

	assert(!im1.empty());
	assert(!im2.empty());
	vector<KeyPoint> kpts1;
	vector<KeyPoint> kpts2;
	int dettime, desctime, mtime;
	Mat image1, image2, image1c, image2c;

	resize(im1, im1, Size(), scale1, scale1);
	resize(im2, im2, Size(), scale2, scale2);
	if (im1.channels() == 3)
	{
		cvtColor(im1, image1, CV_BGR2GRAY);
		image1c = im1.clone();
	}
	else
	{
		image1 = im1.clone();
		cvtColor(im1, image1c, CV_GRAY2BGR);
	}

	if (im2.channels() == 3)
	{
		cvtColor(im2, image2, CV_BGR2GRAY);
		image2c = im2.clone();
	}
	else
	{
		image2 = im2.clone();
		cvtColor(im2, image2c, CV_GRAY2BGR);
	}

	OpenCVfeatures feat;
	vector<DMatch> matches = feat.getLocalPatchMatches2(image1, image2, kpts1, kpts2, type, &dettime, &desctime, &mtime, 1);
	
	
	Mat image1_proj, image1_projc;
	Mat outim;
	Mat imageMask, xoredImage, alignedImages;
	Mat imageMaskc, xoredImagec, alignedImagesc;
	//char* res_file_name = "shalalla.jpg";
	vector<DMatch> inliers;
	double confide[6];
	cout << "number of generated matches: " << matches.size() << endl;
	clock_t start = clock();
	Mat PT = csva::filter_matches(kpts1, kpts2, matches, image1, image2, 0, 352, inliers, confide, 0.01);
	printf("CSVA time = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock(); 
	Mat res = printMatches(kpts1, kpts2, inliers, image1, image2, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	std::set<char> delims{ '\\', '/' };
	std::vector<std::string> path = splitpath(argv[1], delims);
	string resultFolder = path.back();
	makeDirectory(resultFolder);
	imwrite(resultFolder + "/matches.jpg", res);
	Mat result;
	if (!PT.empty())
	{
		cv::warpPerspective(image1, image1_proj, PT, image2.size());
		cv::warpPerspective(image1c, image1_projc, PT, image2.size());

		vector<Mat> channels;
		Mat g = Mat::zeros(image2.size(), CV_8UC1);
		channels.push_back(g);
		channels.push_back(image2);
		channels.push_back(image1_proj);
		merge(channels, outim);
		cv::threshold(image1_proj, imageMask, 1, 255, CV_THRESH_BINARY);
		cv::subtract(image2, imageMask, xoredImage);
		cv::add(xoredImage, image1_proj, alignedImages);
		vector<Mat> xoredLayers;
		xoredLayers.push_back(imageMask);
		xoredLayers.push_back(imageMask);
		xoredLayers.push_back(imageMask);
		merge(xoredLayers, imageMaskc);
		cv::subtract(image2c, imageMaskc, xoredImagec);
		cv::add(xoredImagec, image1_projc, alignedImagesc);
		alignedImagesc.copyTo(result);


		alignedImagesc.copyTo(result);
		//!imwrite(DebugInfoPath + "/kpts_m.jpg", res);
		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(SAVE_JPG_QUALITY);

		imwrite(resultFolder + "/reg_green.jpg", outim, compression_params);
		imwrite(resultFolder + "/alignedc.jpg", alignedImagesc, compression_params);
		imwrite(resultFolder + "/obj.jpg", image1c, compression_params);
		imwrite(resultFolder + "/scene.jpg", image2c, compression_params);
		imwrite(resultFolder + "/transf_obj.jpg", image1_projc, compression_params);
		cout << "confidence: " << confide[0] <<endl;
	}
	else
	{
		printf("object is not detected!\n");
		Mat m = mergeImages(image1, image2, 1);
		m.copyTo(result);
	}
        return 0;
}  

