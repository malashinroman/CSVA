/*
Copyright (C) 2014  Roman Malashin
Copyright (C) 2018  Roman Malashin

*

This is the Author's implementation of CSVA: "Core" structural verification algorithm [1]. Few unpublished modifications and extensions are provided.

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

#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_lib/feature_extractors.h"
#include "csva_lib/csva.h"
#include "3Drecognition.h"
#include <time.h>
using namespace cv;
using namespace std;

vector<DMatch> call3D(char* image1_path, char* image2_path, Mat* matchresult, vector<KeyPoint>& points1, vector<KeyPoint>& points2, int LocalMatchtype = 352, int consoleOuput = 0, int graphicalOutput = 0)
{
	clock_t start = clock();

	Mat image1 = imread(image1_path);
	Mat image2 = imread(image2_path);
	OpenCVfeatures feat;

	if((image1.size().width ==0) || (image2.size().width ==0))
	{
		printf("cannot load on of images: %s or %s\n", image1_path, image2_path);
	}
	clock_t finish = clock();
	int t1 = (finish - start) * 1000 / CLOCKS_PER_SEC;
	int tmp1, tmp2, tmp3;
	std::vector<DMatch> matches = feat.getLocalPatchMatches2(image1, image2, points1, points2, LocalMatchtype, &tmp1, &tmp2, &tmp3, 0);
	clock_t chp = clock();
	int t2 = (chp - finish) * 1000 / CLOCKS_PER_SEC;
	vector<DMatch> goodmatches;
	double confidence[6];
	csva::filter_matches(points1, points2, matches, image1, image2, 1, LocalMatchtype, goodmatches, confidence, 0.001);
	return goodmatches;
}