/*
Copyright (C) 2014  Roman Malashin
Copyright (C) 2018  Roman Malashin

All rights reserved.

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

using namespace cv;
using namespace std;

#define max_(a, b) ((a) > (b) ? (a) : (b))
#define min_(a, b) ((a) < (b) ? (a) : (b))
#define image_diag(image) sqrt((double)(image.size().width*image.size().width + image.size().height*image.size().height))


template <typename T> string tostr(const T& t) { std::ostringstream os; os<<t; return os.str(); };
void decomposeAff(const Mat& transfMat, Mat& Rot, Mat& Shear, Mat& Scale, double& Theta, double& shiftX, double& shiftY, double& scale, double& p, double& r);

void decomposeAffLutsiv(const Mat& transfMat, double* scale, double* theta, double* ascale, double* direction);

Rect getImageProjBbx(const Mat& image1, const Mat& trM);
Point2f WrapTransform(Point2f SamplePoint, const Mat& trMatrix);

vector<Point2f> WrapTransform(vector<Point2f>& SamplePoint, const Mat& trMatrix);

inline double euclideanDistacne(Point2f p1, Point2f p2);

inline double  convertOpencvAngle2GoodAngle(double angle_opencv);

Point2f  predictModelPosition(const KeyPoint& point1, const KeyPoint& point2, Point2f ModelPoint);

Point2f  predictModelPosition2(const KeyPoint& point1, const KeyPoint& point2, Point2f ModelPoint);

int independentMatches(vector<DMatch> matches, vector<KeyPoint> pts1, vector<KeyPoint> pts2, Mat im1, Mat im2);

bool checkMatchIn(vector<DMatch> matches, DMatch newm);
bool checkMatchIn(vector<DMatch> matches, DMatch newm, int& indx);
double matchDistance(const DMatch& m1, const DMatch& m2, const vector<KeyPoint>& pts1, const vector<KeyPoint>& pts2);

double getMutualAngle(const KeyPoint& p1, const KeyPoint& p2);
double getAngleDif(double angle1, double angle2);
double getMutualScale(const KeyPoint& p1, const KeyPoint& p2);
void inline getMutualShifts(const KeyPoint& p1, const KeyPoint& p2, double& shiftx, double& shifty);


void getMatchedKeypoints(const vector<DMatch>& matches,
	const vector<KeyPoint>& keypoints1, const vector<KeyPoint>&  keypoints2,
	vector<KeyPoint>& matchedkeypoints1, vector<KeyPoint>& matchedkeypoints2);

void sortMatchedKeypointsInQualityOrder(vector<DMatch>& matches, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, vector<KeyPoint>& matchedkeypoints1, vector<KeyPoint>& matchedkeypoints2);

void getScaleAndRotation(const Mat& transfMat, double& scale, double& angle);

Mat FindTransformRansac(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<DMatch> matches, int iterations, int PerspectiveTransform);
vector<DMatch> excludeMany2OneMatches(const vector<DMatch>& matches, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2);
vector<DMatch> excludeOne2ManyMatches(const vector<DMatch>& matches, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2);
vector<DMatch> useNNratio(const vector<DMatch>&, double ratio);

int findMatch(DMatch m, vector<DMatch> allmatches, int crossCheck);

Mat AffineToHomography(Mat affine);
double calculateNewImageSquare(cv::Size OriginalSize, Mat transform);
vector<Point2f> getNewOutline_of_image(const Mat& image1, const Mat& Tr);
