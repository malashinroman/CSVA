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

#include <stdio.h>
#include <opencv2/core.hpp>

#define max_(a, b) ((a) > (b) ? (a) : (b))
#define min_(a, b) ((a) < (b) ? (a) : (b))
#define image_diag(image) sqrt((double)(image.size().width*image.size().width + image.size().height*image.size().height))


template <typename T> std::string tostr(const T& t) { std::ostringstream os; os<<t; return os.str(); };
void decomposeAff(const cv::Mat& transfMat, cv::Mat& Rot, cv::Mat& Shear, cv::Mat& Scale, double& Theta, double& shiftX, double& shiftY, double& scale, double& p, double& r);

void decomposeAffLutsiv(const cv::Mat& transfMat, double* scale, double* theta, double* ascale, double* direction);

cv::Rect getImageProjBbx(const cv::Mat& image1, const cv::Mat& trM);
cv::Point2f WrapTransform(cv::Point2f SamplePoint, const cv::Mat& trMatrix);

std::vector<cv::Point2f> WrapTransform(std::vector<cv::Point2f>& SamplePoint, const cv::Mat& trMatrix);

double euclideanDistacne(cv::Point2f p1, cv::Point2f p2);

double  convertOpencvAngle2GoodAngle(double angle_opencv);

cv::Point2f  predictModelPosition(const cv::KeyPoint& point1, const cv::KeyPoint& point2, cv::Point2f ModelPoint);

cv::Point2f  predictModelPosition2(const cv::KeyPoint& point1, const cv::KeyPoint& point2, cv::Point2f ModelPoint);

int independentMatches(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, cv::Mat im1, cv::Mat im2);

bool checkMatchIn(std::vector<cv::DMatch> matches, cv::DMatch newm);
bool checkMatchIn(std::vector<cv::DMatch> matches, cv::DMatch newm, int& indx);
double matchDistance(const cv::DMatch& m1, const cv::DMatch& m2, const std::vector<cv::KeyPoint>& pts1, const std::vector<cv::KeyPoint>& pts2);

double getMutualAngle(const cv::KeyPoint& p1, const cv::KeyPoint& p2);
double getAngleDif(double angle1, double angle2);
double getMutualScale(const cv::KeyPoint& p1, const cv::KeyPoint& p2);
void inline getMutualShifts(const cv::KeyPoint& p1, const cv::KeyPoint& p2, double& shiftx, double& shifty);


void getMatchedKeypoints(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>&  keypoints2,
    std::vector<cv::KeyPoint>& matchedkeypoints1, std::vector<cv::KeyPoint>& matchedkeypoints2);

void sortMatchedKeypointsInQualityOrder(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::KeyPoint>& matchedkeypoints1, std::vector<cv::KeyPoint>& matchedkeypoints2);

void getScaleAndRotation(const cv::Mat& transfMat, double& scale, double& angle);

cv::Mat FindTransformRansac(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::DMatch> matches, int iterations, int PerspectiveTransform);
std::vector<cv::DMatch> excludeMany2OneMatches(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2);
std::vector<cv::DMatch> excludeOne2ManyMatches(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2);
std::vector<cv::DMatch> useNNratio(const std::vector<cv::DMatch>&, double ratio);

int findMatch(cv::DMatch m, std::vector<cv::DMatch> allmatches, int crossCheck);

cv::Mat AffineToHomography(cv::Mat affine);
double calculateNewImageSquare(cv::Size OriginalSize, cv::Mat transform);
std::vector<cv::Point2f> getNewOutline_of_image(const cv::Mat& image1, const cv::Mat& Tr);
