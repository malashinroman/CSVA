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
#include <stdio.h>
#include <stdlib.h>


using namespace std;
using namespace cv;
std::array<double, 6> calculateConfidence(const Mat& PT, vector<DMatch>& matches, const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const Mat& im1, const Mat& im2,
	const vector<DMatch>& excludedMatches, int type, double LoweProb = 0.008);
double calculateConfidenceLowe(Mat im1, Mat im2, const vector<KeyPoint>& kpts1, const vector<KeyPoint>&  kpts2,
	const vector<DMatch>& matches, Mat H, double p,
	const vector<DMatch>& excludedMatches, int& n, int& k);
double calcProbOfSuccess(int n, int k, double p);
double calculateAverageProbabilityOfMatches(const vector<DMatch>& matches);
double calculateProbOfrandomPickQualityMatches(vector<DMatch> all_matches, vector<DMatch> pickMatches, bool biggerIsGood);