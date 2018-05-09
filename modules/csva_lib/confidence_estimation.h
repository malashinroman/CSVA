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