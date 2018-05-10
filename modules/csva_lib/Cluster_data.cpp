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

#include "Cluster_data.h"
#include "misc_functions.h"

#include <opencv2/videostab.hpp>

#include "matching_hough.h"
#include <unordered_set>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>
using namespace cv;


Cluster_data::Cluster_data(const Cluster_data& other)
{
	this->matches = other.matches;
	this->init_size = other.init_size;
	this->position = other.position;
	this->orBin = other.orBin;
	this->scaleBin = other.scaleBin;
	this->XBin = other.XBin;
	this->YBin = other.YBin;
	this->transfMat = this->transfMat.clone();
}
void Cluster_data::exclude(DMatch match)
{
	for (vector<DMatch>::iterator it = this->matches.begin(); it != this->matches.end(); it++)
	{
		DMatch cmatch = *(it);
		int query = cmatch.queryIdx;
		int train = cmatch.trainIdx;
		if ((query == match.queryIdx) && (train = match.trainIdx))
		{
			//vector<DMatch>::iterator toRemove = it;
			//it--;
			this->matches.erase(it);
			break;
		}
	}
}
Cluster_data Cluster_data::operator=(const Cluster_data& other)
{
	this->matches = other.matches;
	//this->level = other.level;
	this->init_size = other.init_size;
	//this->MeanDist = other.MeanDist;
	this->position = other.position;
	this->orBin = other.orBin;
	this->scaleBin = other.scaleBin;
	this->XBin = other.XBin;
	this->YBin = other.YBin;
	this->transfMat = this->transfMat.clone();
	return *this;
}

bool compare(Cluster_data a, Cluster_data b)
{
	return (a.matches.size() > b.matches.size());
}

Cluster_data::~Cluster_data()
{

}

Point2f Cluster_data::ClusterPosition(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2)
{
	double meanx = 0;
	double meany = 0;
	for(vector<DMatch>::iterator match = this->matches.begin(); match != this->matches.end(); match++)
	{
		int qi = match->queryIdx;
		int ti = match->trainIdx;

		meanx += keypoints1.at(qi).pt.x;
		meany += keypoints1.at(qi).pt.y;
	}

	meanx /= this->matches.size();
	meany /= this->matches.size();
	this->position = Point2d(meanx, meany);
	return Point2d(meanx, meany);
}

#define MAX_COMB 400

int comb(int m, int n)
{
	static int mat[MAX_COMB][MAX_COMB];
	int i, j;
	if (n > m) return 0;
	if ((n == 0) || (m == n)) return 1;
	for (j = 0; j < n; j++)
	{
		mat[0][j] = 1;
		if (j == 0)
		{
			for (i = 1; i <= m - n; i++) mat[i][j] = i + 1;
		}
		else
		{
			for (i = 1; i <= m - n; i++) mat[i][j] = mat[i - 1][j] + mat[i][j - 1];
		}
	}
	return (mat[m - n][n - 1]);
}

Mat Cluster_data::fitModelParamsSimilarityRansac(const vector<KeyPoint>& keypoints1, 
	const vector<KeyPoint>& keypoints2, 
	const Mat& image1, const Mat& image2,
	int iterations, int hip_check, double model_distT, double AngleThresh, double ScaleThresh)
{	
	int m = matches.size();
	if (m < 3)
	{
		return Mat();
	}
	double scThresh = ScaleThresh;
	double angleThresh = AngleThresh;
	vector<KeyPoint> kpts1, kpts2;

	sortMatchedKeypointsInQualityOrder(matches, keypoints1, keypoints2, kpts1, kpts2);
	
	int best_inliers_num = 0;
	Mat bestTransf;
	int hypoCheck = 0;
	vector<DMatch> matches_passed;
	int ind1 = 0;
	int ind2 = 1;
	int useProScac = 0;
	int extended_output = 0;
	int n_comb = iterations + 1;
	if (matches.size() < MAX_COMB)
	{
		n_comb = comb(matches.size(), 2);
	}
	if (n_comb < iterations)
	{
		iterations = n_comb;
		useProScac = true;
	}
	for(int i = 0; i < iterations; i++)
	{
		Mat transfMat(2, 3, CV_64F);

		int ind[2];
		if (useProScac)
		{
			ind[0] = ind1;
			ind[1] = ind2;
			ind2++;
			if(ind2 == m)
			{
				ind1++;
				ind2 = ind1 + 1;
				if(ind2 == m)
				{
					break;
				}
			}
		}
		else
		{
			for (int j = 0; j < 2; j++)
			{
				ind[j] = rand() % matches.size();
				for (int k = j - 1; k > -1; k--)
				{
					if (ind[j] == ind[k])
					{
						j--;
						break;
					}
				}
			}
		}
		assert(ind[0] != ind[1]);
		KeyPoint kp11, kp12, kp21, kp22, tkp11, tkp22;
		kp11 = kpts1.at(ind[0]);
		kp12 = kpts1.at(ind[1]);
		kp21 = kpts2.at(ind[0]);
		kp22 = kpts2.at(ind[1]);
		Mat out;
		double x11, x12, x21, x22, y11, y12, y21, y22;
		x11 = kp11.pt.x;
		x12 = kp12.pt.x;
		x21 = kp21.pt.x;
		x22 = kp22.pt.x;

		y11 = kp11.pt.y;
		y12 = kp12.pt.y;
		y21 = kp21.pt.y;
		y22 = kp22.pt.y;
		double mscale1 = getMutualScale(kp11, kp21);
		double mscale2 = getMutualScale(kp12, kp22);

		double mangle1 = getMutualAngle(kp11, kp21);
		double mangle2 = getMutualAngle(kp12, kp22);
		double mangleDev = getAngleDif(mangle1, mangle2);
		double scRatio = mscale1 > mscale2 ? mscale1 / mscale2 : mscale2 / mscale1;

		double a = (x11 - x12)*(x11 - x12) + (y11 - y12)*(y11 - y12);
		double b = (x21 - x22)*(x21 - x22) + (y21 - y22)*(y21 - y22);
		if (a == 0)
			continue;
		double S = sqrt(b / a);

		double dx1 = x11 - x12;
		double dy1 = y11 - y12;
		double dx2 = x21 - x22;
		double dy2 = y21 - y22;
		double theta2 = atan2(-dy2, dx2);
		double theta1 = atan2(-dy1, dx1);
		double theta = (theta2 - theta1);
		double c = (y21 - y22)*(x11 - x12) - (x21 - x22)*(y11 - y12);
		double d = (x11 - x12)*(x21 - x22) + (y21 - y22)*(y11 - y12);
		if (theta < -CV_PI)
			theta = 2*CV_PI + theta;
		if (theta > 180)
			theta = theta - 2*CV_PI;

		theta2 = atan2(c, d);
		double theta3 = atan(c / d);
		double tx = x21- (S*x11*cos(theta2) - S*y11*sin(theta2));
		double ty = y21 - (S*x11*sin(theta2) + S*y11*cos(theta2));

		double tx2 = x22 - (S*x12*cos(theta2) - S*y12*sin(theta2));
		double ty2 = y22 - (S*x12*sin(theta2) + S*y12*cos(theta2));

		double Theta = theta / CV_PI * 180;

		double ThetaDev1 = getAngleDif(Theta, mangle1);
		double ThetaDev2 = getAngleDif(Theta, mangle2);

		double scRatio1 = S > mscale1 ? S / mscale1 : mscale1 / S;
		double scRatio2 = S > mscale2 ? S / mscale2 : mscale2 / S;

		Point2f modelPoint = Point2f(float(image1.size().width) / 2.f, float(image1.size().height) / 2.f);
		Point2f modelp1 = predictModelPosition(kp11, kp21, modelPoint);
		Point2f modelp2 = predictModelPosition(kp12, kp22, modelPoint);
		double dist = euclideanDistacne(modelp1, modelp2);

		if(scRatio > scThresh)
		{
			continue;
		}

		if (fabs(mangleDev) > angleThresh)
		{
			continue;
		}

		if (fabs(ThetaDev1) > angleThresh || fabs(ThetaDev2) > angleThresh)
		{
			continue;
		}

		if ((scRatio1 > scThresh) || (scRatio2 > scThresh))
		{
			continue;
		}

		int maxres = image2.size().width > image2.size().height ? image2.size().width : image2.size().height;

		double biggestIndivScale = mscale1 > mscale2 ? mscale1 : mscale2;
		double distThresh = maxres * biggestIndivScale * model_distT + maxres*0.01;

		if(dist > distThresh)
		{
			continue;
		}
		double inlier_dist = maxres * S * model_distT + maxres*0.01;
		vector<DMatch> goodmatches;
		int inliers_num = 0;

		transfMat.at<double>(0, 0) = S*cos(theta2);
		transfMat.at<double>(0, 1) = -S*sin(theta2);
		transfMat.at<double>(1, 0) = S*sin(theta2);
		transfMat.at<double>(1, 1) = S*cos(theta2);
		transfMat.at<double>(0, 2) = tx;
		transfMat.at<double>(1, 2) = ty;

		for(size_t j = 0; j < kpts1.size(); j++)
		{
			KeyPoint kp1 = kpts1.at(j);
			KeyPoint kp2 = kpts2.at(j);

			Mat p(3, 1, CV_64F);
			p.at<double>(0) = kp1.pt.x;
			p.at<double>(1) = kp1.pt.y;
			p.at<double>(2) = 1.;

			Mat p_ = transfMat*p;
			Point2f pointProj = Point2f(float(p_.at<double>(0)), float(p_.at<double>(1)));
			double dist = euclideanDistacne(pointProj, kp2.pt);

			double mangle1 = getMutualAngle(kp1, kp2);
			double ThetaDev1 = getAngleDif(Theta, mangle1);

			double scRatio1 = S > mscale1 ? S / mscale1 : mscale1 / S;

			if(dist > inlier_dist)
			{
				continue;
			}
			if(scRatio1 > scThresh)
			{
				continue;
			}
			if(fabs(ThetaDev1) > angleThresh)
			{
				continue;
			}
			goodmatches.push_back(matches.at(j));
			inliers_num++;
		}
		if(best_inliers_num < inliers_num)
		{
			matches_passed = goodmatches;
			best_inliers_num = inliers_num;
			bestTransf = transfMat.clone();
		}
		//break;
		hypoCheck++;
		if(hypoCheck > hip_check)
		{
			break;
		}
	}
	double deleteThresh = 0.01;
	if(matches_passed.size() > 0)
	{
		double in_out_ratio = (double)(matches_passed.size()) / matches.size();
		this->matches = matches_passed;
		if (in_out_ratio < deleteThresh)
		{
			this->matches = vector<DMatch>();
		}
	}
	else
	{
		this->matches = vector<DMatch>();
	}
	this->transfMat = bestTransf;
	return this->transfMat;
}

Mat Cluster_data::fitModelParams(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, 
	TransformType transfType, int extended_output, 
	Mat image1, Mat image2, 
	double inlier_dist, 
	int RANSAC_iter, int hip_check, double deleteThresh)
{
	Mat solution;
	int s = this->matches.size();
	if (s < 3)
	{
		return Mat();
	}
	vector<DMatch>::iterator it;
	int n =0;
	int i =0;

	Mat A;
	Mat B;
	if(transfType == TransformType::AFFINE_TRANSFORM)
	{
		A = Mat(s*2, 6, CV_64F, 0.);
		B = Mat(s*2, 1, CV_64F, 0.);
		for(it = this->matches.begin(); it != this->matches.end(); it++)
		{
			Point2f p1 = keypoints1.at(it->queryIdx).pt;
			Point2f p2 = keypoints2.at(it->trainIdx).pt;
			double line1[6] = {p1.x, p1.y, 0, 0, 1, 0};
			double line2[6] = {0, 0, p1.x, p1.y, 0, 1};
			Mat a1(1, 6, CV_64F, &line1);
			Mat a2(1, 6, CV_64F, &line2);

			a1.row(0).copyTo(A.row(i));
			B.at<double>(i) = (p2.x);
			i++;
			a2.row(0).copyTo(A.row(i));
			B.at<double>(i) = (p2.y);
			i++;
		}
	}
	if(transfType == TransformType::SIMILARITY_TRANSFORM)
	{

		A = Mat(s*2, 4, CV_64F, 0.);
		B = Mat(s*2, 1, CV_64F, 0.);
		for(it = this->matches.begin(); it != this->matches.end(); it++)
		{
			Point2f p1 = keypoints1.at(it->queryIdx).pt;
			Point2f p2 = keypoints2.at(it->trainIdx).pt;
			double line1[4] = {p1.x, -p1.y, 1, 0};
			double line2[4] = {p1.y, p1.x, 0, 1};

			Mat a1(1, 4, CV_64F, &line1);
			Mat a2(1, 4, CV_64F, &line2);

			a1.row(0).copyTo(A.row(i));
			B.at<double>(i) = (p2.x);
			i++;
			a2.row(0).copyTo(A.row(i));
			B.at<double>(i) = (p2.y);
			i++;
		}
	}

	if(transfType == TransformType::AFFINE_TRANSFORM 
		|| transfType == TransformType::SIMILARITY_TRANSFORM)
	{
		///Both solutions give the same result
		solve(A, B, solution, DECOMP_SVD);
		//Mat solution2 = (A.t()*A).inv()*(A.t()*B);
	}
	transfMat = Mat(2,3, CV_64F);

	if(transfType == TransformType::AFFINE_TRANSFORM)
	{
		transfMat.at<double>(0,0) = solution.at<double>(0);
		transfMat.at<double>(0,1) = solution.at<double>(1);
		transfMat.at<double>(1,0) = solution.at<double>(2);
		transfMat.at<double>(1,1) = solution.at<double>(3);
		transfMat.at<double>(0,2) = solution.at<double>(4);
		transfMat.at<double>(1,2) = solution.at<double>(5);
	}
	if(transfType == TransformType::SIMILARITY_TRANSFORM)
	{
		transfMat.at<double>(0,0) = solution.at<double>(0);
		transfMat.at<double>(0,1) = -solution.at<double>(1);
		transfMat.at<double>(1,0) = solution.at<double>(1);
		transfMat.at<double>(1,1) = solution.at<double>(0);

		transfMat.at<double>(0,2) = solution.at<double>(2);
		transfMat.at<double>(1,2) = solution.at<double>(3);
	}
	
	if(transfType == TransformType::OPENCV_AFFINE)
	{
		vector<KeyPoint> kpts1;
		vector<KeyPoint> kpts2;
		getMatchedKeypoints(this->matches, keypoints1, keypoints2, kpts1, kpts2);
		vector<Point2f> points2D1;
		vector<Point2f> points2D2;
		Mat P1 = Mat(2, this->matches.size(), CV_64F, 0.);
		Mat P2 = Mat(2, this->matches.size(), CV_64F, 0.);
		for(size_t i = 0; i < kpts1.size(); i++)
		{
			P1.at<double>(0,i) = kpts1.at(i).pt.x;
			P1.at<double>(1,i) = kpts1.at(i).pt.y;
			P2.at<double>(0,i) = kpts2.at(i).pt.x;
			P2.at<double>(1,i) = kpts2.at(i).pt.y;
			points2D1.push_back(kpts1.at(i).pt);
			points2D2.push_back(kpts2.at(i).pt);
		}
		transfMat = estimateRigidTransform(points2D1, points2D2, true);
	}
	if(transfType == TransformType::OPENCV_SIMILARITY)
	{
		vector<KeyPoint> kpts1;
		vector<KeyPoint> kpts2;
		getMatchedKeypoints(this->matches, keypoints1, keypoints2, kpts1, kpts2);
		vector<Point2f> points2D1;
		vector<Point2f> points2D2;
		Mat P1 = Mat(this->matches.size(), 2, CV_64F, 0.);
		Mat P2 = Mat(this->matches.size(), 2, CV_64F, 0.);
		for(size_t i = 0; i < kpts1.size(); i++)
		{
			P1.at<double>(i,0) = kpts1.at(i).pt.x;
			P1.at<double>(i,1) = kpts1.at(i).pt.y;
			P2.at<double>(i,0) = kpts2.at(i).pt.x;
			P2.at<double>(i,1) = kpts2.at(i).pt.y;
			points2D1.push_back(kpts1.at(i).pt);
			points2D2.push_back(kpts2.at(i).pt);
		}
		transfMat = estimateRigidTransform(points2D1, points2D2, false);
	}

	

	if(transfType == TransformType::SIMILARITY_RANSAC)
	{
		int m = matches.size();
		int comb[47] = {0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035};
		//printf("SIMILARITY RANSAC\n");
		int iterations = 100;
		int bruteForce = 0;
		if(RANSAC_iter!=0)
		{
			iterations = RANSAC_iter;
		}
		if(m < 46)
		{
			if(comb[m] < iterations)
			{
				/*iterations = comb[m];
				bruteForce = 1;*/
				//extended_output = 1;
			}
		}
		if(fabs(inlier_dist) < 0.0001)
		{
			inlier_dist = 60;
		}
		double scThresh = sqrt(2.);
		double angleThresh = 10; //FIXME 15 - gives much more false_positive matches
		vector<KeyPoint> kpts1, kpts2;
		getMatchedKeypoints(matches, keypoints1, keypoints2, kpts1, kpts2);
		int best_inliers_num = 0;
		Mat bestTransf;
		int hypoCheck = 0;
		vector<DMatch> matches_passed;
		int ind1 =0;
		int ind2 = 1;
		for(int i = 0; i < iterations; i++)
		{
			int inliers_num = 0;
			int ind[4];
			int PointSetSize = 2;
			if(bruteForce)
			{
				ind[0] = ind1;
				ind[1] = ind2;
				ind2++;
				if(ind2 == m)
				{
					ind1++;
					ind2 = ind1+1;
				}
			}
			else
			{
				for(int j =0; j < PointSetSize; j++)
				{
					ind[j] = rand() % matches.size();
					for(int k = j - 1; k > -1; k--)
					{
						if(ind[j] == ind[k])
						{
							j--;
							break;
						}
					}
				}
			}
			KeyPoint kp11, kp12, kp21, kp22, tkp11, tkp22;
			kp11 = kpts1.at(ind[0]);
			kp12 = kpts1.at(ind[1]);
			kp21 = kpts2.at(ind[0]);
			kp22 = kpts2.at(ind[1]);
			
			Mat out;
			
			double x11, x12, x21, x22, y11, y12, y21, y22;
			x11 = kp11.pt.x;
			x12 = kp12.pt.x;
			x21 = kp21.pt.x;
			x22 = kp22.pt.x;

			y11 = kp11.pt.y;
			y12 = kp12.pt.y;
			y21 = kp21.pt.y;
			y22 = kp22.pt.y;
			double mscale1 = getMutualScale(kp11, kp21);
			double mscale2 = getMutualScale(kp12, kp22);

			double mangle1 = getMutualAngle(kp11, kp21);
			double mangle2 = getMutualAngle(kp12, kp22);
			
			double scRatio = mscale1 > mscale2 ? mscale1 / mscale2 : mscale2 / mscale1;

			if(scRatio > scThresh)
			{
				continue;
			}

			if(fabs(mangle1 - mangle2) > angleThresh)
			{
				continue;
			}

			double a = (x11 - x12)*(x11 - x12) + (y11 - y12)*(y11 - y12);
			double b = (x21 - x22)*(x21 - x22) + (y21 - y22)*(y21 - y22);
			double S = sqrt(b/a);
			double scRatio1 = S > mscale1 ? S / mscale1 : mscale1 / S;
			double scRatio2 = S > mscale2 ? S / mscale2 : mscale2 / S;

			if((scRatio1 > scThresh) || (scRatio2 > scThresh))
			{
				continue;
			}
			double dx1 = x11-x12;
			double dy1 = y11-y12;
			double dx2 = x21-x22;
			double dy2 = y21-y22;
			double Theta2 = atan2(-dy2, dx2) / CV_PI * 180;
			double Theta1 = atan2(-dy1, dx1) / CV_PI * 180;
			double Theta = (Theta2 - Theta1);
			if(Theta < -180)
				Theta = 360 - Theta;
			if(Theta > 180)
				Theta = Theta - 360;

			if(fabs(Theta - mangle1) > angleThresh || fabs(Theta - mangle2) > angleThresh)
			{
				continue;
			}

			Point2f modelPoint = Point2f(image1.size().width / 2.f, image1.size().height / 2.f);
			Point2f modelp1 = predictModelPosition(kp11, kp21, modelPoint);
			Point2f modelp2 = predictModelPosition(kp12, kp22, modelPoint);
			double dist = euclideanDistacne(modelp1, modelp2);
			int maxres = int(0.8 * sqrt(1.*image2.size().width*image2.size().width + image2.size().height*image2.size().height));
			double distThresh = maxres / 8;
			if(dist > distThresh)
			{
				continue;
			}

			Theta = - Theta / 180. * CV_PI;
			double tx = x22 - (S*x12*cos(Theta) - S*y12*sin(Theta));
			double ty = y22 - (S*x12*sin(Theta) + S*y12*cos(Theta));

			double tx2 = x21 - (S*x11*cos(Theta) - S*y11*sin(Theta));
			double ty2 = y21 - (S*x11*sin(Theta) + S*y11*cos(Theta));


			transfMat.at<double>(0,0) = S*cos(Theta);
			transfMat.at<double>(0,1) = -S*sin(Theta);
			transfMat.at<double>(1,0) = S*sin(Theta);
			transfMat.at<double>(1,1) = S*cos(Theta);
			transfMat.at<double>(0,2) = tx;
			transfMat.at<double>(1,2) = ty;

			vector<DMatch> goodmatches;
			for(size_t j = 0; j < kpts1.size(); j++)
			{

				Mat p(3,1, CV_64F);
				p.at<double>(0) = kpts1.at(j).pt.x;
				p.at<double>(1) = kpts1.at(j).pt.y;
				p.at<double>(2) = 1;
				Mat p_ = transfMat*p;
				Point2f pointProj = Point2f(float(p_.at<double>(0)), float(p_.at<double>(1)));
				double dist = euclideanDistacne(pointProj, kpts2.at(j).pt);

				if(dist < inlier_dist)
				{
					double S, Theta;
					getScaleAndRotation(transfMat, S, Theta);
					double mangle1 = getMutualAngle(kpts1.at(j), kpts2.at(j));
					if(fabs(Theta - mangle1) > angleThresh)
					{
						continue;
					}
					double mscale1 = getMutualScale(kpts1.at(j), kpts2.at(j));
					double scRatio1 = S > mscale1 ? S / mscale1 : mscale1 / S;
					if((scRatio1 > scThresh))
					{
						continue;
					}
					goodmatches.push_back(matches.at(j));
					inliers_num++;
				}
			}

			if(best_inliers_num < inliers_num)
			{
				matches_passed = goodmatches;
				best_inliers_num = inliers_num;
				bestTransf = transfMat.clone();
			}
			//break;
			hypoCheck++;
			if(hypoCheck > hip_check)
			{
				break;
			}
		}

		if(matches_passed.size() > 0)
		{
			double in_out_ratio = (double)(matches_passed.size()) / matches.size();
			this->matches = matches_passed;

			if(in_out_ratio < deleteThresh)
			{
				this->matches = vector<DMatch>();
			}
		}
		else
		{
			this->matches = vector<DMatch>();
		}
		this->transfMat = bestTransf;
	}
	
	if(transfType == TransformType::OPENCV_HOMOGRAPHY_LMEDS)
	{
		vector<KeyPoint> kpts1;
		vector<KeyPoint> kpts2;
		getMatchedKeypoints(this->matches, keypoints1, keypoints2, kpts1, kpts2);
		vector<Point2f> points2D1;
		vector<Point2f> points2D2;
		Mat P1 = Mat(this->matches.size(), 2, CV_64F, 0.);
		Mat P2 = Mat(this->matches.size(), 2, CV_64F, 0.);
		for(size_t i = 0; i < kpts1.size(); i++)
		{
			P1.at<double>(i,0) = kpts1.at(i).pt.x;
			P1.at<double>(i,1) = kpts1.at(i).pt.y;
			P2.at<double>(i,0) = kpts2.at(i).pt.x;
			P2.at<double>(i,1) = kpts2.at(i).pt.y;
			points2D1.push_back(kpts1.at(i).pt);
			points2D2.push_back(kpts2.at(i).pt);
		}
		if(points2D1.size() < 4)
		{
			transfMat = Mat();
		}
		else
		{
            double w = image1.size().width;
            double h = image2.size().height;
            double thresh = sqrt(w*w+h*h) * 0.05;
			transfMat = findHomography(points2D1, points2D2, CV_LMEDS);//, thresh);
		}
	}



	if(transfType == TransformType::OPENCV_HOMOGRAPHY_RANSAC)
	{
		vector<KeyPoint> kpts1;
		vector<KeyPoint> kpts2;
		getMatchedKeypoints(this->matches, keypoints1, keypoints2, kpts1, kpts2);
		vector<Point2f> points2D1;
		vector<Point2f> points2D2;
		Mat P1 = Mat(this->matches.size(), 2, CV_64F, 0.);
		Mat P2 = Mat(this->matches.size(), 2, CV_64F, 0.);
		for(size_t i = 0; i < kpts1.size(); i++)
		{
			P1.at<double>(i,0) = kpts1.at(i).pt.x;
			P1.at<double>(i,1) = kpts1.at(i).pt.y;
			P2.at<double>(i,0) = kpts2.at(i).pt.x;
			P2.at<double>(i,1) = kpts2.at(i).pt.y;
			points2D1.push_back(kpts1.at(i).pt);
			points2D2.push_back(kpts2.at(i).pt);
		}
		if(points2D1.size() < 4)
		{
			transfMat = Mat();
		}
		else
		{
            double w = image1.size().width;
            double h = image2.size().height;
            double inlier_dist =image_diag(image2)*0.01;

            //double thresh = sqrt(w*w+h*h) * 0.05;
			transfMat = findHomography(points2D1, points2D2, CV_RANSAC, inlier_dist);
		}
	}



	//if(transfType == EPIPOLAR_CONSTRAINT)
	//{
	//		double diag = sqrt((double)(image1.size().width * image1.size().width + image1.size().height * image1.size().height));
	//		double dist = diag*0.03;
	//		this->matches = EpipolarConstraint(keypoints1, keypoints2, matches, dist);
	//		transfMat = Mat();
	//}
    
	return transfMat;
}

vector<DMatch> invertMatches(vector<DMatch> matches)
{
	vector<DMatch> invMatches;
	for(size_t i = 0; i < matches.size(); i++)
	{
		DMatch m = matches.at(i);
		DMatch im;
		im.distance = m.distance;
		im.imgIdx = m.imgIdx;
		im.queryIdx = m.trainIdx;
		im.trainIdx = m.queryIdx;
		invMatches.push_back(im);
	}
	return invMatches;
}

vector<DMatch> Cluster_data::eliminateOutliers(vector<KeyPoint> points1, vector<KeyPoint> points2, double distThreshPercent, double distThreshModelPercent, double rotationThresh, double ScaleThresh, Mat image1, Mat image2, int extendedOutPut, int inv)
{
	if(this->transfMat.size().width == 0)
	{
		vector<DMatch> el = this->matches;
		this->matches = vector<DMatch>();
		return el;
	}
	Mat trMat;
	if(!inv)
	{
		trMat = this->transfMat;
	}
	else
	{
		invertAffineTransform(this->transfMat, trMat);
	}
	Mat rot, shear, scale;
	double angle, Kscale, shiftX, shiftY;
	double p = 0, r = 0;
	decomposeAff(trMat, rot, shear, scale , angle, shiftX, shiftY, Kscale, p, r);
	
	distThreshModelPercent *= Kscale;
	distThreshPercent *= Kscale;
	double distanceMean = 0.;
	int numberOfeliminatedPoints = 0;

	vector<DMatch> eliminatedMatches;
	vector<DMatch> clusterMatches;
	vector<DMatch> afterElimination;
	if(!inv)
	{
		clusterMatches = this->matches;
	}
	else
	{
		clusterMatches = invertMatches(this->matches);
	}
	for(unsigned int i = 0; i < clusterMatches.size(); i ++)
	{
		int indx1 = clusterMatches.at(i).queryIdx;
		int indx2 = clusterMatches.at(i).trainIdx;
		KeyPoint keyp1 = points1.at(indx1);
		KeyPoint keyp2 = points2.at(indx2);

		double MatchRotation = getMutualAngle(keyp1, keyp2);
		double MatchScale = getMutualScale(keyp1, keyp2);
		Point2f p1 = keyp1.pt;
		Point2f p2 = keyp2.pt;
		Point2f clusterPos = this->ClusterPosition(points1, points2);
		Point2f ModelPoint = (clusterPos + keyp1.pt);
		ModelPoint.x /= 2;
		ModelPoint.y /= 2;
		Point2f modelPosition_byPoint = predictModelPosition(keyp1, keyp2, ModelPoint);
		Point2f modelPosition_byAff = WrapTransform(ModelPoint, trMat);
		Point2f p1_ = WrapTransform(p1, trMat);

		double distanceProj = euclideanDistacne(p1_, p2);
		double distanceModel = euclideanDistacne(modelPosition_byPoint, modelPosition_byAff);

		double angleDev = fabs(MatchRotation - angle);
		angleDev = angleDev > 180 ? 360 - angleDev: angleDev;
		double Proportion = max_(Kscale, MatchScale) / min_(Kscale, MatchScale);

		double Model2PointDist = euclideanDistacne(modelPosition_byAff, p1_);

		vector<Point2f> wrapped_points = getNewOutline_of_image(image1, trMat);
		double d1 = euclideanDistacne(wrapped_points.at(0), wrapped_points.at(2));
		double d2 = euclideanDistacne(wrapped_points.at(1), wrapped_points.at(3));
		double max_diag = d1 > d2 ? d1 : d2;
		double widdd = image1.size().width;
		double dist_thresh = max_diag; //*0.66

		//!FIXME: distThreshProjection shouldn't(?) depend on distance to ModelPoint; Better to rely on distance btw point and clusterCenter
		double distThreshProjection = distThreshPercent * Model2PointDist + dist_thresh / 90; //FIXME: 40 gives more
		double distThreshModel = distThreshModelPercent * Model2PointDist + dist_thresh / 90;

		if( (Proportion > ScaleThresh)  || (angleDev > rotationThresh) || (distanceProj> distThreshProjection) || (distanceModel > distThreshModel))
		{
			eliminatedMatches.push_back(clusterMatches.at(i));
			numberOfeliminatedPoints++;
		}
		else
		{
			afterElimination.push_back(clusterMatches.at(i));
		}

	}

	if((double)(clusterMatches.size()) / this->init_size < 0.05)
	{
		clusterMatches = vector<DMatch>();
	}
	this->matches = afterElimination;
	return eliminatedMatches;
}
