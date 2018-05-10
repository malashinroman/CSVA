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


#include "opencv2/core.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matching_hough.h"
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
//#include "statistics.h"
#include <omp.h>
#include <opencv2/videostab/global_motion.hpp>
#include <opencv2/videostab/stabilizer.hpp>
using namespace cv;
using namespace std;

/*
* ---------------------------
* Core Functions
* ---------------------------
*/
Hough_Transform::Hough_Transform(double BinOrSize, double BinXSize, double BinYSize, double BinScFactor, Mat image1, Mat image2/*, HoughAccHashType hashtype*/)
{
	this->image1 = image1.clone();
	this->image2 = image2.clone();

	this->ImWidth = this->image1.size().width;
	this->ImHeight = this->image1.size().height;


	this->BinOrSize = BinOrSize;
	this->NumOrBin = (int)(360 / BinOrSize);
	double maxSize = 32.;
	this->NumScaleBin = int(log(maxSize) / log(BinScFactor) * 2 + 1);
	this->BinScFactor = BinScFactor;
	this->BinXSize = BinXSize; //4 - we should have positive and negative position of the model aswell
	this->BinYSize = BinYSize; // it's not an exact value, we assume square map for the model
}

Hough_Transform::~Hough_Transform()
{
	//delete this->houghList;
	//switch (hashtype)
	//{
	//case HashCpp11: {break;}
	////case HashManual: {delete this->houghList; break;}
	////case HashManualND: {delete this->houghListND; break;}
	//default: {printf("unknown hash type\n");  break;}
	//}
}
void Hough_Transform::excludeDuplicates()
{
	for (vector<Cluster_data>::iterator itc = this->clusters.begin(); this->clusters.end() != itc; itc++)
	{
		for (vector<DMatch>::iterator itm = itc->matches.begin(); itm != itc->matches.end(); itm++)
		{
			for (vector<Cluster_data>::iterator itc_next = itc + 1; this->clusters.end() != itc_next; itc_next++)
			{
				itc_next->exclude(*(itm));
			}
		}
	}
}

void Hough_Transform::removeSmallClusters(int threshold, bool independent)
{
	for (unsigned int i = 0; i < this->clusters.size(); i++)
	{
		int indmatches = independent ? independentMatches(this->clusters.at(i).matches, this->keypoints1, this->keypoints2, this->image1, this->image2) : this->clusters.at(i).matches.size();
		if (indmatches/*this->clusters.at(i).matches.size()*/ < threshold)
		{
			int size = this->clusters.at(i).matches.size();

			this->clusters.erase(this->clusters.begin() + i);
			i -= 1;
		}
	}
}

void Hough_Transform::ExcludeOne2ManyFromClusters()
{
	for (size_t i = 0; i < this->clusters.size(); i++)
	{
		this->clusters.at(i).matches = excludeOne2ManyMatches(this->clusters.at(i).matches, this->keypoints1, this->keypoints2);
	}
}

void Hough_Transform::ExcludeMany2OneFromClusters()
{
	for (size_t i = 0; i < this->clusters.size(); i++)
	{
		this->clusters.at(i).matches = excludeMany2OneMatches(this->clusters.at(i).matches, this->keypoints1, this->keypoints2);
	}
}
vector<DMatch> Hough_Transform::getAllClusterMatches()
{
	vector<DMatch> allmatches;
	for (vector<Cluster_data>::iterator itc = this->clusters.begin(); itc != this->clusters.end(); itc++)
	{
		for (vector<DMatch>::iterator itMatches = itc->matches.begin(); itMatches != itc->matches.end(); itMatches++)
		{
			if (!checkMatchIn(allmatches, *(itMatches)))
			{
				allmatches.push_back(*(itMatches));
			}
		}
	}
	return allmatches;
}

void Hough_Transform::FillAccNewBoundary2(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<DMatch> matches, int ShiftAcc)
{
	this->allmatches = matches;
	this->keypoints1 = points1;
	this->keypoints2 = points2;
	int numberOfPoint = 0;
	//int i;
	double deltaScale = ShiftAcc > 0 ? 0.5 : 0;//this->BinScFactor/2 : 0;

	double deltaOr = ShiftAcc > 0 ? this->BinOrSize / 2 : 0;
	//printf("Accumulating matches\n");
	for (size_t i = 0; i < matches.size(); i++)
	{
		numberOfPoint++;
		int indx1 = matches.at(i).queryIdx;
		int indx2 = matches.at(i).trainIdx;

		KeyPoint keyp1 = points1.at(indx1);
		KeyPoint keyp2 = points2.at(indx2);
		Point2d modelLoc;
		int bina = 0;
		int binx = 0;
		int biny = 0;
		int bins = 0;
		double ang1 = convertOpencvAngle2GoodAngle(keyp1.angle);
		double ang2 = convertOpencvAngle2GoodAngle(keyp2.angle);

		double ang = ang2 - ang1;
		ang = ang < 0. ? ang + 360 : ang;

		bina = ((int)((ang + deltaOr) / this->BinOrSize)) % this->NumOrBin;
		int bina0 = ((int)((ang) / this->BinOrSize)) % this->NumOrBin;

		double MutualScale = keyp2.size / keyp1.size;
		double ScaleBin = log(MutualScale) / log(this->BinScFactor) + deltaScale;
		bins = int(ScaleBin);

		// We assume model position in the center of the image
		Point2d im1C(ImWidth / 2, ImHeight / 2);
		modelLoc = predictModelPosition(keyp1, keyp2, im1C);


		double mscale = pow(2, bins);
		//float curBinXsize = this->BinXSize;
		//float curBinYsize = this->BinYSize;
		float curBinXsize = float(this->BinXSize / mscale);
		float curBinYsize = float(this->BinYSize / mscale);

		double deltaX = ShiftAcc > 0 ? curBinXsize / 2 : 0;
		double deltaY = ShiftAcc > 0 ? curBinYsize / 2 : 0;

		double shiftx = modelLoc.x + deltaX;
		double shifty = modelLoc.y + deltaY;
		binx = (int)(shiftx / curBinXsize);
		biny = (int)(shifty / curBinYsize);

		DMatch m = matches.at(i);
		houghListCpp.AddMatch(m, binx, biny, bina, bins);
	}
}

void Hough_Transform::FillAccNewBoundary(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<DMatch> matches, int ShiftAcc)
{
	this->allmatches = matches;
	this->keypoints1 = points1;
	this->keypoints2 = points2;
	int numberOfPoint = 0;
	//int i;
	double deltaScale = ShiftAcc > 0 ? 0.5 : 0;//this->BinScFactor/2 : 0;
	double deltaX = ShiftAcc > 0 ? this->BinXSize / 2 : 0;
	double deltaY = ShiftAcc > 0 ? this->BinYSize / 2 : 0;
	double deltaOr = ShiftAcc > 0 ? this->BinOrSize / 2 : 0;
	//printf("Accumulating matches\n");
	for (size_t i = 0; i < matches.size(); i++)
	{
		numberOfPoint++;
		int indx1 = matches.at(i).queryIdx;
		int indx2 = matches.at(i).trainIdx;

		KeyPoint keyp1 = points1.at(indx1);
		KeyPoint keyp2 = points2.at(indx2);
		Point2d modelLoc;
		int bina = 0;
		int binx = 0;
		int biny = 0;
		int bins = 0;
		double ang1 = convertOpencvAngle2GoodAngle(keyp1.angle);
		double ang2 = convertOpencvAngle2GoodAngle(keyp2.angle);

		double ang = ang2 - ang1;
		ang = ang < 0. ? ang + 360 : ang;

		bina = ((int)((ang + deltaOr) / this->BinOrSize)) % this->NumOrBin;
		int bina0 = ((int)((ang) / this->BinOrSize)) % this->NumOrBin;

		double MutualScale = keyp2.size / keyp1.size;
		/*
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			*/
		/*deltaScale = ShiftAcc > 0 ? this->BinScFactor / 2 : 0;
		double ScaleF = log(MutualScale) + deltaScale;
		bins = cvRound(ScaleF);*/
		double ScaleBin = log(MutualScale) / log(this->BinScFactor) + deltaScale;
		bins = int(ScaleBin);
		/*int bins0 =  cvRound(log(MutualScale)); */
		//double corr = cvRound(ScaleF) - ScaleF;
		// We assume model position in the center of the image
		Point2d im1C(ImWidth / 2, ImHeight / 2);
		modelLoc = predictModelPosition(keyp1, keyp2, im1C);
		double shiftx = modelLoc.x + deltaX;
		double shifty = modelLoc.y + deltaY;
		binx = (int)(shiftx / this->BinXSize);
		biny = (int)(shifty / this->BinYSize);
		//int binx0 = (int)(modelLoc.x / this->BinXSize);
		//int biny0 = (int)(modelLoc.y / this->BinXSize);
		DMatch m = matches.at(i);
		houghListCpp.AddMatch(m, binx, biny, bina, bins);
	}
}

void Hough_Transform::FillAcc(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<DMatch> matches)
{
	this->allmatches = matches;
	this->keypoints1 = points1;
	this->keypoints2 = points2;
	vector<DMatch>::iterator it;
	//Mat imagePosAcc = Mat::zeros(this->image1.size(), CV_8U);
	int numberOfPoint = 0;
	//int i;
	//printf("Accumulating matches\n");
	for (size_t i = 0; i < matches.size(); i++)
	{
		numberOfPoint++;

		int indx1 = matches.at(i).queryIdx;
		int indx2 = matches.at(i).trainIdx;

		KeyPoint keyp1 = points1.at(indx1);
		KeyPoint keyp2 = points2.at(indx2);
		Point2d modelLoc;
		int bina = 0, bina2 = 0;
		int binx = 0, binx2 = 0;
		int biny = 0, biny2 = 0;
		int bins = 0, bins2 = 0;

		int binx_s2 = 0, biny_s2 = 0;
		int binx2_s2 = 0, biny2_s2 = 0;

		double ang1 = convertOpencvAngle2GoodAngle(keyp1.angle);
		double ang2 = convertOpencvAngle2GoodAngle(keyp2.angle);

		double ang = ang2 - ang1;
		ang = ang < 0. ? ang + 360 : ang;

		bina = ((int)((ang) / this->BinOrSize)) % this->NumOrBin;

		double deltaA = (ang / this->BinOrSize - bina - 0.5);
		bina2 = deltaA > 0. ? bina + 1 : bina - 1;
		if (bina2 < 0)
		{
			bina2 += this->NumOrBin;
		}
		if (bina2 == this->NumOrBin)
		{
			bina2 = 0;
		}
		double MutualScale = keyp2.size / keyp1.size;

		double ScaleBin = log(MutualScale) / log(this->BinScFactor);
		bins = cvRound(ScaleBin);

		double corr = cvRound(ScaleBin) - ScaleBin;
		/*double ScaleF = log(MutualScale);
		bins = cvRound(ScaleF);
		double corr = cvRound(ScaleF) - ScaleF;*/
		bins2 = (corr > 0) ? bins - 1 : bins + 1;

		// We assume model position in the center of the image
		Point2d im1C(ImWidth / 2, ImHeight / 2);

		modelLoc = predictModelPosition(keyp1, keyp2, im1C);

		double shiftx = modelLoc.x;
		double shifty = modelLoc.y;

		double mscale = pow(this->BinScFactor, bins);

		float curBinXsize = float(this->BinXSize * mscale);
		float curBinYsize = float(this->BinYSize * mscale);

		//float curBinXsize = this->BinXSize;
		//float curBinYsize = this->BinYSize;


		binx = (int)(shiftx / curBinXsize); //
		biny = (int)(shifty / curBinYsize); //

		//binx = (int)(shiftx / ((this->BinXSize) / )); //
		//biny = (int)(shifty / ((this->BinYSize) / )); //

		//! vote in neighbour bin for current scale
		double deltaX = shiftx / curBinXsize - binx - 0.5;
		binx2 = deltaX > 0.0 ? binx + 1 : binx - 1;
		//deltaX = fabs(deltaX);
		double deltaY = shifty / curBinYsize - biny - 0.5;
		biny2 = deltaY > 0.0 ? biny + 1 : biny - 1;
		//deltaY = fabs(deltaY);


		/*we have different cells for different scale*/
		double mscale2 = pow(this->BinScFactor, bins2);
		/*float curBinXsize2 = this->BinXSize;
		float curBinYsize2 = this->BinYSize;*/
		/*float curBinXsize2 = this->BinXSize / mscale2;
		float curBinYsize2 = this->BinYSize / mscale2;*/

		float curBinXsize2 = float(this->BinXSize * mscale2);
		float curBinYsize2 = float(this->BinYSize * mscale2);


		binx_s2 = (int)(shiftx / curBinXsize2); //
		biny_s2 = (int)(shifty / curBinYsize2); //

		double deltaX_s2 = shiftx / curBinXsize2 - binx_s2 - 0.5;
		binx2_s2 = deltaX_s2 > 0.0 ? binx_s2 + 1 : binx_s2 - 1;
		//deltaX_s2 = fabs(deltaX_s2);

		double deltaY_s2 = shifty / curBinYsize2 - biny_s2 - 0.5;
		biny2_s2 = deltaY_s2 > 0.0 ? biny_s2 + 1 : biny_s2 - 1;
		//deltaY_s2 = fabs(deltaY_s2);


		int bina_[2];
		int bins_[2];
		int binx_[2];
		int biny_[2];

		int binx_s2_[2];
		int biny_s2_[2];

		bina_[0] = bina;
		bina_[1] = bina2;
		bins_[0] = bins;
		bins_[1] = bins2;
		binx_[0] = binx;
		binx_[1] = binx2;
		biny_[0] = biny;
		biny_[1] = biny2;

		binx_s2_[0] = binx_s2;
		biny_s2_[0] = biny_s2;
		binx_s2_[1] = binx2_s2;
		biny_s2_[1] = biny2_s2;

		for (int k = 0; k < 16; k++)
		{
			int n1 = k % 2;
			int n2 = (k >> 1) % 2;
			int n3 = (k >> 2) % 2;
			int n4 = (k >> 3) % 2;
			DMatch m = matches.at(i);
			int x = binx_[n4];
			int y = biny_[n3];
			int a = bina_[n1];
			int s = bins_[n2];
			if (n2 == 1)
			{
				x = binx_s2_[n4];
				y = biny_s2_[n3];
			}
			houghListCpp.AddMatch(m, x, y, a, s);
		}
	}

}

Cluster_data Hough_Transform::MaxCluster()
{
	int maxvotes = 0;
	Cluster_data MaxCluster;
	for (auto kv : houghListCpp.hash_table)
	{
		float ScFactor = 1;
		if (kv.first[3] < 0)
		{
			ScFactor = float(pow(2, -kv.first[3]));
		}
		//!int curvotes = kv.second.size() * ScFactor;
		int curvotes = kv.second.size();
		if (curvotes >= maxvotes)
		{
			MaxCluster.matches.clear();
			MaxCluster.matches.insert(MaxCluster.matches.begin(), kv.second.begin(), kv.second.end());
			maxvotes = curvotes;
		}
	}
	MaxCluster.init_size = MaxCluster.matches.size();
	return MaxCluster;
}

void Hough_Transform::FindClusters(int voteThresh)
{
	int sumVotes = 0;
	for (auto kv : houghListCpp.hash_table)
	{
		if (kv.second.size() >= (unsigned int)voteThresh)
		{
			Cluster_data NewCData;
			int x = kv.first[0];
			int y = kv.first[1];
			int orbin = kv.first[2];
			int scbin = kv.first[3];
			NewCData.XBin = x;
			NewCData.YBin = y;
			NewCData.orBin = orbin;
			NewCData.scaleBin = scbin;
			//NewCData.matches.insert(NewCData.matches.begin(), kv.second.begin(), kv.second.end());
			NewCData.matches = kv.second;
			NewCData.init_size = NewCData.matches.size();

			//DMatch& m = kv.second.at(0);
			//DMatch& m = kv.second.at(0);
			/*NewCData.matches.at(0).queryIdx = -110;
			kv.second.at(0).queryIdx = -11110;*/

			this->clusters.push_back(NewCData);
		}
	}
	std::sort(this->clusters.begin(), this->clusters.end(), [](Cluster_data const& f, Cluster_data const& s){ return f.matches.size() > s.matches.size(); });
}

void Hough_Transform::sortClusters()
{
	std::sort(this->clusters.begin(), this->clusters.end(), 
		[](Cluster_data const& f, Cluster_data const& s){ return f.matches.size() > s.matches.size(); });
}

bool compareMatches(DMatch const & m1, DMatch const& m2)
{
	if (m1.queryIdx > m2.queryIdx)
	{
		return true;
	}
	if (m1.queryIdx < m2.queryIdx)
	{
		return false;
	}
	if (m1.trainIdx > m2.trainIdx)
	{
		return true;
	}
	return false;
}

int compareMatches2(DMatch const & m1, DMatch const& m2)
{
	if (m1.queryIdx > m2.queryIdx)
	{
		return 1;
	}
	if (m1.queryIdx < m2.queryIdx)
	{
		return -1;
	}
	if (m1.trainIdx > m2.trainIdx)
	{
		return 1;
	}
	if (m1.trainIdx < m2.trainIdx)
	{
		return -1;
	}
	return 0;
}

void Hough_Transform::UseTransformConstraint(double initPointPercTh, double initModelPercTh, double InitialRotationThresh, double InitialScaleThresh, TransformType transfType, double delClThresh, int hip_check, int RANSAC_iter, double RANSAC_inlierDistT)
{
	int eliminated_num = 0;
	char key = 'o';

	//int nthreads;
	int clusters_num = this->clusters.size();
	vector<Cluster_data> newClusters;
	vector<DMatch> allfoundMatches;
	size_t smallClusterSize = 4;
	//int i = 0;
#if !defined(_DEBUG) && defined(OMP_OPTIMIZATION)
#pragma omp parallel for private(i) schedule(guided,1) //shared(newClusters)
#endif
	for (size_t i = 0; i < this->clusters.size(); i++)
	{
	//	int numthreads = omp_get_num_threads();
	//	int numThread = omp_get_thread_num();
		int orSize = this->clusters.at(i).matches.size();
		if (allfoundMatches.size() >= smallClusterSize)
		{
			std::sort(this->clusters.at(i).matches.begin(), this->clusters.at(i).matches.end(), compareMatches);
			vector<DMatch> tmp(allfoundMatches.size() + this->clusters.at(i).matches.size());
			std::vector<DMatch>::iterator it = std::set_difference(this->clusters.at(i).matches.begin(), this->clusters.at(i).matches.end(), allfoundMatches.begin(), allfoundMatches.end(), tmp.begin(), compareMatches);
			tmp.resize(it - tmp.begin());
			this->clusters.at(i).matches = tmp;
		}
		vector<DMatch> eliminatedMatches = UseTransfForCluster(this->clusters.at(i), initPointPercTh, initModelPercTh, InitialRotationThresh, InitialScaleThresh, transfType, delClThresh, hip_check, RANSAC_iter, RANSAC_inlierDistT);
		vector<DMatch> foundMatches = this->clusters.at(i).matches;

		if (foundMatches.size() >= smallClusterSize)
		{
			std::sort(foundMatches.begin(), foundMatches.end(), compareMatches);
			if (allfoundMatches.size() == 0)
			{
				allfoundMatches.insert(allfoundMatches.begin(), foundMatches.begin(), foundMatches.end());
			}
			else
			{
				vector<DMatch> tmp(allfoundMatches.size() + foundMatches.size());
				std::vector<DMatch>::iterator it = std::set_union(allfoundMatches.begin(), allfoundMatches.end(), foundMatches.begin(), foundMatches.end(), tmp.begin(), compareMatches);
				tmp.resize(it - tmp.begin());
				allfoundMatches = tmp;

			}
		}
	}
	//int numthreads = omp_get_num_threads();
	this->sortClusters();
}


vector<DMatch> Hough_Transform::UseTransfForCluster(Cluster_data& NewCData, double initPointPercTh, double initModelPercTh, double InitialRotationThresh, double InitialScaleThresh, TransformType transfType, double deleteClusterThresh, int hip_check, int RANSAC_iter, double RANSAC_inlierDistT)
{
	double meanDistance = DBL_MAX;
	int repeat = 0;
	double rejectDist = 140;
	double ModelThresh = initModelPercTh;
	double PointThresh = initPointPercTh;
	double rotaionThresh = InitialRotationThresh;
	double scaleThresh = InitialScaleThresh;
	int eliminated = 0;
	//char windowname[200];

	vector<DMatch> allEliminatedMatches;
	double scale1 = 0;
	double scale2 = 0;
	double ratio = 0;
	for (size_t i = 0; i < NewCData.matches.size(); i++)
	{
		DMatch m = NewCData.matches.at(i);
		int indx1 = m.queryIdx;
		int indx2 = m.trainIdx;

		scale1 = keypoints1.at(indx1).size;
		scale2 = keypoints2.at(indx2).size;
		ratio += scale2 / scale1;
	}

	ratio /= NewCData.matches.size();
	
	double inlier_dist = RANSAC_inlierDistT;
	NewCData.fitModelParams(this->keypoints1, this->keypoints2, transfType, 0, image1, image2, inlier_dist, RANSAC_iter, hip_check, deleteClusterThresh);
	TransformType transfTypeNew = TransformType::SIMILARITY_TRANSFORM;
	if (transfType == TransformType::AFFINE_TRANSFORM)
	{
		transfTypeNew = TransformType::AFFINE_TRANSFORM;
	}
	if (transfType != TransformType::OPENCV_HOMOGRAPHY_RANSAC)
	{
		NewCData.fitModelParams(this->keypoints1, this->keypoints2, transfTypeNew, 0, image1, image2);
	}
	int tmp = NewCData.transfMat.size().width;
	vector<DMatch> eliminatedMatches;
	eliminatedMatches = NewCData.eliminateOutliers(this->keypoints1, this->keypoints2, PointThresh, ModelThresh, rotaionThresh, scaleThresh, this->image1, this->image2, 0);
	allEliminatedMatches.insert(allEliminatedMatches.end(), eliminatedMatches.begin(), eliminatedMatches.end());

	return allEliminatedMatches;
}