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

#include <opencv2/core/core.hpp>

#include "csva.h"
#include "matching_hough.h"
#include "misc_functions.h"
#include "confidence_estimation.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>

namespace csva
{
    std::vector<cv::DMatch> verify_cluster(std::vector<cv::DMatch> matches,  const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, const cv::Mat& image1, const cv::Mat& image2, cv::Mat& PT)
	{
        std::vector<cv::DMatch> inliers;
        cv::Mat trM;
		//int inlier_dist = sqrt((double)(image2.size().width * image2.size().width + image2.size().height * image2.size().height)) * 0.04;
		int init_size = matches.size();
		Cluster_data mcl;
		mcl.matches = matches;
		trM = mcl.fitModelParamsSimilarityRansac(kpts1, kpts2, image1, image2, 1000, 100, 0.08, 15, 2);
		mcl.fitModelParams(kpts1, kpts2, SIMILARITY_TRANSFORM, 0, image1, image2);
		mcl.eliminateOutliers(kpts1, kpts2, 0.8, 0.8, 30, 2, image1, image2, 0);
		trM = mcl.fitModelParams(kpts1, kpts2, AFFINE_TRANSFORM, 0);
		mcl.eliminateOutliers(kpts1, kpts2, 0.4, 0.9, 30, 2, image1, image2, 0);
		trM = mcl.fitModelParams(kpts1, kpts2, AFFINE_TRANSFORM, 0);

		
		if (!trM.empty())
		{
            PT = cv::Mat::zeros(3, 3, CV_64F);//cv::Mat(3, 3, CV_64F);
			if (trM.size().height > 2)
			{
				PT = trM.clone();
			}
			else
			{
				PT.at<double>(0, 0) = trM.at<double>(0, 0);
				PT.at<double>(0, 1) = trM.at<double>(0, 1);
				PT.at<double>(1, 0) = trM.at<double>(1, 0);
				PT.at<double>(1, 1) = trM.at<double>(1, 1);
				PT.at<double>(1, 2) = trM.at<double>(1, 2);
				PT.at<double>(0, 2) = trM.at<double>(0, 2);
				PT.at<double>(2, 0) = 0;
				PT.at<double>(2, 1) = 0;
				PT.at<double>(2, 2) = 1;
			}
		}
		return mcl.matches;

	}

    std::vector<cv::DMatch> verify_cluster3D(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, const cv::Mat& image1, const cv::Mat& image2, cv::Mat& PT)
	{
        std::vector<cv::DMatch> inliers;
        cv::Mat trM;
		int init_size = matches.size();
		Cluster_data mcl;
		mcl.matches = matches;
		trM = mcl.fitModelParamsSimilarityRansac(kpts1, kpts2, image1, image2, 40, 0, 0.08, 30, 2.);
		mcl.eliminateOutliers(kpts1, kpts2, 0.4, 0.2, 15, sqrt(2.), image1, image2, 0);
		trM =  mcl.fitModelParams(kpts1, kpts2, TransformType::SIMILARITY_TRANSFORM, 0, image1, image2);
		
		if (!trM.empty())
		{
            PT = cv::Mat::zeros(3, 3, CV_64F);//cv::Mat(3, 3, CV_64F);
			if (trM.size().height > 2)
			{
				PT = trM.clone();
			}
			else
			{
				PT.at<double>(0, 0) = trM.at<double>(0, 0);
				PT.at<double>(0, 1) = trM.at<double>(0, 1);
				PT.at<double>(1, 0) = trM.at<double>(1, 0);
				PT.at<double>(1, 1) = trM.at<double>(1, 1);
				PT.at<double>(1, 2) = trM.at<double>(1, 2);
				PT.at<double>(0, 2) = trM.at<double>(0, 2);
				PT.at<double>(2, 0) = 0;
				PT.at<double>(2, 1) = 0;
				PT.at<double>(2, 2) = 1;
			}
		}
		return mcl.matches;

	}

    std::vector<cv::DMatch> matchesHoughConstraint(cv::Mat image1, cv::Mat image2, cv::Mat* matchresult, std::vector<cv::KeyPoint> points1, std::vector<cv::KeyPoint> points2, std::vector<cv::DMatch> matches, int graphicalOuput, int consoleOuput)
	{
		double RANSAC_inlierdist = sqrt((double)(image2.size().width * image2.size().width) + (image2.size().height * image2.size().height)) * 0.06;

		int maxres = static_cast<int>(0.8 * sqrt(1.*image2.size().width*image2.size().width + image2.size().height*image2.size().height));
		Hough_Transform HoughTransform(((double)(360.)) / 24., maxres / 8., maxres / 8., 2., image1, image2);
		int demanded_size_of_cluster = matches.size() / 100;
		demanded_size_of_cluster = demanded_size_of_cluster < 3 ? 4 : demanded_size_of_cluster < 5 ? 5 : demanded_size_of_cluster;
		if (1)
		{
			HoughTransform.FillAcc(points1, points2, matches);
			HoughTransform.FindClusters(demanded_size_of_cluster);
		}
		TransformType transfType = TransformType::SIMILARITY_RANSAC;
		double ProjF = 0;
		double ProjFModel = 0;
		double RotF = 0;
		double ScaleF = 0;

		double ScThresh = 1.41;
		double RotThresh = 15;
		double DistModelThresh = 0.2;
		double DistProjThresh = 0.4;

		int num = 1;
		int hip_check = 0;
		int ransac_iterations = 20;
		double delClThresh = 0.3;
		HoughTransform.UseTransformConstraint(DistProjThresh, DistModelThresh, RotThresh, ScThresh, transfType, delClThresh, hip_check, ransac_iterations, RANSAC_inlierdist);

        int smallClusterSize = matches.size() / 100;
        if (smallClusterSize < 5)
        {
            if (smallClusterSize < 3)
                smallClusterSize = 4;
            else
                smallClusterSize = 5;
        }

        HoughTransform.ExcludeMany2OneFromClusters();
        HoughTransform.ExcludeOne2ManyFromClusters();
        HoughTransform.removeSmallClusters(smallClusterSize - 1, false);

        return HoughTransform.getAllClusterMatches();
    }

    cv::Mat csva_filtering_aero(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, const std::vector<cv::DMatch>& matches, const cv::Mat& image1, const cv::Mat& image2, int type, std::vector<cv::DMatch>& inliers, double* confidence, double LoweProb)
    {
        std::vector<cv::DMatch> goodmatches;
		clock_t start = clock();
		csva::primary_filtering(kpts1, kpts2, matches, 0.98f, goodmatches);
		printf("primary_filtering = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock(); 
        std::vector<std::vector<cv::DMatch> > clusters, verified;
        std::vector<cv::Mat> transforms;
		csva::hough_transform(kpts1, kpts2, goodmatches, image1, image2, clusters);
		printf("hough_transform = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock();
		int number_of_matches = 0;
		size_t max_cluster = 0;
        for (std::vector<cv::DMatch> ms : clusters)
		{
			number_of_matches += ms.size();
			if (ms.size() > max_cluster)
			{
				max_cluster = ms.size();
			}
		}
		csva::verify_clusters(clusters, verified, transforms, kpts1, kpts2, image1, image2);
		printf("verify_clusters = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock();
		double maxConf = 0.;
        cv::Mat PT;
        std::array<double, 6> bestconf;

		/*
		select the best solution
		*/
		for (size_t i = 0; i < verified.size(); i++)
		{
            cv::Mat tr = transforms.at(i);
            std::vector<cv::DMatch>& ms = verified.at(i);
			if (!tr.empty())
			{
				std::array<double, 6> conf = csva::confidence_estimation(ms, tr, kpts1, kpts2, matches, image1, image2, 0, type, LoweProb * 2);
				if (maxConf < conf[0])
				{
					PT = tr.clone();
					maxConf = conf[0];
					inliers = verified.at(i);
					bestconf = conf;
				}
			}
		}
		printf("findt the best solution = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock();
		for (int k = 0; k < 6; k++)
		{
			confidence[k] = bestconf[k];
		}
		return PT;
	}

    bool compareMatches(cv::DMatch const & m1, cv::DMatch const& m2)
	{
		if (m1.queryIdx > m2.queryIdx)
			return true;
		if (m1.queryIdx < m2.queryIdx)
			return false;
		if (m1.trainIdx > m2.trainIdx)
			return true;

		return false;
	}

    void csva_filtering_3D(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, const std::vector<cv::DMatch>& matches, const cv::Mat image1, const cv::Mat& image2, int type, std::vector<cv::DMatch>& inliers, double* confidence, double LoweProb)
	{	
		srand(0);
        cv::Mat result;
		int time = 0;
        std::vector<cv::DMatch> goodmatches;
		goodmatches = matches;

		csva::primary_filtering(kpts1, kpts2, matches, 1, goodmatches);
        std::vector<std::vector<cv::DMatch> > clusters;
		size_t demanded_size_of_cluster = goodmatches.size() / 100;
		demanded_size_of_cluster = demanded_size_of_cluster < 3 ? 4 : demanded_size_of_cluster < 5 ? 5 : demanded_size_of_cluster;
		csva::hough_transform(kpts1, kpts2, goodmatches, image1, image2, clusters, demanded_size_of_cluster);
		//matchesHoughConstraint(image1, image2, &result, kpts1, kpts2, goodmatches, 0, 0);
        std::vector<cv::DMatch> allfoundMatches;
		for (size_t i = 0; i < clusters.size(); i++)
		{
            std::vector<cv::DMatch> cluster_matches = clusters.at(i);

			if (allfoundMatches.size() >= demanded_size_of_cluster)
			{
				std::sort(cluster_matches.begin(), cluster_matches.end(), compareMatches);
                std::vector<cv::DMatch> tmp(allfoundMatches.size() + cluster_matches.size());
                std::vector<cv::DMatch>::iterator it = std::set_difference(cluster_matches.begin(), cluster_matches.end(), allfoundMatches.begin(), allfoundMatches.end(), tmp.begin(), compareMatches);
				tmp.resize(it - tmp.begin());
				cluster_matches = tmp;
			}

            cv::Mat PT;

            std::vector<cv::DMatch> inliers_in_cluster = verify_cluster3D(cluster_matches, kpts1, kpts2, image1, image2, PT);
			if (inliers_in_cluster.size() < demanded_size_of_cluster || PT.empty())
				continue;
			
            std::array<double, 6> conf = csva::confidence_estimation(inliers_in_cluster, PT, kpts1, kpts2, goodmatches, image1, image2, 0, type, LoweProb * 2);
			
			if (conf[0] < 0.95)
				continue;

            for (cv::DMatch m : inliers_in_cluster)
			{
				allfoundMatches.push_back(m);
			}
			std::sort(allfoundMatches.begin(), allfoundMatches.end(), compareMatches);
		}
		inliers = allfoundMatches;

	}
	
    CSVA_LIB_API cv::Mat filter_matches(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, const std::vector<cv::DMatch>& matches,
        const cv::Mat& im1, const cv::Mat& im2, geometry_mode mode,
        int type, std::vector<cv::DMatch> &inliers,
		double* confidence, double LoweProb)
	{
		if (mode == 0)
		{
			return csva_filtering_aero(kpts1, kpts2, matches, im1, im2, type, inliers, confidence, LoweProb);
		}
		else
		{
			csva_filtering_3D(kpts1, kpts2, matches, im1, im2, type, inliers, confidence, LoweProb);
            return cv::Mat::eye(3, 3, CV_32F);
		}
	}
	
    CSVA_LIB_API void primary_filtering(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2,
        const std::vector<cv::DMatch>& matches, float NNthresh, std::vector<cv::DMatch> &inliers)
	{
        std::vector<cv::DMatch> initialMatches(matches);
		int intialNumberOfMatches = matches.size();
        std::vector<cv::DMatch> goodmatches = matches;
		if (NNthresh < 1.)
			goodmatches = useNNratio(goodmatches, NNthresh);

		inliers = excludeMany2OneMatches(goodmatches, kpts1, kpts2);
	}
    CSVA_LIB_API void hough_transform(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, const std::vector<cv::DMatch>& matches, const cv::Mat& image1, const cv::Mat& image2, std::vector<std::vector<cv::DMatch> >& clusters, int vote_thresh)
	{
        std::vector<cv::DMatch> goodmatches = matches;
		bool useOneAcc = true;
		Cluster_data mcl_max;
		int max_cluster_size = 0;
		int maxres = image1.size().width > image1.size().height ? image1.size().width : image1.size().height;
        std::vector<Cluster_data> potential_clusters;
		if (useOneAcc)
		{
			Hough_Transform HoughTransform(((double)(360.)) / 12., maxres / 8, maxres / 8, 2, image1, image2);
			HoughTransform.FillAcc(kpts1, kpts2, goodmatches);
			mcl_max = HoughTransform.MaxCluster();
			max_cluster_size = mcl_max.matches.size();
			if (!vote_thresh)
				vote_thresh = int(mcl_max.matches.size() * 0.6);
			if (vote_thresh < 3)
				vote_thresh = 3;

			HoughTransform.FindClusters(vote_thresh);
			potential_clusters = HoughTransform.clusters;
		}
		else
		{
			Hough_Transform HoughTransform1(((double)(360.)) / 8, maxres / 3, maxres / 3, 2., image1, image2);
			Hough_Transform HoughTransform2(((double)(360.)) / 8, maxres / 3, maxres / 3, 2., image1, image2);
			HoughTransform1.FillAccNewBoundary(kpts1, kpts2, goodmatches, 1);
			HoughTransform2.FillAccNewBoundary(kpts1, kpts2, goodmatches, 0);
			Cluster_data mcl_max1 = HoughTransform1.MaxCluster();
			Cluster_data mcl_max2 = HoughTransform2.MaxCluster();
			mcl_max = mcl_max1.matches.size() > mcl_max2.matches.size() ? mcl_max1 : mcl_max2;
			max_cluster_size = mcl_max.matches.size();
			int vote_thresh = int(mcl_max.matches.size() * 0.6);
			if (vote_thresh < 3)
				vote_thresh = 3;

			HoughTransform1.FindClusters(vote_thresh);
			HoughTransform2.FindClusters(vote_thresh);
			potential_clusters = HoughTransform1.clusters;
			potential_clusters.insert(potential_clusters.end(), HoughTransform2.clusters.begin(), HoughTransform2.clusters.end());
		}

		for (Cluster_data c: potential_clusters)
		{
			clusters.push_back(c.matches);
		}
	}

    CSVA_LIB_API void verify_clusters(const std::vector<std::vector<cv::DMatch> >& clusters, std::vector<std::vector<cv::DMatch> >& filtered,
        std::vector<cv::Mat>& transforms, const std::vector<cv::KeyPoint>& kpts1,
        const std::vector<cv::KeyPoint>& kpts2, const cv::Mat& image1, const cv::Mat& image2)
	{
		for (size_t i = 0; i < clusters.size(); i++)
		{
            cv::Mat trM_;
            std::vector<cv::DMatch> matches = clusters.at(i);
            cv::Mat PT;
            std::vector<cv::DMatch> inliers = verify_cluster(matches, kpts1, kpts2, image1, image2, PT);
			if (inliers.size() > 0 && !PT.empty())
			{
				filtered.push_back(inliers);
				transforms.push_back(PT);
			}
		}
	}

    CSVA_LIB_API std::array<double, 6> confidence_estimation(std::vector<cv::DMatch>& inliers, const cv::Mat& PT, std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2,
        const std::vector<cv::DMatch> &excludedMatches, cv::Mat im1, cv::Mat im2, int mode, int type, double LoweProb)
	{
		std::array<double, 6> confidence{ 0,0,0,0,0,0 };
		if (!PT.empty() && inliers.size() > 0)
			confidence = calculateConfidence(PT, inliers, kpts1, kpts2, im1, im2, excludedMatches, type, LoweProb);

		return confidence;
	}
}
