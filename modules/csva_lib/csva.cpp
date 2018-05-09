#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;
#include "csva.h"
#include "matching_hough.h"
#include "misc_functions.h"
#include "confidence_estimation.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
//#include "aerospace_demo.h"
//#include "3Drecognition.h"
//#define DEBUG_INFO_OUTPUT
#ifdef _DEBUG
#include "misc_visualizers.h"
#endif
namespace csva
{
	vector<DMatch> verify_cluster(vector<DMatch> matches,  const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const Mat& image1, const Mat& image2, Mat& PT)
	{
		vector<DMatch> inliers;
		Mat trM;
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

	vector<DMatch> verify_cluster3D(const vector<DMatch>& matches, const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const Mat& image1, const Mat& image2, Mat& PT)
	{
		vector<DMatch> inliers;
		Mat trM;
		int init_size = matches.size();
		Cluster_data mcl;
		mcl.matches = matches;
		trM = mcl.fitModelParamsSimilarityRansac(kpts1, kpts2, image1, image2, 40, 0, 0.08, 30, 2.);
		mcl.eliminateOutliers(kpts1, kpts2, 0.4, 0.2, 15, sqrt(2.), image1, image2, 0);
		trM =  mcl.fitModelParams(kpts1, kpts2, SIMILARITY_TRANSFORM, 0, image1, image2);
		//mcl.eliminateOutliers(kpts1, kpts2, 0.4, 0.9, 30, 2, image1, image2, 0);
		
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
	//std::array<double, 6>  calculateConfidenceClusterData(Cluster_data& mcl, const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const Mat& image1, const Mat& image2,
	//	const vector<DMatch>& excludedMatches, int type, Mat& PT, double LoweProb, int extoutput)
	//{
	//	return calculateConfidenceClusterData2(mcl, kpts1, kpts2, image1, image2, excludedMatches, type, PT, LoweProb, extoutput);
	//}

	vector<DMatch> matchesHoughConstraint(Mat image1, Mat image2, Mat* matchresult, vector<KeyPoint> points1, vector<KeyPoint> points2, vector<DMatch> matches, int graphicalOuput, int consoleOuput)
	{
		/*int num = 1;
		int hip_check = 0;
		int ransac_iterations = 27;
		double delClThresh = 0.3;
		double RANSAC_inlierdist = sqrt((double)(image2.size().width * image2.size().width)+(image2.size().height * image2.size().height)) * 0.08;*/

		double RANSAC_inlierdist = sqrt((double)(image2.size().width * image2.size().width) + (image2.size().height * image2.size().height)) * 0.06;

		int maxres = 0.8 * sqrt(1.*image2.size().width*image2.size().width + image2.size().height*image2.size().height);//image1.size().width > image1.size().height ? image1.size().width : image1.size().height; //
		Hough_Transform HoughTransform(((double)(360.)) / 24., maxres / 8., maxres / 8., 2., image1, image2);
		int demanded_size_of_cluster = matches.size() / 100;
		demanded_size_of_cluster = demanded_size_of_cluster < 3 ? 4 : demanded_size_of_cluster < 5 ? 5 : demanded_size_of_cluster;
		//clock_t start = clock();
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
			{
				smallClusterSize = 4;
			}
			else
			{
				smallClusterSize = 5;
			}
		}
		//if (consoleOuput)
		//{
		//	printf("transform constraint time = %d ms\n", (chpoint3 - start) * 1000 / CLOCKS_PER_SEC);
		//	printf("smallClusterSize = %d\n", smallClusterSize);
		//}
		//////system("pause");
		//if (graphicalOuput)
		//{
		//	HoughTransform.ShowClusters(HoughTransform.clusters, 3, "After Elimination");
		//}

		//HoughTransform.removeSmallClusters(smallClusterSize, false);
		//if (graphicalOuput)
		//{
		//	HoughTransform.extended_output[1] = 1;
		//}

		//HoughTransform.excludeDuplicates();
		//if (graphicalOuput)
		//{
		//	HoughTransform.ShowClusters(HoughTransform.clusters, 3, "excluded duplicates");
		//}
		HoughTransform.ExcludeMany2OneFromClusters();
		HoughTransform.ExcludeOne2ManyFromClusters();
		HoughTransform.removeSmallClusters(smallClusterSize - 1, false);

		//if (matchresult != NULL)
		//{
		//	clock_t finish = clock();
		//	start = clock();
		//	*matchresult = HoughTransform.printClusters(HoughTransform.clusters, -1);
		//	clock_t chpoint5 = clock();
		//	if (consoleOuput)
		//	{
		//		printf("\nprint time = %d ms\n", (chpoint5 - start) * 1000 / CLOCKS_PER_SEC);
		//	}
		//}
		//if (graphicalOuput)
		//{
		//	HoughTransform.ShowClusters(HoughTransform.clusters, 3, "final");
		//}
		return HoughTransform.getAllClusterMatches();
	}

	cv::Mat csva_filtering_aero(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const vector<DMatch>& matches, const Mat& image1, const Mat& image2, int type, vector<DMatch>& inliers, double* confidence, double LoweProb)
	{
		vector<DMatch> goodmatches;
		
		csva::primary_filtering(kpts1, kpts2, matches, 0.98, goodmatches);
		vector<vector<DMatch> > clusters, verified;
		vector<Mat> transforms;
		csva::hough_transform(kpts1, kpts2, goodmatches, image1, image2, clusters);
		int number_of_matches = 0;
		int max_cluster = 0;
		for (vector<DMatch> ms : clusters)
		{
			number_of_matches += ms.size();
			if (ms.size() > max_cluster)
			{
				max_cluster = ms.size();
			}
		}
		csva::verify_clusters(clusters, verified, transforms, kpts1, kpts2, image1, image2);
		float maxConf = 0.f;
		Mat PT;
		array<double, 6> bestconf;

		/*
		select the best solution
		*/
		for (int i = 0; i < verified.size(); i++)
		{
			Mat tr = transforms.at(i);
			vector<DMatch>& ms = verified.at(i);
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

		for (int k = 0; k < 6; k++)
		{
			confidence[k] = bestconf[k];
		}
		return PT;
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

	void csva_filtering_3D(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const vector<DMatch>& matches, const Mat image1, const Mat& image2, int type, vector<DMatch>& inliers, double* confidence, double LoweProb)
	{	
		srand(0);
		Mat result;
		int time = 0;
		vector<DMatch> goodmatches;
		goodmatches = matches;

		csva::primary_filtering(kpts1, kpts2, matches, 1, goodmatches);
		vector< vector<DMatch> > clusters;
		int demanded_size_of_cluster = goodmatches.size() / 100;
		demanded_size_of_cluster = demanded_size_of_cluster < 3 ? 4 : demanded_size_of_cluster < 5 ? 5 : demanded_size_of_cluster;
		csva::hough_transform(kpts1, kpts2, goodmatches, image1, image2, clusters, demanded_size_of_cluster);
		//matchesHoughConstraint(image1, image2, &result, kpts1, kpts2, goodmatches, 0, 0);
		vector<DMatch> allfoundMatches;
		for (int i = 0; i < clusters.size(); i++)
		{
			
			vector<DMatch> cluster_matches = clusters.at(i);
#ifdef _DEBUG
			result = printMatches(kpts1, kpts2, cluster_matches, image1, image2);
			//showMatches(kpts1, kpts2, cluster_matches, image1, image2, "initial", 1);
#endif
			if (allfoundMatches.size() >= demanded_size_of_cluster)
			{
				std::sort(cluster_matches.begin(), cluster_matches.end(), compareMatches);
				vector<DMatch> tmp(allfoundMatches.size() + cluster_matches.size());
				std::vector<DMatch>::iterator it = std::set_difference(cluster_matches.begin(), cluster_matches.end(), allfoundMatches.begin(), allfoundMatches.end(), tmp.begin(), compareMatches);
				tmp.resize(it - tmp.begin());
				cluster_matches = tmp;
			}
#ifdef _DEBUG
			result = printMatches(kpts1, kpts2, cluster_matches, image1, image2);
			//showMatches(kpts1, kpts2, cluster_matches, image1, image2, "excluded_dups", 1);
#endif
			Mat PT;

			vector<DMatch> inliers_in_cluster = verify_cluster3D(cluster_matches, kpts1, kpts2, image1, image2, PT);
			if (inliers_in_cluster.size() < demanded_size_of_cluster || PT.empty())
			{
				continue;
			}
			
			array<double, 6> conf = csva::confidence_estimation(inliers_in_cluster, PT, kpts1, kpts2, goodmatches, image1, image2, 0, type, LoweProb * 2);
#ifdef _DEBUG
			printf("confid: %f\n", conf[0]);
			result = printMatches(kpts1, kpts2, inliers_in_cluster, image1, image2);
			//showMatches(kpts1, kpts2, inliers_in_cluster, image1, image2, "inliers", 1);
#endif
			
			if (conf[0] < 0.95)
			{
				continue;
			}
			//csva::confidence_estimation(inliers_in_cluster, PT, kpts1, kpts2, goodmatches, image1, image2, 0, type, LoweProb);
			for (DMatch m : inliers_in_cluster)
			{
				allfoundMatches.push_back(m);
			}
			std::sort(allfoundMatches.begin(), allfoundMatches.end(), compareMatches);
		}
		inliers = allfoundMatches;
#ifdef _DEBUG
		result = printMatches(kpts1, kpts2, inliers, image1, image2);
		//showMatches(kpts1, kpts2, inliers, image1, image2, "sfs", 1);
#endif

	}
	
	CSVA_LIB_API cv::Mat filter_matches(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const vector<DMatch>& matches,
		const Mat& im1, const  Mat& im2,  int mode,
		int type, vector<DMatch> &inliers, 
		double* confidence, double LoweProb)
	{
		if (mode == 0)
		{
			return csva_filtering_aero(kpts1, kpts2, matches, im1, im2, type, inliers, confidence, LoweProb);
		}
		else
		{
			csva_filtering_3D(kpts1, kpts2, matches, im1, im2, type, inliers, confidence, LoweProb);
			return cv::Mat::eye(4, 4, CV_32F);
		}
	}
	
	CSVA_LIB_API void primary_filtering(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2,
		const vector<DMatch>& matches, float NNthresh, vector<DMatch> &inliers)
	{
		vector<DMatch> initialMatches(matches);
		int intialNumberOfMatches = matches.size();
		vector<DMatch> goodmatches = matches;
		if (NNthresh < 1.)
		{
			goodmatches = useNNratio(goodmatches, NNthresh);
		}
		inliers = excludeMany2OneMatches(goodmatches, kpts1, kpts2);
	}
	CSVA_LIB_API void hough_transform(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const vector<DMatch>& matches, const Mat& image1, const Mat& image2, vector<vector<DMatch> >& clusters, int vote_thresh)
	{
		vector<DMatch> goodmatches = matches;
		bool useOneAcc = true;
		Cluster_data mcl_max;
		int max_cluster_size = 0;
		int maxres = image1.size().width > image1.size().height ? image1.size().width : image1.size().height;
		vector<Cluster_data> potential_clusters;
		if (useOneAcc)
		{
			Hough_Transform HoughTransform(((double)(360.)) / 12., maxres / 8, maxres / 8, 2, image1, image2);
			HoughTransform.FillAcc(kpts1, kpts2, goodmatches);
			mcl_max = HoughTransform.MaxCluster();
			max_cluster_size = mcl_max.matches.size();
			if (!vote_thresh)
			{
				vote_thresh = mcl_max.matches.size() * 0.6;
			}
			if (vote_thresh < 3)
			{
				vote_thresh = 3;
			}
			HoughTransform.FindClusters(vote_thresh);
			potential_clusters = HoughTransform.clusters;
#ifdef DEBUG_INFO_OUTPUT
			HoughTransform.ShowClusters(potential_clusters, 3, "clusters");
#endif
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
			int vote_thresh = mcl_max.matches.size() * 0.6;
			if (vote_thresh < 3)
			{
				vote_thresh = 3;
			}
			HoughTransform1.FindClusters(vote_thresh);
			HoughTransform2.FindClusters(vote_thresh);
			potential_clusters = HoughTransform1.clusters;
			potential_clusters.insert(potential_clusters.end(), HoughTransform2.clusters.begin(), HoughTransform2.clusters.end());
#ifdef DEBUG_INFO_OUTPUT
			HoughTransform1.ShowClusters(potential_clusters, 3, "clusters");
			HoughTransform1.ShowClusters(3, "clusters1");
			HoughTransform2.ShowClusters(3, "clusters1");
#endif
		}

		//vector< vector<DMatch> > clusters;
		for (Cluster_data c: potential_clusters)
		{
			clusters.push_back(c.matches);
		}
	}
//
//	CSVA_LIB_API void hough_transform2(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const vector<DMatch>& matches, const Mat& image1, const Mat& image2, vector<vector<DMatch> >& clusters, int vote_thresh)
//	{
//		vector<DMatch> goodmatches = matches;
//		bool useOneAcc = true;
//		Cluster_data mcl_max;
//		int max_cluster_size = 0;
//		int maxres = 0.8 * sqrt(1.*image2.size().width*image2.size().width + image2.size().height*image2.size().height);//image1.size().width > image1.size().height ? image1.size().width : image1.size().height; //
//		
//		vector<Cluster_data> potential_clusters;
//		if (useOneAcc)
//		{
//			Hough_Transform HoughTransform(((double)(360.)) / 24., maxres / 8., maxres / 8., 2., image1, image2);
//			//Hough_Transform HoughTransform(((double)(360.)) / 12., maxres / 8, maxres / 8, 2, image1, image2);
//			HoughTransform.FillAcc(kpts1, kpts2, goodmatches);
//			mcl_max = HoughTransform.MaxCluster();
//			max_cluster_size = mcl_max.matches.size();
//			if (!vote_thresh)
//			{
//				vote_thresh = mcl_max.matches.size() * 0.6;
//			}
//			if (vote_thresh < 3)
//			{
//				vote_thresh = 3;
//			}
//			HoughTransform.FindClusters(vote_thresh);
//			potential_clusters = HoughTransform.clusters;
//#ifdef DEBUG_INFO_OUTPUT
//			HoughTransform.ShowClusters(potential_clusters, 3, "clusters");
//#endif
//		}
//		else
//		{
//			Hough_Transform HoughTransform1(((double)(360.)) / 8, maxres / 3, maxres / 3, 2., image1, image2);
//			Hough_Transform HoughTransform2(((double)(360.)) / 8, maxres / 3, maxres / 3, 2., image1, image2);
//			HoughTransform1.FillAccNewBoundary(kpts1, kpts2, goodmatches, 1);
//			HoughTransform2.FillAccNewBoundary(kpts1, kpts2, goodmatches, 0);
//			Cluster_data mcl_max1 = HoughTransform1.MaxCluster();
//			Cluster_data mcl_max2 = HoughTransform2.MaxCluster();
//			mcl_max = mcl_max1.matches.size() > mcl_max2.matches.size() ? mcl_max1 : mcl_max2;
//			max_cluster_size = mcl_max.matches.size();
//			int vote_thresh = mcl_max.matches.size() * 0.6;
//			if (vote_thresh < 3)
//			{
//				vote_thresh = 3;
//			}
//			HoughTransform1.FindClusters(vote_thresh);
//			HoughTransform2.FindClusters(vote_thresh);
//			potential_clusters = HoughTransform1.clusters;
//			potential_clusters.insert(potential_clusters.end(), HoughTransform2.clusters.begin(), HoughTransform2.clusters.end());
//#ifdef DEBUG_INFO_OUTPUT
//			HoughTransform1.ShowClusters(potential_clusters, 3, "clusters");
//			HoughTransform1.ShowClusters(3, "clusters1");
//			HoughTransform2.ShowClusters(3, "clusters1");
//#endif
//		}
//
//		//vector< vector<DMatch> > clusters;
//		for (Cluster_data c : potential_clusters)
//		{
//			clusters.push_back(c.matches);
//		}
//	}

	CSVA_LIB_API void verify_clusters(const vector< vector<DMatch> >& clusters, vector< vector<DMatch> >& filtered, 
		vector<Mat>& transforms, const vector<KeyPoint>& kpts1, 
		const vector<KeyPoint>& kpts2, const Mat& image1, const Mat& image2)
	{
		//int number_of_matches_in_clusters;
		for (int i = 0; i < clusters.size(); i++)
		{
			Mat trM_;
			//Cluster_data cl = potential_clusters.at(i);
			vector<DMatch> matches = clusters.at(i);
			//number_of_matches_in_clusters += matches.size();
			Mat PT;
			vector<DMatch> inliers = verify_cluster(matches, kpts1, kpts2, image1, image2, PT);
			if (inliers.size() > 0 && !PT.empty())
			{
				filtered.push_back(inliers);
				transforms.push_back(PT);
			}
		}
	}

	CSVA_LIB_API std::array<double, 6> confidence_estimation(vector<DMatch>& inliers, const Mat& PT, vector<KeyPoint> kpts1, vector<KeyPoint> kpts2,
		const vector<DMatch> &excludedMatches, Mat im1, Mat im2, int mode, int type, double LoweProb)
	{
		std::array<double, 6> confidence{ 0,0,0,0,0,0 };
		if (!PT.empty() && inliers.size() > 0)
		{
			confidence = calculateConfidence(PT, inliers, kpts1, kpts2, im1, im2, excludedMatches, type, LoweProb);
			//confidence[0] = pow(confidence[0], clusters_num);
		}
		return confidence;
	}
}