#pragma once;
#include <opencv2/core.hpp>



#if defined(_MSC_VER)
	#ifdef csva_lib_EXPORTS
		#define CSVA_LIB_API __declspec(dllexport) 
	#else
		#define CSVA_LIB_API __declspec(dllimport) 
	#endif
#elif defined(__GNUC__)
	#ifdef aero_lib_EXPORTS
		#define CSVA_LIB_API __attribute__((visibility("default")))
	#else
		#define CSVA_LIB_API
	#endif
#else
//  do nothing and hope for the best?
		#define CSVA_LIB_API
	#pragma warning Unknown dynamic link import/export semantics.
#endif


#include <opencv2/core/core.hpp>
#include <array>
using namespace cv;
using namespace std;
namespace csva
{
	CSVA_LIB_API cv::Mat filter_matches(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2, const vector<DMatch>& matches,
		const Mat& im1, const  Mat& im2, int mode,
		int type, vector<DMatch> &inliers,
		double* confidence, double LoweProb);
	
	CSVA_LIB_API void primary_filtering(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2,
		const vector<DMatch>& matches, float NNthresh, vector<DMatch> &inliers);

	CSVA_LIB_API std::array<double, 6> confidence_estimation(vector<DMatch>& inliers, const Mat& PT, vector<KeyPoint> kpts1, vector<KeyPoint> kpts2,
		const vector<DMatch> &excludedMatches, Mat im1, Mat im2, int mode, int type, double LoweProb);
	
	CSVA_LIB_API void verify_clusters(const vector< vector<DMatch> >& clusters, vector< vector<DMatch> >& filtered,
		vector<Mat>& transforms, const vector<KeyPoint>& kpts1,
		const vector<KeyPoint>& kpts2, const Mat& image1, const Mat& image2);

	//CSVA_LIB_API void hough_transform(vector<KeyPoint> kpts1, vector<KeyPoint> kpts2, vector<DMatch> matches,
	//	Mat image1, Mat image2, vector<vector<DMatch> >& clusters, int vote_thresh = 0);
	//
	CSVA_LIB_API void hough_transform(const vector<KeyPoint>& kpts1, const vector<KeyPoint>& kpts2,
		const vector<DMatch>& matches, const Mat& image1, const Mat& image2,
		vector<vector<DMatch> >& clusters, int vote_thresh=0);

	//CSVA_LIB_API void hough_transform2(vector<KeyPoint> kpts1, vector<KeyPoint> kpts2, vector<DMatch> matches,
	//	Mat image1, Mat image2, vector<vector<DMatch> >& clusters, int vote_thresh = 0);
}