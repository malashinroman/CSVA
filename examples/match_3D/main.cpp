#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_lib/feature_extractors.h"
//#include "common_lib/DirectoryParser.h"
#include "csva_lib/csva.h"
#include "3Drecognition.h"
#include <time.h>
using namespace cv;
using namespace std;

Mat printMatches(vector<KeyPoint> pts1, vector<KeyPoint> pts2, vector<DMatch> matches, Mat image1, Mat image2, int flags)
{
	Mat outImage;
	drawMatches(image1, pts1, image2, pts2, matches, outImage, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0), vector<char>(), flags);
	return outImage;
}
void main(int argc, char* argv[])
{
	Mat result;
	vector<KeyPoint> kpts1;
	vector<KeyPoint> kpts2;
	Mat image1 = imread(argv[1]);
	Mat image2 = imread(argv[2]);
	vector<DMatch> inliers = call3D(argv[1], argv[2], &result, kpts1, kpts2, 352, 0, 0);
	result = printMatches(kpts1, kpts2, inliers, image1, image2, 
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}