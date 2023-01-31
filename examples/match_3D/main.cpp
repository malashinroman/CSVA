#include <time.h>
#include <opencv2/opencv.hpp>

#include "csva_lib/csva.h"
#include "3Drecognition.h"

///
cv::Mat printMatches(std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, std::vector<cv::DMatch> matches, cv::Mat image1, cv::Mat image2, cv::DrawMatchesFlags flags)
{
    cv::Mat outImage;
    cv::drawMatches(image1, pts1, image2, pts2, matches, outImage, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0), std::vector<char>(), flags);
	return outImage;
}

///
int main(int argc, char* argv[])
{
    cv::Mat result;
    std::vector<cv::KeyPoint> kpts1;
    std::vector<cv::KeyPoint> kpts2;
    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);
    std::vector<cv::DMatch> inliers = call3D(argv[1], argv[2], &result, kpts1, kpts2, 352, 0, 0);
    result = printMatches(kpts1, kpts2, inliers, image1, image2, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imwrite("result.jpg", result);
    return 0;
}
