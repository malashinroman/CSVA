#pragma once
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"

std::vector<cv::DMatch> call3D(char* image1_path, char* image2_path, cv::Mat* matchresult, std::vector<cv::KeyPoint>& points1, std::vector<cv::KeyPoint>& points2, int LocalMatchtype, int consoleOuput, int graphicalOutput);
