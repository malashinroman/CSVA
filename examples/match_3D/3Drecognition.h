#pragma once
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

vector<DMatch> call3D(char* image1_path, char* image2_path, Mat* matchresult, vector<KeyPoint>& points1, vector<KeyPoint>& points2, int LocalMatchtype, int consoleOuput, int graphicalOutput);