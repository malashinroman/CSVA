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

#include "misc_functions.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/algorithm/minmax.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <iostream>


std::vector<cv::Point2f> WrapTransform(std::vector<cv::Point2f>& SamplePoints, const cv::Mat& trMatrix)
{
    std::vector<cv::Point2f> wrappedPoints;
    for(std::vector<cv::Point2f>::iterator it = SamplePoints.begin(); it!= SamplePoints.end(); it++)
    {
        wrappedPoints.push_back(WrapTransform(*it, trMatrix));
    }
    return wrappedPoints;
}

cv::Rect getImageProjBbx(cv::Mat image1, cv::Mat trM)
{
    std::vector<cv::Point2f> showPoints;
    showPoints.push_back(cv::Point(0, 0));
    showPoints.push_back(cv::Point(image1.size().width, 0));
    showPoints.push_back(cv::Point(image1.size().width, image1.size().height));
    showPoints.push_back(cv::Point(0, image1.size().height));
    showPoints.push_back(cv::Point(image1.size().width / 2, image1.size().height / 2));
    showPoints.push_back(cv::Point(image1.size().width / 2, 0));

    std::vector<cv::Point2f> drawPoints = WrapTransform(showPoints, trM);
    std::vector<float> xx;
    std::vector<float> yy;
	for (int i = 0; i < 4; i++)
	{
		xx.push_back(drawPoints.at(i).x);
		yy.push_back(drawPoints.at(i).y);
	}
    typedef std::vector<float>::const_iterator iterator;
    std::pair< iterator, iterator > resultX = boost::minmax_element(xx.begin(), xx.end());
	double xmin = *resultX.first;
	double xmax = *resultX.second;
    std::pair< iterator, iterator > resultY = boost::minmax_element(yy.begin(), yy.end());
	double ymin = *resultY.first;
	double ymax = *resultY.second;
    return cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
}

cv::Point2f WrapTransform(cv::Point2f SamplePoint, const cv::Mat& trMatrix)
{
    cv::Mat p1(3,1, CV_64F);
    p1.at<double>(0) = SamplePoint.x;
    p1.at<double>(1) = SamplePoint.y;
    p1.at<double>(2) = 1.;
    cv::Mat Result = trMatrix * p1;
    double r = 1;
    if(Result.size().height > 2)
        r = Result.at<double>(2);

    cv::Point2f np((float)(Result.at<double>(0) / r), (float)(Result.at<double>(1) / r));
    return np;
}

double euclideanDistacne(cv::Point2f p1, cv::Point2f p2)
{
    return sqrt((double)(p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

cv::Point2f predictModelPosition2(const cv::KeyPoint& point1, const cv::KeyPoint& point2, cv::Point2f ModelPoint)
{
    double dx = ModelPoint.x - point1.pt.x;
    double dy = ModelPoint.y - point1.pt.y;
    double ang1 = convertOpencvAngle2GoodAngle(point1.angle);
    double ang2 = convertOpencvAngle2GoodAngle(point2.angle);
    double ang = ang2 - ang1;
    ang = ang / 180. * CV_PI;
    cv::Point2f IP;
    double MutualScale = point2.size / point1.size;
    IP.x = (float)(dx*MutualScale);
    IP.y = (float)(dy*MutualScale);
    cv::Point2f modelLoc;
    modelLoc.x = point2.pt.x + (float)(cos(ang)*IP.x - sin(ang)*IP.y);
    modelLoc.y = point2.pt.y + (float)(sin(ang)*IP.x + cos(ang)*IP.y);
    return modelLoc;
}
/*
*	By knowing rotation, mutual scale and position of matched keyPoints we can describe
*	similarity transform and therefore predict position of any point by applying
*	this transform.
*	Matches of every cluster should predict nearly the same model position
*/
cv::Point2f predictModelPosition(const cv::KeyPoint& point1, const cv::KeyPoint& point2, cv::Point2f ModelPoint)
{
    cv::Point2f p1 = point1.pt;
    cv::Point2f p2 = point2.pt;
    double angle1 = convertOpencvAngle2GoodAngle(point1.angle);
    double angle2 = convertOpencvAngle2GoodAngle(point2.angle);

    /*
    * similarty transform has the following template
    *
    *
    * |u|	|S*cos(a) -S*sin(a)| |x| +  |tx|
    * |v|	|S*sin(a)  S*cos(a)| |y|	|ty|
    */
    double size1 = point1.size;
    double size2 = point2.size;

    double S = size2 / size1;
    double a = angle2 - angle1;
    a = a / 180. * CV_PI;

    /*let us assume interest point is in the center of coordinate system
    * In that coordinate system modelPosition has coordinates [x,y]
    */
    double x = (ModelPoint.x) - p1.x;
    double y = (ModelPoint.y) - p1.y;

    //then we can get translation parameters to convert to image coordinate system
    double tx = p2.x;
    double ty = p2.y;


    //we use negative angle sice we have left-nanded coordinate system (y points down)
    double u = x*S*cos(-a) - y*S*sin(-a) + tx;
    double v = x*S*sin(-a) + y*S*cos(-a) + ty;

    return cv::Point2d(u, v);
}

/*
*	opencv - 0-360 clockwise
*	good [0,360] counter clockwise
*/
double convertOpencvAngle2GoodAnglePositive(double angle_opencv)
{

    double goodAngle = 360 - angle_opencv;
    //goodAngle = goodAngle > 180 ? goodAngle - 360 : goodAngle;
    return goodAngle;
}
/*
*	opencv - 0-360 clockwise
*	good [-180,180] counter clockwise
*/
double convertOpencvAngle2GoodAngle(double angle_opencv)
{

    double goodAngle = 360 - angle_opencv;
    goodAngle = goodAngle > 180 ? goodAngle - 360 : goodAngle;
    return goodAngle;
}

int independentMatches(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, cv::Mat im1, cv::Mat im2)
{
    int size = matches.size();
    std::vector<cv::DMatch> copy(matches);
    for(unsigned int i = 0; i < copy.size(); i++)
    {
        cv::DMatch cmatch = copy.at(i);
        for(unsigned int j = i + 1; j < copy.size(); j++)
        {
            cv::DMatch checkm = copy.at(j);
            double dist = matchDistance(cmatch, checkm, pts1, pts2);
            if(dist < 10)
            {
                copy.erase(copy.begin() + j);
                j--;
                size--;
            }
        }

    }
    matches = std::vector<cv::DMatch>(copy);
    return copy.size();
}

bool checkMatchIn(std::vector<cv::DMatch> matches, cv::DMatch newm)
{
    bool retval = false;
    for(std::vector<cv::DMatch>::iterator it = matches.begin(); it!=matches.end(); it++)
    {
        if(it->queryIdx == newm.queryIdx && it->trainIdx == newm.trainIdx)
            retval = true;
    }
    return retval;
}

bool checkMatchIn(std::vector<cv::DMatch> matches, cv::DMatch newm, int& indx)
{
    bool retval = false;
    int num = 0;
    indx = -1;
    for(std::vector<cv::DMatch>::iterator it = matches.begin(); it!=matches.end(); it++)
    {
        if(it->queryIdx == newm.queryIdx && it->trainIdx == newm.trainIdx)
        {
            retval = true;
            indx = num;
        }
        num++;
    }
    return retval;
}


double inline matchDistance(const cv::DMatch& m1, const cv::DMatch& m2, const std::vector<cv::KeyPoint>& pts1, const std::vector<cv::KeyPoint>& pts2)
{
    cv::Point2f m1p1 = pts1.at(m1.queryIdx).pt;
    cv::Point2f m1p2 = pts2.at(m1.trainIdx).pt;

    cv::Point2f m2p1 = pts1.at(m2.queryIdx).pt;
    cv::Point2f m2p2 = pts2.at(m2.trainIdx).pt;

    double dist1 = euclideanDistacne(m1p1, m2p1);
    double dist2 = euclideanDistacne(m1p2, m2p2);
    return dist1 < dist2 ? dist1 : dist2;
}

void decomposeAffLutsiv(const cv::Mat&  transfMat, double* scale, double* theta, double* ascale, double* direction)
{
    double a1 = transfMat.at<double>(0,0);
    double a2 = transfMat.at<double>(0,1);
    double a5 = transfMat.at<double>(0,2);

    double a3 = transfMat.at<double>(1,0);
    double a4 = transfMat.at<double>(1,1);
    double a6 = transfMat.at<double>(1,2);

    double c = (-a2 + a3);
    double z = (a1 + a4);
    double Beta = atan2(c,z);

    double Alpha = 0.5*(atan2(a2 + a3, a1 - a4) - Beta);

    double M = 0.5*( (a1+a4) / cos(Beta) - (a1-a4) / cos(2*Alpha+Beta) );
    double Mu = (a1 + a4) / (M*cos(Beta)) -1;
    *scale = M;
    *theta = Beta * 180 / CV_PI;
    *ascale = Mu;
    *direction = Alpha * 180 / CV_PI;
}


void decomposeAff(const cv::Mat& transfMat, cv::Mat& Rot, cv::Mat& Shear, cv::Mat& Scale, double& Theta, double& shiftX, double& shiftY, double& scale, double& p, double& r)
{
    double a = transfMat.at<double>(0,0);
    double b = transfMat.at<double>(0,1);
    double c = transfMat.at<double>(0,2);

    double d = transfMat.at<double>(1,0);
    double e = transfMat.at<double>(1,1);
    double f = transfMat.at<double>(1,2);

    Rot = cv::Mat::zeros(cv::Size(2,2), CV_64F);
    Shear = cv::Mat::zeros(cv::Size(2,2), CV_64F);
    Scale = cv::Mat::zeros(cv::Size(2,2), CV_64F);

    shiftX = c;
    shiftY = f;
    Theta = atan2(b, a);
    scale = (a*d + b*e) / (a*e - b*d);
    p = sqrt(a*a + b*b);
    r = (a*e - b*d) / p;
    scale = (p+r) / 2;
    Rot.at<double>(0,0) = cos(Theta);
    Rot.at<double>(0,1) = -sin(Theta);
    Rot.at<double>(1,0) = sin(Theta);
    Rot.at<double>(1,1) = cos(Theta);

    Shear.at<double>(0,0) = 1;
    Shear.at<double>(0,1) = (b*cos(Theta) + e*sin(Theta)) / (e * cos(Theta) - b* sin(Theta));
    Shear.at<double>(1,0) = 0;
    Shear.at<double>(1,1) = 1;

    Scale.at<double>(0,0) = sqrt(a*a + d*d);
    Scale.at<double>(0,1) = 0;
    Scale.at<double>(1,0) = 0;
    Scale.at<double>(1,1) = e*cos(Theta) - b*sin(Theta);
    Theta*=180 / CV_PI;
    Theta < 0 ? Theta + 360: Theta;
}

void getScaleAndRotation(const cv::Mat& transfMat, double& scale, double& angle)
{
    cv::Mat Scale, Shear, Rot;
    double shiftx, shifty, p, r;
    decomposeAff(transfMat, Rot, Shear, Scale, angle, shiftx, shifty, scale, p, r);
}

/*
*
*	[-180, 180]
*/
double getMutualAngle(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
{
    double a1 = convertOpencvAngle2GoodAngle(p1.angle);
    double a2 = convertOpencvAngle2GoodAngle(p2.angle);
    double dif = a2 - a1;
    double mangle = dif;
    if(dif > 180)
        mangle = mangle - 360;
    if(dif < -180)
        mangle = 360 + mangle;

    return mangle;
}

/*
*
*	[-180, 180]
*/
double getAngleDif(double angle1, double angle2)
{
	double dif = angle1 - angle2;
	if(dif > 180)
		dif = dif - 360;
	if(dif < -180)
		dif = 360 + dif;

	return dif;
}

double getMutualScale(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
{
    double size1 = p1.size;
    double size2 = p2.size;
	assert(size1 > 0);
	assert(size2 > 0);
    double ScaleK = size2 / size1;
    return ScaleK;
}

void getMutualShifts(const cv::KeyPoint& p1, const cv::KeyPoint& p2, double& shiftx, double& shifty)
{
    shiftx = p2.pt.x - p1.pt.x;
    shifty = p2.pt.y - p1.pt.y;
}

void sortMatchedKeypointsInQualityOrder(std::vector<cv::DMatch>&  matches, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::KeyPoint>& matchedkeypoints1, std::vector<cv::KeyPoint>& matchedkeypoints2)
{
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {return m1.distance < m2.distance; });
	return getMatchedKeypoints(matches, keypoints1, keypoints2, matchedkeypoints1, matchedkeypoints2);
}

void getMatchedKeypoints(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>&  keypoints1, const std::vector<cv::KeyPoint>&  keypoints2,
    std::vector<cv::KeyPoint>& matchedkeypoints1, std::vector<cv::KeyPoint>& matchedkeypoints2)
{
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		matchedkeypoints1.push_back(keypoints1.at(matches.at(i).queryIdx));
		matchedkeypoints2.push_back(keypoints2.at(matches.at(i).trainIdx));
	}
}
std::vector<cv::DMatch> excludeMany2OneMatches(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2)
{
    int* ind = (int*)malloc(sizeof(int) * keypoints2.size());

    cv::DMatch* matches_link = (cv::DMatch*)malloc(sizeof(cv::DMatch) * keypoints2.size());
    std::vector<cv::DMatch> excluded_matches;
    for(size_t i = 0; i < keypoints2.size(); i++)
    {
        ind[i] = -1;
    }
    for(size_t i = 0; i < matches.size(); i++)
    {
        int indx1 = matches.at(i).queryIdx;
        int indx2 = matches.at(i).trainIdx;
        if(ind[indx2] == -1)
        {
            ind[indx2] = indx1;
            matches_link[indx2] = matches.at(i);
        }
        else
        {
            excluded_matches.push_back(matches_link[indx2]);
            if(matches.at(i) < matches_link[indx2])
            {
                ind[indx2] = indx1;
                matches_link[indx2] = matches.at(i);
            }
        }
    }
    std::vector<cv::DMatch> newmatches;
    for(size_t i =0; i < keypoints2.size(); i++)
    {
        if(ind[i] > -1)
            newmatches.push_back(matches_link[i]);
    }
    free(ind);
    free(matches_link);
    return newmatches;
}

std::vector<cv::DMatch> excludeOne2ManyMatches(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2)
{
    int* ind = (int*)malloc(sizeof(int) * keypoints1.size());
    cv::DMatch* matches_link = (cv::DMatch*)malloc(sizeof(cv::DMatch) * keypoints1.size());
    std::vector<cv::DMatch> excluded_matches;
    for(size_t i = 0; i < keypoints1.size(); i++)
    {
        ind[i] = -1;
    }
    for(size_t i = 0; i < matches.size(); i++)
    {
        int indx1 = matches.at(i).queryIdx;
        int indx2 = matches.at(i).trainIdx;
        if(ind[indx1] == -1)
        {
            ind[indx1] = indx2;
            matches_link[indx1] = matches.at(i);
        }
        else
        {
            excluded_matches.push_back(matches_link[indx1]);
            if(matches.at(i) < matches_link[indx1])
            {
                ind[indx1] = indx2;
                matches_link[indx1] = matches.at(i);
            }
        }
    }
    std::vector<cv::DMatch> newmatches;
    for(size_t i =0; i < keypoints1.size(); i++)
    {
        if(ind[i] > -1)
            newmatches.push_back(matches_link[i]);
    }
    free(ind);
    free(matches_link);
    return newmatches;
}

int findMatch(const cv::DMatch& m, const std::vector<cv::DMatch>& allmatches, int crossCheck)
{
    for(size_t i = 0; i < allmatches.size(); i++)
    {
        if(crossCheck)
        {
            int quer1 = allmatches.at(i).queryIdx;
            int quer2 = m.trainIdx;
            int train1 = allmatches.at(i).trainIdx;
            int train2 = m.queryIdx;
            if((allmatches.at(i).trainIdx == m.queryIdx) && (allmatches.at(i).queryIdx == m.trainIdx))
                return i;

            if((train1 == train2) && (quer1 == quer2))
                int justflag = 0;
        }
        else
        {
            if((allmatches.at(i).queryIdx == m.queryIdx) && (allmatches.at(i).trainIdx == m.trainIdx))
                return i;
        }
    }
    return -1;
}

std::vector<cv::DMatch> useNNratio(const std::vector<cv::DMatch>& matches, double ratio)
{
    std::vector<cv::DMatch> outMatches;
    for(size_t i = 0; i < matches.size(); i++)
    {
        cv::DMatch m = matches.at(i);
        if(m.distance < ratio)
            outMatches.push_back(m);
    }
    return outMatches;
}

float angleBetweenLines(const cv::Point &v1, const cv::Point &v2)
{
	float len1 = float(sqrt(v1.x * v1.x + v1.y * v1.y));
	float len2 = float(sqrt(v2.x * v2.x + v2.y * v2.y));

	float dot = float(v1.x * v2.x + v1.y * v2.y);

	float a = dot / (len1 * len2);

	if (a >= 1.0)
		return 0.0;
	else if (a <= -1.0)
		return float(CV_PI);
	else
		return acos(a); // 0..PI
}

double calculateNewImageSquare(const cv::Size& OriginalSize, const cv::Mat& transform)
{
	float height =float( OriginalSize.height);
	float width = float(OriginalSize.width);
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(0, 0));
    points.push_back(cv::Point2f(width, 0));
    points.push_back(cv::Point2f(width, height));
    points.push_back(cv::Point2f(0, height));
    std::vector<cv::Point2f> wrapped_points = WrapTransform(points, transform);
	double d1 = euclideanDistacne(wrapped_points.at(0), wrapped_points.at(2));
	double d2 = euclideanDistacne(wrapped_points.at(1), wrapped_points.at(3));
    cv::Point v1 = wrapped_points.at(0) - wrapped_points.at(2);
    cv::Point v2 = wrapped_points.at(1) - wrapped_points.at(3);
	double a = angleBetweenLines(v1, v2);
	double S = 0.5 * sin(a) * d1 * d2;
	return S;
}

std::vector<cv::Point2f> getNewOutline_of_image(const cv::Mat& image, const cv::Mat& Tr)
{
	float height = float(image.size().height);
	float width = float(image.size().width);
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(0, 0));
    points.push_back(cv::Point2f(width, 0));
    points.push_back(cv::Point2f(width, height));
    points.push_back(cv::Point2f(0, height));
    std::vector<cv::Point2f> wrapped_points = WrapTransform(points, Tr);
	return wrapped_points;
}

cv::Mat AffineToHomography(cv::Mat affine)
{
    cv::Mat PT(3, 3, CV_64F);
	PT.at<double>(0, 0) = affine.at<double>(0, 0);
	PT.at<double>(0, 1) = affine.at<double>(0, 1);
	PT.at<double>(1, 0) = affine.at<double>(1, 0);
	PT.at<double>(1, 1) = affine.at<double>(1, 1);
	PT.at<double>(1, 2) = affine.at<double>(1, 2);
	PT.at<double>(0, 2) = affine.at<double>(0, 2);
	PT.at<double>(2, 0) = 0;
	PT.at<double>(2, 1) = 0;
	PT.at<double>(2, 2) = 1;
	return PT;
}
