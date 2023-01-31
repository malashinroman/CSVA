//#include "misc.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"

#include "time.h"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/algorithm/minmax.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <iostream>
#include "feature_extractors.h"

int findMatch(cv::DMatch m, std::vector<cv::DMatch> allmatches, int crossCheck)
{
    for (int i = 0; i < allmatches.size(); i++)
    {
        if (crossCheck)
        {
            int quer1 = allmatches.at(i).queryIdx;
            int quer2 = m.trainIdx;
            int train1 = allmatches.at(i).trainIdx;
            int train2 = m.queryIdx;
            if ((allmatches.at(i).trainIdx == m.queryIdx) && (allmatches.at(i).queryIdx == m.trainIdx))
            {
                return i;
            }
            if ((train1 == train2) && (quer1 == quer2))
            {
                int justflag = 0;
                //return i;
            }
        }
        else
        {
            if ((allmatches.at(i).queryIdx == m.queryIdx) && (allmatches.at(i).trainIdx == m.trainIdx))
            {
                return i;
            }
        }

    }
    return -1;
}

void getDenseKeypoints(cv::Mat image, std::vector<cv::KeyPoint>& keypoints)
{
    int imWidth = image.size().width;
    int imHeight = image.size().height;
    int KeyRes = imWidth > imHeight ? imHeight : imWidth;

    int startSize = 20;
    for(int s = startSize; s < KeyRes/4; s*=1.2)
    {
        for(int i = s / 2; i <imHeight - s/2; i += s/2)
        {
            for(int j = s / 2; j < imWidth - s/2; j += s/2)
            {
                cv::KeyPoint kp(cv::Point2f(j,i), s, 0.);
                keypoints.push_back(kp);
            }
        }
    }
}

std::vector<cv::KeyPoint> OpenCVfeatures::getKeyPoints(cv::Mat image, int type, int ConsoleOutput)
{
    std::vector<cv::KeyPoint> points;

    if(type % 10 == AGAST_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("AGAST_DETECTOR_TYPE\n");
        this->detectorAgast->detect( image, points);
    }
    if(type % 10 == SIFT_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("SIFT_DETECTOR_TYPE\n");
        this->detectorSift->detect( image, points);
    }
    if(type % 10 == SURF_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("SURF_DETECTOR_TYPE\n");
        this->detectorSurf->detect(image, points);
    }
    if(type % 10 == FAST_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("FAST_DETECTOR_TYPE\n");
        this->detectorFast->detect(image, points );
    }
    if(type % 10 == AKAZE_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("AKAZE_DETECTOR_TYPE\n");
        this->detectorAkaze->detect(image, points);
    }
    if(type % 10 == MSER_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("MSER_DETECTOR_TYPE\n");
        this->detectorMser->detect(image, points );
    }
    if(type % 10 == STAR_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("STAR_DETECTOR_TYPE\n");
        this->detectorStar->detect(image, points );
    }

    if(type % 10 == SIMPLEBLOB_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("SimpleBlob_DETECTOR_TYPE\n");
        this->detectorSimpleBlob->detect(image, points);
    }
    if(type % 10 == GOODFEAT_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("FOOD_FEATURES_DETECTOR_TYPE\n");
        this->detectorGoodFeat->detect(image, points);
    }
    if(type % 10 == ORB_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("ORB_DETECTOR_TYPE\n");
        this->detectorOrb->detect(image, points);
    }
    if(type % 10 == DENSE_DETECTOR_TYPE)
    {
        printf("DENSE_DETECTOR\n");
        getDenseKeypoints(image, points);
    }

    if(type % 10 == BRISK_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("BRISK_DETECTOR\n");
        this->detectorBrisk->detect(image,points);
    }
    if(type % 10 == DAISY_DETECTOR_TYPE)
    {
        if(ConsoleOutput)
            printf("DAISY_DETECTOR_TYPE\n");
        this->detectorDaisy->detect(image,points);
    }
    if ((type % 100) / 10 == VGG_DETECTOR_TYPE / 10)
    {
        if (ConsoleOutput)
            printf("VGG_DESCRIPTOR_TYPE\n");
        this->vggfeatures->detect(image, points);
    }
    return points;
}

cv::Mat OpenCVfeatures::getDescriptors(cv::Mat image, std::vector<cv::KeyPoint>& points, int type, int ConsoleOutput)
{
    cv::Mat descriptors;
    if( (type % 100) / 10 == SIFT_DESCRIPTOR_TYPE/10 )
    {
        if(ConsoleOutput)
            printf("SIFT_DESCRIPTOR_TYPE\n");
        this->extractorSift->compute( image, points, descriptors );
    }
    if( (type % 100) / 10 == SURF_DESCRIPTOR_TYPE / 10 )
    {
        if(ConsoleOutput)
            printf("SURF_DESCRIPTOR_TYPE\n");
        this->extractorSurf->compute( image, points, descriptors );
    }
    if( (type % 100) / 10 == AKAZE_DESCRIPTOR_TYPE / 10 )
    {
        if(ConsoleOutput)
            printf("AKAZE_DESCRIPTOR_TYPE\n");
        this->extractorAkaze->compute(image, points, descriptors);
    }
    if( (type % 100) / 10 == BRIEF_DESCRIPTOR_TYPE / 10 )
    {
        if(ConsoleOutput)
            printf("BRIEF_DESCRIPTOR_TYPE\n");
        this->extractorBrief->compute( image, points, descriptors );
    }
    if( (type % 100) / 10 == BRISK_DESCRIPTOR_TYPE / 10 )
    {
        if(ConsoleOutput)
            printf("BRISK_DESCRIPTOR\n");
        this->extractorBrisk->compute(image, points, descriptors);
    }
    if( (type % 100) / 10 == LUCID_DESCRIPTOR_TYPE / 10)
    {
        if(ConsoleOutput)
            printf("LUCID_DESCRIPTOR_TYPE\n");
        this->extractorLucid->compute(image, points, descriptors);
    }
    if( (type % 100) / 10 == ORB_DESCRIPTOR_TYPE / 10 )
    {
        if(ConsoleOutput)
            printf("ORB_DESCRIPTOR_TYPE\n");
        this->extractorOrb->compute(image, points, descriptors);
    }
    if( (type % 100) / 10 == FREAK_DESCRIPTOR_TYPE / 10 )
    {
        if(ConsoleOutput)
            printf("FREAK_DESCRIPTOR_TYPE\n");
        this->extractorFreak->compute(image, points, descriptors);
    }
    if( (type % 100) / 10 == DAISY_DESCRIPTOR_TYPE / 10 )
    {
        if(ConsoleOutput)
            printf("DAISY_DESCRIPTOR_TYPE\n");
        this->extractorDaisy->compute(image, descriptors);
    }
    if( (type % 100) / 10 == LATCH_DESCRIPTOR_TYPE / 10 )
    {
        printf("LATCH_DESCRIPTOR_TYPE\n");
        this->extractorLatch->compute(image, points, descriptors);
    }
    if ((type % 100) / 10 == VGG_DESCRIPTOR_TYPE / 10)
    {
        printf("VGG_DESCRIPTOR_TYPE\n");
        this->vggfeatures->compute(image, points, descriptors);
    }
    //std::cout<<"descriptor size = "<<descriptors.size()<<std::endl;
    return descriptors;
}

std::vector<cv::DMatch> OpenCVfeatures::getMatches(cv::Mat descs1, cv::Mat descs2, int type, int ConsoleOutput)
{
    std::vector<cv::DMatch> matches;
    if((type % 1000) / 100 == FLANN_MATCHER_TYPE / 100)
    {
        if(ConsoleOutput)
        {
            printf("FLANN_BASED_MATCHER_TYPE\n");
        }
        this->matcher1.match(descs1, descs2, matches);
    }
    /*if( (type % 1000) / 100 == BF_MATCHER_TYPE / 100 )
        {
            if(ConsoleOutput)
            {
                printf("BF_MATCHER_TYPE\n");
            }
            matcher2.match(descs1, descs2, matches);
        }*/
    if( (type % 1000) / 100 == HAMMING_MATCHER / 100 )
    {
        if(ConsoleOutput)
        {
            printf("HAMMING_NORM_MATCHER_TYPE\n");
        }
        //matcher3->match(descs1, descs2, matches);
        std::vector<std::vector<cv::DMatch>> couplemathces;
        this->matcher3->knnMatch(descs1, descs2, couplemathces, 2);

        for(int i = 0; i < descs1.size().height; i++)
        {
            cv::DMatch m1 = couplemathces.at(i).at(0);
            cv::DMatch m2 = couplemathces.at(i).at(1);
            //assert(m1.queryIdx == m2.queryIdx);
            //assert(m1.trainIdx == m2.trainIdx);
            m1.distance = m1.distance / m2.distance;
            matches.push_back(m1);
        }
    }
    if(((type % 1000) / 100 == CROSSCHECK_MATCHER_TYPE / 100))
    {
        if(ConsoleOutput)
        {
            printf("CROSS_CHECK_MATCHER\n");
        }
        std::vector<cv::DMatch> matches2;
        std::vector<std::vector<cv::DMatch>> matches12;
        std::vector<std::vector<cv::DMatch>> matches21;
        std::vector<cv::DMatch> matches12_;
        std::vector<cv::DMatch> matches21_;
        this->matcher2.knnMatch(descs1, descs2, matches12, 2);
        this->matcher2.knnMatch(descs2, descs1, matches21, 2);
        for(int i = 0; i < matches12.size(); i++)
        {
            cv::DMatch m1 = matches12[i][0];
            matches12_.push_back(m1);
        }
        for(int i = 0; i < matches21.size(); i++)
        {
            cv::DMatch m1 = matches21[i][0];
            matches21_.push_back(m1);
        }
        for(int i = 0; i < matches12.size(); i++)
        {
            cv::DMatch m = matches12[i][0];
            int indx = findMatch(m, matches21_, 1);
            if(indx < 0)
            {
                continue;
            }
            cv::DMatch m1a = matches12[i][0];
            const cv::DMatch &m2a = matches12[i][1];
            double ratio1 = m1a.distance / m2a.distance;
            cv::DMatch m1b = matches21[indx][0];
            const cv::DMatch m2b = matches21[indx][1];

            double ratio2 = m1b.distance / m2b.distance;
            double ratio = ratio1 > ratio2 ? ratio1: ratio2;
            if(ratio1 > 1 || ratio2 > 1)
            {
                printf("ratio > 1!!!\n");
                //system("pause");
            }
            m1a.distance = ratio;
            matches.push_back(m1a);
        }
        //matcher_noCrossCheck.match(descs1, descs2, matches2);

        /*matcher_noCrossCheck.match(descs1, descs2, matches);
            matcher_noCrossCheck.match(descs2, descs1, matches2);*/

    }
    if( ((type % 1000) / 100 == KNN_MATCHER / 100)  || ((type % 1000) / 100 == BF_MATCHER_TYPE / 100))
    {
        //double nndrRatio = 0.81;
        if(ConsoleOutput)
        {
            printf("KNN_MATCHER\n");
        }
        std::vector<std::vector<cv::DMatch>> matches_tmp;
        int nn_num = 2;
        this->matcher2.knnMatch(descs1, descs2, matches_tmp, 2);
        for (size_t i = 0; i < matches_tmp.size(); ++i)
        {
            if (matches_tmp[i].size() < 2)
                continue;

            cv::DMatch &m1 = matches_tmp[i][0];
            const cv::DMatch &m2 = matches_tmp[i][1];
            double ratio = m1.distance / m2.distance;
            //if(m1.distance <= nndrRatio * m2.distance)
            m1.distance = ratio;
            matches.push_back(m1);
        }
    }
    return matches;
}

cv::Mat vocabulary;
static cv::Ptr<cv::BOWImgDescriptorExtractor> bowide;

std::vector<cv::DMatch> generateMatches(std::vector<std::vector<int>> pointIndexes1, std::vector<std::vector<int>> pointIndexes2, int threshold)
{
    std::vector<cv::DMatch> matches;
    for(int i = 0; i <  pointIndexes1.size(); i++)
    {
        /*FIXME: what number is enough*/
        if(pointIndexes1.at(i).size() > threshold)
            continue;
        if(pointIndexes2.at(i).size() > threshold)
            continue;
        for(int j =0; j < pointIndexes1.at(i).size(); j++)
        {
            for(int k = 0; k < pointIndexes2.at(i).size(); k++)
            {
                cv::DMatch m(pointIndexes1.at(i).at(j), pointIndexes2.at(i).at(k), 0.1);
                matches.push_back(m);
            }
        }
    }
    return matches;
}

std::vector<cv::DMatch> getBowMatches(std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, cv::Mat image1, cv::Mat image2, int type, int threshold)
{
    if(bowide.empty())
    {
        printf("Bowide Init\n");
        //initVocabulary(type);
    }
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    std::vector<std::vector<int>> pointIndexes1;
    std::vector<std::vector<int>> pointIndexes2;
    bowide->compute(image1, keypoints1, descriptors1, &pointIndexes1);
    bowide->compute(image2, keypoints2, descriptors2, &pointIndexes2);
    printf("computed\n");
    std::vector<cv::DMatch> matches;
    for(int i = 0; i < pointIndexes1.size(); i++)
    {
        if(pointIndexes1.at(i).size() > threshold)
            continue;
        if(pointIndexes2.at(i).size() > threshold)
            continue;

        for(int j =0; j < pointIndexes1.at(i).size(); j++)
        {
            for(int k = 0; k < pointIndexes2.at(i).size(); k++)
            {
                cv::DMatch m(pointIndexes1.at(i).at(j), pointIndexes2.at(i).at(k), 0.1);
                matches.push_back(m);
            }
        }
    }
    return matches;
}

std::vector<cv::DMatch> useNNratio(std::vector<cv::DMatch> matches, double ratio)
{
    std::vector<cv::DMatch> outMatches;
    for(int i = 0; i < matches.size(); i++)
    {
        cv::DMatch m = matches.at(i);
        if(m.distance > 1)
            printf("raio > 1!!!\n");
        if(m.distance < ratio)
            outMatches.push_back(m);
    }
    return outMatches;
}


///---- extended part ---
std::vector<cv::KeyPoint> OpenCVfeatures::refineNotUniqueKeypoints(std::vector<cv::KeyPoint> keypoints, cv::Mat image, int k, double nnr_thresh, int type)
{
    cv::Mat descs = OpenCVfeatures::getDescriptors(image, keypoints, type, 0);
    std::vector<std::vector<cv::DMatch>> couplemathces;
    cv::Mat mask;
    if ((type % 100) / 10 == BRISK_DESCRIPTOR_TYPE / 10)
        this->matcher3->knnMatch(descs, descs, couplemathces, k);
    else
        this->matcher2.knnMatch(descs, descs, couplemathces, k);

    std::vector<cv::KeyPoint> refinedKeypoints;
    for (int i = 0; i < descs.size().height; i++)
    {
        cv::DMatch m1 = couplemathces.at(i).at(1);
        cv::DMatch m2 = couplemathces.at(i).at(k - 1);
        double nnr = m1.distance / m2.distance;
        if (nnr < nnr_thresh)
            refinedKeypoints.push_back(keypoints.at(i));
    }
    return refinedKeypoints;
}

void OpenCVfeatures::initDetectorDescriptors()
{
    detectorSift = cv::SIFT::create();
    detectorSurf = cv::xfeatures2d::SURF::create(112);
    detectorFast = cv::FastFeatureDetector::create();
    detectorMser = cv::MSER::create(5, 30);
    detectorStar = cv::xfeatures2d::StarDetector::create();
    detectorAkaze = cv::AKAZE::create();
    detectorLatch = cv::xfeatures2d::LATCH::create();
    detectorBrisk = cv::BRISK::create();
    detectorOrb = cv::ORB::create();
    detectorAgast = cv::AgastFeatureDetector::create();
    detectorDaisy = cv::xfeatures2d::DAISY::create();
    extractorSift = cv::SIFT::create();
    extractorSurf = cv::xfeatures2d::SURF::create();
    extractorBrief = cv::xfeatures2d::BriefDescriptorExtractor::create();
    extractorOrb = cv::ORB::create();
    extractorAkaze = cv::AKAZE::create();
    extractorFreak = cv::xfeatures2d::FREAK::create();
    extractorLatch = cv::xfeatures2d::LATCH::create();
    extractorBrisk = cv::BRISK::create();
    extractorLucid = cv::xfeatures2d::LUCID::create(3, 2);
    extractorDaisy = cv::xfeatures2d::DAISY::create();
    matcher3 = cv::DescriptorMatcher::create("BruteForce-Hamming");
    vggfeatures = cv::xfeatures2d::VGG::create();
    globalInitializationDescriptors = true;
}

void OpenCVfeatures::releaseDetectorDescriptors()
{
    detectorSift.release();
    detectorSurf.release();
    detectorFast.release();
    detectorMser.release();
    detectorStar.release();
    detectorAkaze.release();
    detectorLatch.release();
    detectorBrisk.release();
    detectorOrb.release();
    detectorAgast.release();
    detectorDaisy.release();
    extractorSift.release();
    extractorSurf.release();
    extractorBrief.release();
    extractorOrb.release();
    extractorAkaze.release();
    extractorFreak.release();
    extractorLatch.release();
    extractorBrisk.release();
    extractorLucid.release();
    extractorDaisy.release();
}

OpenCVfeatures::OpenCVfeatures()
{
    this->initDetectorDescriptors();
}

OpenCVfeatures::~OpenCVfeatures()
{
    this->releaseDetectorDescriptors();
}

std::vector<cv::DMatch> OpenCVfeatures::getLocalPatchMatches2(cv::Mat image1, cv::Mat image2, std::vector<cv::KeyPoint>& points1, std::vector<cv::KeyPoint>& points2,
                                                          int type, int ConsoleOutput)
{
    cv::Mat descriptors_1, descriptors_2;
    clock_t start = clock();
    points1 = getKeyPoints(image1, type, ConsoleOutput);
    
    if(ConsoleOutput) printf("detection 1st image = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock();
    points2 = getKeyPoints(image2, type, ConsoleOutput);
    if(ConsoleOutput) printf("detection 2nd image = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock();
    descriptors_1 = getDescriptors(image1, points1, type, ConsoleOutput);
    if(ConsoleOutput) printf("description 1st image = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock();
    descriptors_2 = getDescriptors(image2, points2, type, ConsoleOutput);
    if(ConsoleOutput) printf("description 2nd image = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC); start = clock();
    std::vector<cv::DMatch> matches;
    if ((descriptors_1.size().height != 0) && descriptors_2.size().height != 0)
    {
        matches = getMatches(descriptors_1, descriptors_2, type, ConsoleOutput);
        if(ConsoleOutput)
            printf("matching time = %f s\n", ((float)(clock())-start) / CLOCKS_PER_SEC);
        start = clock();
    }
    return matches;
}
