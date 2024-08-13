#pragma once

#include "OpenCVApplication.h"
#include "stdafx.h"
class SixthLaboratory
{
public:

public:
    std::vector<Point2i> SixthLaboratory::getBlackPointsCoords(const Mat3b& f_bigMat);
    std::vector<Mat> computeParameters(int d, const char* path);
    Mat computeKmean(const int k, int d, const char* path, int T);
    Mat colorPoints(const int &nK, const std::uniform_int_distribution<int> &dist, std::default_random_engine &gen,
        int d, const char * path, const cv::Mat &points, const cv::Mat &clusters, const cv::Mat &labels);
    void computeDistances(const cv::Mat &points, const int &k, int d, const cv::Mat &clusters, cv::Mat &labels, bool &modified);
    void recomputeCenters(cv::Mat &clusters, const cv::Mat &points, const cv::Mat &labels, int d, int nK);
    void Voronoi(const char* path, int noClusters,int noIterations);

};

