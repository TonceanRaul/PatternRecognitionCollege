#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"

class EightLaboratory
{
public:
    EightLaboratory();
    ~EightLaboratory();
public:
    int* calcHistogram(Mat img3Channels, int nr_bins);
    std::vector<Mat> computeTrainSets(int nr_bins);
    int kNNAlgorithm(std::vector<Mat> trainSet, Mat img, int k, int nr_bins);
    Mat computeClassifierMatrix(std::vector<Mat> trainSet, int k, int nr_bins, int type);
    float computeAccuracy(const Mat& kNN);
    static const int nrclasses = 6;
    char classes[nrclasses][15] = { "beach", "city", "desert", "forest", "landscape", "snow" };
    int dimensions[6] = { 177, 67, 55, 71, 35, 267 };

};

