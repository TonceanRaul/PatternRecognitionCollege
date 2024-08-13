#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"

class NinthLaboratory
{
public:
    NinthLaboratory();
    ~NinthLaboratory();

public:
    std::vector<cv::Mat> computeBayesTrainSet();
    Mat computeAPriori(const std::vector<Mat>& trainSet);
    Mat computeLikelihood(const std::vector<Mat>& trainSet);
    int naiveBayes(std::vector<Mat> trainSet, const Mat& img, const Mat& priors, const Mat& likelihood);
    Mat computeBayesClassifier(std::vector<Mat> trainSet, Mat priors, Mat likelihood);


};

