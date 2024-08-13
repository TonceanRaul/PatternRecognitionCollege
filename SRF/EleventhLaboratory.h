#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"
#define MAXT 30

class EleventhLaboratory
{
public:
    EleventhLaboratory();
    ~EleventhLaboratory();

    struct weaklearner
    {
        int feature_i;
        int threshold;
        int class_label;
        float error;
        int classify(Mat X)
        {
            if (X.at<float>(feature_i) < threshold)
            {
                return class_label;
            }
            else
            {
                return -class_label;
            }
        }
    };

    struct classifier
    {
        int T;
        float alphas[MAXT];
        weaklearner hs[MAXT];
        int classify(Mat X)
        {
            float theSum = 0;
            for (int i = 0; i < MAXT; ++i) 
            {
                theSum += alphas[i] * hs[i].classify(X);
            }
            if (theSum >= 0)
            {
                return 1;
            }
            else
            {
                return -1;
            }
        }
    };

public:
    std::vector<cv::Mat> readAdaBoostPoints(const char* filename, int& dim);
    struct weaklearner findWeakLearner(const std::vector<Mat>& trainSet, int dim);
    struct classifier adaBoost(std::vector<Mat> trainSet, int dim);
    void drawBoundary(const char* filename, struct classifier adaBoostClassifier);

};

