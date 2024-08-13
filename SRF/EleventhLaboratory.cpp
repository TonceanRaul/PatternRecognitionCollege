#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "EleventhLaboratory.h"


EleventhLaboratory::EleventhLaboratory()
{
}


EleventhLaboratory::~EleventhLaboratory()
{
}

std::vector<cv::Mat> EleventhLaboratory::readAdaBoostPoints(const char* filename, int& dim)
{
    std::vector<cv::Mat> trainVector;
    Mat initialMat = imread(filename, CV_LOAD_IMAGE_COLOR);
    int counter = 0;
    for (int i = 0; i < initialMat.rows; ++i)
    {
        for (int j = 0; j < initialMat.cols; ++j)
        {
            if (initialMat.at<Vec3b>(i, j) == Vec3b(255, 0, 0) || 
                initialMat.at<Vec3b>(i, j) == Vec3b(0, 0, 255))
            {
                ++counter;
            }
        }
    }

    Mat X(counter, 2, CV_32FC1);
    Mat Y(counter, 1, CV_32FC1);
    Mat W(counter, 1, CV_32FC1);
    int C = 0;

    for (int i = 0; i < initialMat.rows; ++i)
    {
        for (int j = 0; j < initialMat.cols; ++j)
        {
            if (initialMat.at<Vec3b>(i, j) == Vec3b(255, 0, 0))
            {
                X.at<float>(C, 0) = static_cast<float>(j);
                X.at<float>(C, 1) = static_cast<float>(i);
                Y.at<float>(C, 0) = static_cast<float>(-1);
                C++;
            }
            if (initialMat.at<Vec3b>(i, j) == Vec3b(0, 0, 255))
            {
                X.at<float>(C, 0) = static_cast<float>(j);
                X.at<float>(C, 1) = static_cast<float>(i);
                Y.at<float>(C, 0) = static_cast<float>(1);
                C++;
            }
        }
    }

    for (int i = 0; i < counter; ++i)
    {
        W.at<float>(i, 0) = 1.0 / counter;
    }

    dim = initialMat.rows < initialMat.cols ? initialMat.rows : initialMat.cols;
    trainVector.push_back(X);
    trainVector.push_back(Y);
    trainVector.push_back(W);

    return trainVector;
}

struct EleventhLaboratory::weaklearner EleventhLaboratory::findWeakLearner(const std::vector<Mat>& trainSet, int dim)
{
    Mat X = trainSet.at(0);
    Mat Y = trainSet.at(1);
    Mat W = trainSet.at(2);

    int class_label[2] = { -1, 1 };
    struct EleventhLaboratory::weaklearner bestLearner = EleventhLaboratory::weaklearner();

    vfc::float32_t bestError = FLT_MAX;


    for (int j = 0; j < X.cols; ++j)
    {
        for (int threshold = 0; threshold < dim; ++threshold)
        {
            for (int classLabel = 0; classLabel < 2; ++classLabel)
            {
                float e = 0;

                for (int i = 0; i < X.rows; ++i)
                {
                    int Z = 0;

                    if (X.at<float>(i, j) < threshold)
                    {
                        Z = class_label[classLabel];
                    }
                    else
                    {
                        Z = -class_label[classLabel];
                    }

                    if (Z * Y.at<float>(i, 0) < 0)
                    {
                        e += W.at<float>(i, 0);
                    }
                }
                if (e < bestError)
                {
                    bestError = e;

                    bestLearner.feature_i   = j;
                    bestLearner.threshold   = threshold;
                    bestLearner.class_label = class_label[classLabel];
                    bestLearner.error       = e;
                }
            }
        }
    }

    return bestLearner;
}

void EleventhLaboratory::drawBoundary(const char*                           filename, 
                                      struct EleventhLaboratory::classifier adaBoostClassifier)
{
    Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
    Mat dst = img.clone();

    imshow("img", img);

    for (int i = 0; i < dst.rows; ++i)
    {
        for (int j = 0; j < dst.cols; ++j)
        {
            if (img.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
            {
                Mat aux(1, 2, CV_32FC1);
                aux.at<float>(0, 0) = j;
                aux.at<float>(0, 1) = i;
                int label = adaBoostClassifier.classify(aux);

                if (label == 1)
                {
                    dst.at<Vec3b>(i, j) = Vec3b(255, 255, 0);
                }
                else
                {
                    dst.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
                }
            }
        }
    }
    imshow("AdaBoost", dst);
}

struct EleventhLaboratory::classifier EleventhLaboratory::adaBoost(std::vector<Mat> trainSet, int dim)
{
    Mat X = trainSet.at(0);
    Mat Y = trainSet.at(1);
    Mat W = trainSet.at(2);

    struct EleventhLaboratory::classifier theClassifier = EleventhLaboratory::classifier();
    
    for (int t = 0; t < MAXT; ++t)
    {
        theClassifier.hs[t] = findWeakLearner(trainSet, dim);
        theClassifier.alphas[t] = 0.5 * log((1.0 - theClassifier.hs[t].error) / theClassifier.hs[t].error);
        float S = 0.0f;

        for (int i = 0; i < X.rows; ++i)
        {
            Mat aux(1, 2, CV_32FC1);
            aux.at<float>(0, 0) = X.at<float>(i, 0);
            aux.at<float>(0, 1) = X.at<float>(i, 1);

            W.at<float>(i, 0) = W.at<float>(i, 0) * cv::exp((-1.0) * theClassifier.alphas[t] * 
                                Y.at<float>(i, 0) * (theClassifier.hs[t].classify(aux))); 
            S += W.at<float>(i, 0);
        }

        for (int i = 0; i < X.rows; ++i)
        {
            W.at<float>(i, 0) = W.at<float>(i, 0) / S;
        } 
    }

    return theClassifier;
}

