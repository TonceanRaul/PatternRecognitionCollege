#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "NinthLaboratory.h"


NinthLaboratory::NinthLaboratory()
{
}

NinthLaboratory::~NinthLaboratory()
{
}

std::vector<cv::Mat> NinthLaboratory::computeBayesTrainSet()
{
    std::vector<Mat> trainSet;

    Mat trainMatrix(60000, 28 * 28, CV_8UC1);
    Mat labelMatrix(60000, 1, CV_32SC1);
    Mat indexMatrix(10, 1, CV_32SC1);
    
    char fname[MAX_PATH] = "prs_res_Bayes/train";
    int matrixIndex = 0;
    int classesNo = 10;
    int indexStarting = 0;

    for (int c = 0; c < 10; ++c)
    {
        int index = 0;
        while (1)
        {
            sprintf(fname, "prs_res_Bayes/train/%d/%06d.png", c, index);
            Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

            if (img.cols == 0)
            {
                indexMatrix.at<int>(c, 0) = index;
                indexStarting += index;
                break;
            }

            for (int i = 0; i < img.rows; ++i)
            {
                for (int j = 0; j < img.cols; ++j)
                {
                    if (img.at<uchar>(i, j) > 128)
                    {
                        trainMatrix.at<uchar>(index + indexStarting, i * img.cols + j) = 255;
                    }
                    else
                    {
                        trainMatrix.at<uchar>(index + indexStarting, i * img.cols + j) = 0;
                    }
                }
            }

            index++;
            indexMatrix.at<int>(c, 0) = matrixIndex;
            matrixIndex++;
        }
    }

    trainSet.push_back(trainMatrix);
    trainSet.push_back(labelMatrix);
    trainSet.push_back(indexMatrix);

    return trainSet;
}

Mat NinthLaboratory::computeAPriori(const std::vector<Mat>& trainSet)
{
    Mat trainMatrix = trainSet.at(0);
    Mat labelMatrix = trainSet.at(1);
    Mat indexMatrix = trainSet.at(2);

    int classesNo = 10;
    Mat priors(classesNo, 1, CV_32FC1);

    for (int i = 0; i < classesNo; ++i)
    {
        float counter = 0.0f;
        for (int j = 0; j < trainMatrix.rows; ++j)
        {
            if (labelMatrix.at<int>(j, 0) == i)
            {
                counter++;
            }
        }
        counter /= trainMatrix.rows;
        priors.at<float>(i, 0) = counter;
    }

    return priors;
}

Mat NinthLaboratory::computeLikelihood(const std::vector<Mat>& trainSet)
{
    Mat trainMatrix = trainSet.at(0);
    Mat labelMatrix = trainSet.at(1);
    Mat indexMatrix = trainSet.at(2);

    int indexStarting = 0;
    int classesNo = 10;

    Mat likelihood = Mat::zeros(classesNo, 28 * 28, CV_32FC1);

    for (int c = 0; c < classesNo; ++c)
    {
        for (int j = 0; j < trainMatrix.cols; ++j)
        {
            float counter = 0.0f;
            for (int k = 0; k < indexMatrix.at<int>(c, 0); ++k)
            {
                if (trainMatrix.at<uchar>(indexStarting + k, j) == 255)
                {
                    counter++;
                }
            }
            likelihood.at<float>(c, j) = (counter + 1) / (indexMatrix.at<int>(c, 0) + classesNo);
        }

        indexStarting += indexMatrix.at<int>(c, 0);
    }

    return likelihood;
}

int NinthLaboratory::naiveBayes(std::vector<Mat> trainSet, const Mat& img, const Mat& priors, const Mat& likelihood)
{
    Mat trainMatrix = trainSet.at(0);
    Mat labelMatrix = trainSet.at(1);
    Mat indexMatrix = trainSet.at(2);

    std::vector<float> probabilities;
    Mat t(1, 28 * 28, CV_8UC1);
    Mat posteriors(10, 1, CV_32FC1);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<uchar>(i, j) > 128)
            {
                t.at<uchar>(0, i * img.cols + j) = 255;
            }
            else
            {
                t.at<uchar>(0, i * img.cols + j) = 0;
            }
        }
    }

    for (int c = 0; c < 10; ++c)
    {
        float sum = 0.0;
        for (int i = 0; i < trainMatrix.cols; ++i)
        {
            if (t.at<uchar>(0, i) == 255)
            {
                sum += log(likelihood.at<float>(c, i));
            }
            else
            {
                sum += log(1.0 - likelihood.at<float>(c, i));
            }
        }

        posteriors.at<float>(c, 0) = priors.at<float>(c, 0) + sum;

    }

    double minVal, maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc(posteriors, &minVal, &maxVal, &minLoc, &maxLoc);
    return maxLoc.y;
}

Mat NinthLaboratory::computeBayesClassifier(const std::vector<Mat> trainSet, Mat priors, Mat likelihood)
{
    Mat confusionMatrix = Mat::zeros(10, 10, CV_32SC1);
    char fnameBayes[256] = "prs_res_Bayes/test";
    int theLabel = 0;
    for (int i = 0; i < 10; ++i)
    {
        int fileNr = 0;
        while (1)
        {
            sprintf(fnameBayes, "prs_res_Bayes/test/%d/%06d.png", i, fileNr++);
            Mat img = imread(fnameBayes, CV_LOAD_IMAGE_GRAYSCALE);

            if (img.cols == 0)
            {
                break;
            }

            theLabel = naiveBayes(trainSet, img, priors, likelihood);
            confusionMatrix.at<int>(theLabel, i)++;
        }
    }

    return confusionMatrix;
}
