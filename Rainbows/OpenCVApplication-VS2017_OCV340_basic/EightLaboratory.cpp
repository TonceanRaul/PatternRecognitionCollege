#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "EightLaboratory.h"

EightLaboratory::EightLaboratory()
{
}


EightLaboratory::~EightLaboratory()
{
}

int* EightLaboratory::calcHistogram(Mat img3Channels, int nr_bins)
{
    int d = 3;
    int* hist = (int*)malloc(d * nr_bins * sizeof(int));
    memset(hist, 0, 3 * nr_bins * sizeof(int));

    cvtColor(img3Channels, img3Channels, CV_BGR2HSV);

    int m = 256 / nr_bins;
    for (int i = 0; i < img3Channels.rows; ++i)
    {
        for (int j = 0; j < img3Channels.cols; ++j)
        {
            Vec3b currentPixel = img3Channels.at<Vec3b>(i, j);
            hist[currentPixel[0] / m]++;
            hist[currentPixel[1] / m + nr_bins]++;
            hist[currentPixel[2] / m + 2 * nr_bins]++;
        }
    }

    return hist;
}

std::vector<Mat> EightLaboratory::computeTrainSets(int nr_bins)
{
    std::vector<Mat> trainSets;
    char fname[MAX_PATH];

    int feature_dim = 256;
    int nrinst = 6;
    // Mat C(nrclasses, nrclasses, CV_8UC1);

    int d = 3; // nr canale
    Mat trainMatrix(672, d * nr_bins, CV_32SC1);
    Mat labelMatrix(672, 1, CV_8UC1); // fiecare valoare de la 0-5 indica daca e beach, snow etc
    int index = 0;

    for (int i = 0; i < nrinst; ++i)
    {
        for (int j = 0; j < dimensions[i]; ++j)
        {
            sprintf(fname, "prs_res_KNN/train/%s/%06d.jpeg", classes[i], j);
            Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);

            int* histogram = (int*)malloc(3 * nr_bins * sizeof(int));
            memset(histogram, 0, 3 * nr_bins * sizeof(int));

            histogram = calcHistogram(img, nr_bins);

            for (int h = 0; h < d * nr_bins; ++h)
            {
                trainMatrix.at<int>(index, h) = histogram[h];
            }

            labelMatrix.at<uchar>(index, 0) = i;
            index++;
        }
    }

    trainSets.push_back(trainMatrix);
    trainSets.push_back(labelMatrix);

    return trainSets;
}

struct structureBecauseFuckThisLaboratory
{
    int i;
    float distance;
    bool operator < (const structureBecauseFuckThisLaboratory& o) const 
    { 
        return distance < o.distance; 
    }
} structureBecauseFuckThisLaboratorySecondTime;

int EightLaboratory::kNNAlgorithm(std::vector<Mat> trainSet, Mat img, int k, int nr_bins)
{
    int* hist = (int*)malloc(3 * nr_bins * sizeof(int));
    memset(hist, 0, 3 * nr_bins * sizeof(int));
    hist = calcHistogram(img, nr_bins);

    Mat trainMatrix = trainSet.at(0);
    Mat labelsMatrix = trainSet.at(1);

    std::vector<structureBecauseFuckThisLaboratory> distanceWithCorrespondingIndex;
    structureBecauseFuckThisLaboratory distanceStructure;

    for (int i = 0; i < trainMatrix.rows; ++i)
    {
        float distance = 0.0f;
        for (int j = 0; j < trainMatrix.cols; ++j)
        {
            distance += (trainMatrix.at<int>(i, j) - hist[j]) * (trainMatrix.at<int>(i, j) - hist[j]);
        }
        
        distanceStructure.distance = sqrt(distance);
        distanceStructure.i = i;
        distanceWithCorrespondingIndex.push_back(distanceStructure);

    }
    
    Mat label = Mat::zeros(nrclasses, 1, CV_32FC1);
    std::sort(distanceWithCorrespondingIndex.begin(), distanceWithCorrespondingIndex.end());

    for (int i = 0; i < k; ++i)
    {
        label.at<float>(labelsMatrix.at<uchar>(distanceWithCorrespondingIndex.at(i).i, 0), 0) += 1 / (distanceWithCorrespondingIndex.at(i).distance + 1);
    }

    float max = label.at<float>(0, 0);
    int index = 0;
    for (int i = 1; i < nrclasses; ++i)
    {
        if (label.at<float>(i, 0) > max)
        {
            max = label.at<float>(i, 0);
            index = i;
        }
    } 

    return index;
} 

Mat EightLaboratory::computeClassifierMatrix(std::vector<Mat> trainSet, int k, int nr_bins, int type)
{
    Mat trainMatrix = trainSet.at(0);
    Mat labelsMatrix = trainSet.at(1);

    Mat confusionMatrix = Mat::zeros(nrclasses, nrclasses, CV_32SC1);
    char fname[256] = "prs_res_KNN/test";
    int theFuckingLabel = 0;

    if (type == 0) // matricea de confuzie
    {
        for (int i = 0; i < nrclasses; ++i)
        {
            int fileNr = 0;
            while (1)
            {
                sprintf(fname, "prs_res_KNN/test/%s/%06d.jpeg", classes[i], fileNr++);
                Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);

                if (img.cols == 0)
                {
                    break;
                }

                theFuckingLabel = kNNAlgorithm(trainSet, img, k, nr_bins);
                confusionMatrix.at<int>(theFuckingLabel, i)++;
            }
        }
    }
    else // acuratetea
    {

    }
    return confusionMatrix;
}

float EightLaboratory::computeAccuracy(const Mat& kNN)
{ 
    float numitor = 0;
    float numarator = 0;

    for (int i = 0; i < kNN.rows; ++i)
    {
        for (int j = 0; j < kNN.cols; ++j)
        {
            if (i == j)
            {
                numarator += kNN.at<int>(i, i);
            }
            numitor += kNN.at<int>(i, j);
        }
    }

    return numarator / numitor;
}
