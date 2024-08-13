#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "TenthLaboratory.h"


TenthLaboratory::TenthLaboratory()
{
}


TenthLaboratory::~TenthLaboratory()
{
}

std::vector<Mat> TenthLaboratory::computeParams(const char* filename)
{
    std::vector<Mat> trainSet;
    
    Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
    
    int redPointsCounter = 0;
    int bluePointsCounter = 0;
    std::vector<std::tuple<int, int, int>> red;
    std::vector<std::tuple<int, int, int>> blue;

    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            if (img.at<Vec3b>(i, j) == Vec3b(0, 0, 255))
            {
                redPointsCounter++;
                red.push_back(std::make_tuple(i, j, 1));
            }
            if (img.at<Vec3b>(i, j) == Vec3b(255, 0, 0))
            {
                bluePointsCounter++;
                blue.push_back(std::make_tuple(i, j, -1));
            }
        }
    }

    Mat redPoints(redPointsCounter, 3, CV_32FC1);
    Mat bluePoints(bluePointsCounter, 3, CV_32FC1);
    
    Mat X(redPointsCounter + bluePointsCounter, 3, CV_32FC1);
    Mat Y(redPointsCounter + bluePointsCounter, 1, CV_32FC1);

    int counter = 0;
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            if (img.at<Vec3b>(i, j) == Vec3b(0, 0, 255))
            {
                X.at<float>(counter, 0) = 1.0f;
                X.at<float>(counter, 1) = static_cast<float>(i);
                X.at<float>(counter, 2) = static_cast<float>(j);
                Y.at<float>(counter, 0) = 1.0f;
                counter++;
            }
            if (img.at<Vec3b>(i, j) == Vec3b(255, 0, 0))
            {
                X.at<float>(counter, 0) = 1.0f;
                X.at<float>(counter, 1) = static_cast<float>(i);
                X.at<float>(counter, 2) = static_cast<float>(j);
                Y.at<float>(counter, 0) = -1.0f;
                counter++;
            }
        }
    }

    trainSet.push_back(X);
    trainSet.push_back(Y);

    return trainSet;
}

void TenthLaboratory::onlinePerceptron(const std::vector<Mat>& params, const char* filename)
{
    Mat X = params.at(0);
    Mat Y = params.at(1);
    float eta = 0.00001;
    Mat w(1, 3, CV_32FC1);
    float eLimit = 0.00001;
    int max_iter = pow(10, 5);

    for (int i = 0; i < 3; ++i)
    {
        w.at<float>(0, i) = 0.1f;
    }
    
    for (int iter = 0; iter < max_iter; ++iter)
    {
        float E = 0.0f;
        for (int i = 0; i < X.rows; ++i)
        {
            float z = 0.0f;
            for (int j = 0; j < X.cols; ++j)
            {
                z += w.at<float>(0, j) * X.at<float>(i, j);
            }

            if (z * Y.at<float>(i, 0) < 0)
            {
                for (int j = 0; j < w.cols; ++j)
                {
                    w.at<float>(0, j) += eta * Y.at<float>(i, 0) * X.at<float>(i, j);
                }
                E++;
            }
        }

        E = E / X.rows;
        drawLine(w, filename);
        waitKey(25);
        if (E < eLimit)
        {
            break;
        }
    }

}

void TenthLaboratory::batchPerceptron(const std::vector<Mat>& params, const char* filename)
{
    Mat X = params.at(0);
    Mat Y = params.at(1);
    float eta = 0.00001;
    Mat w(1, 3, CV_32FC1);
    float eLimit = 0.00001;
    int max_iter = pow(10, 5);

    for (int i = 0; i < w.cols; ++i)
    {
        w.at<float>(0, i) = 0.1;
    }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        float E = 0.0f;
        float L = 0.0f;
        Mat deltaL = Mat::zeros(1, 3, CV_32FC1);

        for (int i = 0; i < X.rows; ++i)
        {
            float z = 0.0f;
            for (int j = 0; j < X.cols; ++j)
            {
                z += w.at<float>(0, j) * X.at<float>(i, j);
            }

            if (z * Y.at<float>(i, 0) <= 0)
            {
                E++;
                L -= Y.at<float>(i, 0) * z;
                for (int l = 0; l < deltaL.cols; ++l)
                {
                    deltaL.at<float>(0, l) -= Y.at<float>(i, 0) * X.at<float>(i, l);
                }
            }
        }
        
        E /= X.rows;
        L /= X.rows;
        
        for (int i = 0; i < deltaL.cols; ++i)
        {
            deltaL.at<float>(0, i) /= X.rows;
        }
        
        if (E < eLimit)
        {
            break;
        }

        for (int i = 0; i < w.cols; ++i)
        {
            w.at<float>(0, i) -= eta * deltaL.at<float>(0, i);
        }
        drawLine(w, filename);
        waitKey(250);
    }
}

void TenthLaboratory::drawLine(Mat parameters, const char* filename) {
    Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
    int x1, x2, y1, y2;
    int min_val = img.rows < img.cols ? img.rows : img.cols;

    float a = parameters.at<float>(0, 2) == 0.0 ? 0.0000001 : parameters.at<float>(0, 2);
    float b = parameters.at<float>(0, 1) == 0.0 ? 0.0000001 : parameters.at<float>(0, 1);
    float c = parameters.at<float>(0, 0);
    
    if (abs(-a / b) > 1) 
    {
        y1 = 0;
        y2 = min_val;
        x1 = -c / a;
        x2 = (-c - b * y2) / a;
    }
    else 
    {
        x1 = 0;
        x2 = min_val;
        y1 = -c / b;
        y2 = (-c - a * x2) / b;
    }

    line(img, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 0));
   // resizeImg(img, img, 200, true);
    imshow("Perceptron", img);
}

