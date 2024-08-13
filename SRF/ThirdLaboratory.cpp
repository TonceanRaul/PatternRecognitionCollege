#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "ThirdLaboratory.h"


ThirdLaboratory::ThirdLaboratory()
{
} 


ThirdLaboratory::~ThirdLaboratory()
{
}

std::vector<std::pair<int, int>> ThirdLaboratory::whitePointsVector(const Mat1b& f_matrix)
{
    std::vector<std::pair<int, int>> l_whitePointsVector;

    for (vfc::int32_t i = 0; i < f_matrix.rows; ++i)
    {
        for (vfc::int32_t j = 0; j < f_matrix.cols; ++j)
        {
            if (f_matrix(i, j) == 255)
            {
                l_whitePointsVector.push_back(std::make_pair(j, i));
            }
        }
    }

    return l_whitePointsVector;
}


bool ThirdLaboratory::isInWindow(int i, int j, int rows, int cols)
{
    return i >= 0 && j >= 0 && i < rows && j < cols;
}

void ThirdLaboratory::houghSinus(const cv::Mat& Hough)
{
    int maxHough = 0;

    for (int i = 0; i < Hough.rows; ++i)
    {
        for (int j = 0; j < Hough.cols; ++j)
        {
            if (Hough.at<int>(i, j) > maxHough)
            {
                maxHough = Hough.at<int>(i, j);
            }
        }
    }
    Mat houghImg;
    Hough.convertTo(houghImg, CV_8UC1, 255.f / (float)maxHough);
    imshow("Hough sinus", houghImg);
}

Mat ThirdLaboratory::computeHoughMatrix(const Mat1b& l_grayScaleMatrix)
{
    std::vector<std::pair<int, int>> l_whitePointsVector;
    l_whitePointsVector = whitePointsVector(l_grayScaleMatrix);

    vfc::float32_t l_diagonal = sqrt(l_grayScaleMatrix.rows * l_grayScaleMatrix.rows +
        l_grayScaleMatrix.cols * l_grayScaleMatrix.cols);
    l_diagonal += 1;
    Mat Hough(360, l_diagonal, CV_32SC1);
    Hough.setTo(0);

    for (vfc::int32_t i = 0; i < l_whitePointsVector.size(); ++i)
    {
        for (int theta = 0; theta < 360; ++theta)
        {
            vfc::float32_t rad = (theta * CV_PI) / 180;
            vfc::float32_t l_ro = l_whitePointsVector[i].first  * cos(rad) +
                l_whitePointsVector[i].second * sin(rad);
            if (l_ro >= 0 && l_ro <= l_diagonal)
            {
                Hough.at<int>(theta, l_ro)++;
            }
        }
    }

    return Hough;
}

std::vector<ThirdLaboratory::peak> ThirdLaboratory::computePeakVector(const Mat& Hough, const Mat& l_grayScaleMatrix)
{
    vfc::int32_t nWindow = 3;
    std::vector<ThirdLaboratory::peak> l_peakVector;
    vfc::int32_t kernelSize_i32 = nWindow - 1;
    for (int i = 0; i < Hough.rows; ++i)
    {
        for (int j = 0; j < Hough.cols; ++j)
        {
            int max = Hough.at<int>(i, j);
            for (int k1 = -kernelSize_i32 / 2; k1 <= kernelSize_i32 / 2; ++k1)
            {
                for (int k2 = -kernelSize_i32 / 2; k2 <= kernelSize_i32 / 2; ++k2)
                {
                    if (isInWindow(i + k1, j + k2, l_grayScaleMatrix.rows, l_grayScaleMatrix.cols))
                    {
                        if (Hough.at<int>(i + k1, j + k2) > max)
                        {
                            max = Hough.at<int>(i + k1, j + k2);
                        }
                    }
                }
            }

            if (max == Hough.at<int>(i, j))
            {
                peak l_point;
                l_point.theta = i;
                l_point.ro = j;
                l_point.hval = max;
                l_peakVector.push_back(l_point);
            }
        }
    }

    std::sort(l_peakVector.begin(), l_peakVector.end());
    return l_peakVector;
}