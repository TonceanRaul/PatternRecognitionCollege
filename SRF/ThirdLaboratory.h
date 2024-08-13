#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"


class ThirdLaboratory
{
public:
    ThirdLaboratory();
    ~ThirdLaboratory();
    
    struct peak
    {
        int theta, ro, hval;
        bool operator <(const peak& o) const
        {
            return hval > o.hval;
        }
    };

public:
    static std::vector<std::pair<int, int>> whitePointsVector(const Mat1b& f_matrix);
    static bool isInWindow(int i, int j, int rows, int cols);
    void houghSinus(const cv::Mat& Hough);
    Mat computeHoughMatrix(const Mat1b& l_grayScaleMatrix);
    std::vector<peak> computePeakVector(const Mat& Hough, const Mat& l_grayScaleMatrix);

};

