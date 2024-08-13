#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "FourthLaboratory.h"


FourthLaboratory::FourthLaboratory()
{
}


FourthLaboratory::~FourthLaboratory()
{
}


vfc::float32_t FourthLaboratory::computeEuclidianDistance(vfc::int32_t x1,
                                                          vfc::int32_t x2,
                                                          vfc::int32_t y1,
                                                          vfc::int32_t y2)
{
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

vfc::float32_t FourthLaboratory::computeChessDistance(vfc::int32_t x1,
                                                      vfc::int32_t x2,
                                                      vfc::int32_t y1,
                                                      vfc::int32_t y2)
{
    return max(x2 - x1, y2 - y1);
}

std::pair<int, int> FourthLaboratory::computeCenterOfMass(const Mat1b& f_image)
{
    float match = 0;
    int r = 0;
    int c = 0;
    for (int i = 0; i < f_image.rows; ++i)
    {
        for (int j = 0; j < f_image.cols; ++j)
        {
            if (0 == f_image(i, j))
            {
                match++;
                r += i;
                c += j;
            }
        }
    }

    return std::make_pair(c / match, r / match);
}

template<typename T>
void FourthLaboratory::normalize(T* a, T min, T max)
{
    if (*a < min)
    {
        *a = min;
    }
    else if (*a > max)
    {
        *a = max;
    }
}


Mat1b FourthLaboratory::compute2Scans(const Mat1b& l_countourMatrix)
{
    Mat1b dt = l_countourMatrix.clone();

    float wHV = 2;
    float wD = wHV * sqrt(2);

    int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    int weight[8] = { 0, 1, 0, 1, 1, 0, 1, 0 };

    for (int i = 1; i < dt.rows; ++i)
    {
        for (int j = 1; j < dt.cols - 1; ++j)
        {
            int minPixel = dt(i, j);
            for (int k = 0; k < 4; ++k)
            {
                int w = (weight[k] == 0 ? wD : wHV);

                uchar pixel = 0;

                if (dt(i + di[k], j + dj[k]) + w > 255)
                {
                    pixel = 255;
                }
                else if (dt(i + di[k], j + dj[k]) + w < 0)
                {
                    pixel = 0;
                }
                else
                {
                    pixel = dt(i + di[k], j + dj[k]) + w;
                }

                if (pixel < minPixel)
                {
                    minPixel = pixel;
                }
            }
            dt(i, j) = minPixel;
        }
    }

    for (int i = dt.rows - 2; i >= 0; --i)
    {
        for (int j = dt.cols - 2; j >= 1; --j)
        {
            int minPixel = dt(i, j);
            for (int k = 4; k < 8; ++k)
            {
                int w = (weight[k] == 0 ? 3 : 2);

                uchar pixel = 0;

                if (dt(i + di[k], j + dj[k]) + w > 255)
                {
                    pixel = 255;
                }
                else if (dt(i + di[k], j + dj[k]) + w < 0)
                {
                    pixel = 0;
                }
                else
                {
                    pixel = dt(i + di[k], j + dj[k]) + w;
                }

                if (pixel < minPixel)
                {
                    minPixel = pixel;
                }
            }
            dt(i, j) = minPixel;
        }
    }

    FILE* f = fopen("dest.txt", "w");
    for (int i = 0; i < dt.rows; ++i)
    {
        for (int j = 0; j < dt.cols; ++j)
        {
            fprintf(f, "%d ", dt(i, j));
        }
        fprintf(f, "\n");
    }

    return dt;
}

float FourthLaboratory::patternMatching(const Mat1b& f_templateObject, const Mat1b& f_leafDT)
{
    double match = 0;
    int numbers = 0;

    if (f_templateObject.size != f_leafDT.size) 
    {
        return 0;
    }

    for (int i = 0; i < f_leafDT.rows; ++i)
    {
        for (int j = 0; j < f_leafDT.cols; ++j)
        {
            if (0 == f_templateObject(i, j))
            {
                match += f_leafDT(i, j);
                numbers++;
            }
        }
    }
    return match /= numbers;
}
