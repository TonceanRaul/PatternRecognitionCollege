#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "SecondLaboratory.h"


std::vector<std::pair<int, int>> SecondLaboratory::findPositionOfBlackPoints(const cv::Mat1b&   f_grayScaleMatrix)
{
    std::vector<std::pair<int, int>> l_blackPoints;

    for (vfc::int32_t i = 0; i < f_grayScaleMatrix.rows; ++i)
    {
        for (vfc::int32_t j = 0; j < f_grayScaleMatrix.cols; ++j)
        {
            if (f_grayScaleMatrix(i, j) == 0)
            {
                l_blackPoints.push_back(std::make_pair(i, j));
            }
        }
    }

    return l_blackPoints;
}

std::tuple<float, float, float> SecondLaboratory::ransacAlgorithm(const std::vector<std::pair<int, int>>& f_blackPoints)
{
    vfc::int32_t    t = 10;
    vfc::int32_t    s = 2;
    vfc::float32_t  p = 0.99;

    vfc::float32_t  q = 0.3;

    vfc::float32_t threshold = t / 3;

    vfc::float32_t N = log(1 - p) / log(1 - pow(q, s));
    vfc::float32_t T = q * f_blackPoints.size();

    std::default_random_engine l_gen(time(0));
    std::uniform_int_distribution<vfc::int32_t> l_d(0, f_blackPoints.size() - 1);

    vfc::int32_t noInliners = 0;
    vfc::int32_t counter = 0;
    vfc::int32_t max = INT_MIN;
    std::tuple<float, float, float> l_parameters;

    while (noInliners <= T || counter < N)
    {
        std::pair<int, int> firstPoint  = f_blackPoints.at(l_d(l_gen));
        std::pair<int, int> secondPoint = f_blackPoints.at(l_d(l_gen));
        noInliners                      = 0;

        float a = firstPoint.second - secondPoint.second;
        float b = secondPoint.first - firstPoint.first;
        float c = firstPoint.first  * secondPoint.second - secondPoint.first * firstPoint.second;

        for (const std::pair<int, int>& l_point : f_blackPoints)
        {
            float distance = abs(a * l_point.first + b * l_point.second + c) / (sqrt(a * a + b * b));

            if (distance < t)
            {
                noInliners++;
            }
        }

        if (max < noInliners)
        {
            max = noInliners;
            l_parameters = std::make_tuple(a, b, c);
        }
        counter++;
    }

    return l_parameters;

}
