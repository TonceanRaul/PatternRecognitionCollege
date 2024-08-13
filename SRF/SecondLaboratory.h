#pragma once

#include "OpenCVApplication.h"
#include "stdafx.h"

class SecondLaboratory
{
public:


public:
    std::vector<std::pair<int, int>> findPositionOfBlackPoints
    (
        const cv::Mat1b&   f_grayScaleMatrix
    );

    std::tuple<float, float, float> ransacAlgorithm
    (
        const std::vector<std::pair<int, int>>& f_blackPoints
    );

};
