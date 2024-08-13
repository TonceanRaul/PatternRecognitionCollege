#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"

class FourthLaboratory
{
public:
    FourthLaboratory();
    ~FourthLaboratory();
public:

    vfc::float32_t computeEuclidianDistance(vfc::int32_t                   x1, 
                                            vfc::int32_t                   x2, 
                                            vfc::int32_t                   y1, 
                                            vfc::int32_t                   y2);

    vfc::float32_t FourthLaboratory::computeChessDistance(vfc::int32_t     x1,
                                                          vfc::int32_t     x2,
                                                          vfc::int32_t     y1,
                                                          vfc::int32_t     y2);
    Mat1b compute2Scans(const Mat1b& l_countourMatrix);

    float patternMatching(const Mat1b& f_templateObject, const Mat1b& f_leafDT);
    std::pair<int, int> computeCenterOfMass(const Mat1b& f_image);

    template<typename T>
    void normalize(T* a, T min, T max);

};

