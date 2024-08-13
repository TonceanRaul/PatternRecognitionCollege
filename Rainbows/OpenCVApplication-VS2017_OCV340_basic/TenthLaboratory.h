#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"
class TenthLaboratory
{
public:
    TenthLaboratory();
    ~TenthLaboratory();

public:
    std::vector<Mat> computeParams(const char* filename);
    void onlinePerceptron(const std::vector<Mat>& params, const char* filename);
    void batchPerceptron(const std::vector<Mat>& params, const char* filename);
    void drawLine(Mat parameters, const char* filename);

};

