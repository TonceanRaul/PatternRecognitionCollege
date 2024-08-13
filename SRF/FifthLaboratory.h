#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"
class FifthLaboratory
{
public:
    FifthLaboratory();
    ~FifthLaboratory();
    Mat1b constructImage();
    std::vector<vfc::float32_t> computeMeanScanningRows(Mat1b f_matrix);
    void computeCovarianceMatrix(Mat1b f_intensity);
    void computeCorelationCoeff(Mat1b f_covariance);
    std::vector<Mat1b> computeMatForGraphics(Mat1b f_matrix); 
    void displayGraphics(std::vector<Mat1b> f_matrixVector);
    void FifthLaboratory::gaussianDensity(Mat1b f_matrix);
};

