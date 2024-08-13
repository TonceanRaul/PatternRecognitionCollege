#pragma once
#include "OpenCVApplication.h"
#include "stdafx.h"

class FirstLaboratory
{
public:
    FirstLaboratory();
    ~FirstLaboratory();

public:

    void drawLineFirstApproach
    (
        const Mat&                                       f_matrix,
        const std::pair<float, float>&                   f_thetaPair
    );

    void drawLineSecondApproach
    (
        const Mat&                                       f_matrix,
        const std::pair<float, float>&                   f_betaRoPair
    );

    void computeThetaOneAndTwo
    (
        const std::vector<std::pair<float, float>>&      f_pointsVector,
        std::pair<float, float>&                         f_thetaPair
    );

    void computeBetaAndRo
    (
        const std::vector<std::pair<float, float>>&      f_pointsVector,
        std::pair<float, float>&                         f_betaRoPair
    );

    void readPointsFromFile
    (
        std::vector<std::pair<float, float>>&            f_pointsVector,
        const Mat&                                       f_matrix
    );

    void firstLearningModel
    (
        const Mat&                                       f_matrix,
        const std::vector<std::pair<float, float>>&      f_pointsVector,
        std::pair<float, float>&                         f_thetaPair
    );

    void matrixLearnigModel
    (
        const Mat&                                              f_matrix,
        const std::vector<std::pair<float, float>>&             f_pointsVector
    );

};

