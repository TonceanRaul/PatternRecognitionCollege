#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "FirstLaboratory.h"

FirstLaboratory::FirstLaboratory()
{
}


FirstLaboratory::~FirstLaboratory()
{
}

void FirstLaboratory::drawLineFirstApproach(const Mat&                                       f_matrix,
                                            const std::pair<float, float>&                   f_thetaPair)
{
    int32_t l_x1_i32 = 0;
    int32_t l_x2_i32 = 0;
    int32_t l_y1_i32 = 0;
    int32_t l_y2_i32 = 0;

    if (abs(f_thetaPair.second) > 1)
    {
        l_x1_i32 = -f_thetaPair.first / f_thetaPair.second;
        l_x2_i32 = (500 - f_thetaPair.first) / f_thetaPair.second;
        l_y1_i32 = 0;
        l_y2_i32 = 500;
    }
    else
    {
        l_y1_i32 = f_thetaPair.first;
        l_y2_i32 = f_thetaPair.first + f_thetaPair.second * 500;
        l_x1_i32 = 0;
        l_x2_i32 = 500;
    }

    line(f_matrix, Point(l_x1_i32, l_y1_i32), Point(l_x2_i32, l_y2_i32), Scalar(0, 0, 255));
    imshow("2D Points", f_matrix);

    waitKey();
}

void FirstLaboratory::drawLineSecondApproach(const Mat&                                      f_matrix,
                                             const std::pair<float, float>&                  f_betaRoPair)
{
    vfc::int32_t l_x1_i32 = 0;
    vfc::int32_t l_x2_i32 = 0;
    vfc::int32_t l_y1_i32 = 0;
    vfc::int32_t l_y2_i32 = 0;

    if (abs(f_betaRoPair.first) < 1)
    {
        l_x1_i32 = f_betaRoPair.second / cos(f_betaRoPair.first);
        l_x2_i32 = (f_betaRoPair.second - 500 * sin(f_betaRoPair.first)) / cos(f_betaRoPair.first);
        l_y1_i32 = 0;
        l_y2_i32 = 500;
    }
    else
    {
        l_y1_i32 = f_betaRoPair.second / sin(f_betaRoPair.first);
        l_y2_i32 = (f_betaRoPair.second - 500 * cos(f_betaRoPair.first)) / sin(f_betaRoPair.first);
        l_x1_i32 = 0;
        l_x2_i32 = 500;
    }

    line(f_matrix, Point(l_x1_i32, l_y1_i32), Point(l_x2_i32, l_y2_i32), Scalar(0, 0, 255));

    imshow("2D cos and sin", f_matrix);
    waitKey(0);
}

void FirstLaboratory::computeThetaOneAndTwo(const std::vector<std::pair<float, float>>&      f_pointsVector,
                                            std::pair<float, float>&                         f_thetaPair)
{
    int32_t l_vectorSize_i32 = f_pointsVector.size();
    float_t l_theta0_f32 = 0;
    float_t l_theta1_f32 = 0;
    float_t l_xi_f32 = 0;
    float_t l_yi_f32 = 0;
    float_t l_xiMyi_f32 = 0;
    float_t l_xiSquare_f32 = 0;

    for (int i = 0; i < l_vectorSize_i32; ++i)
    {
        l_xiMyi_f32 += f_pointsVector.at(i).first * f_pointsVector.at(i).second;
        l_xi_f32 += f_pointsVector.at(i).first;
        l_yi_f32 += f_pointsVector.at(i).second;
        l_xiSquare_f32 += f_pointsVector.at(i).first * f_pointsVector.at(i).first;
    }

    l_theta1_f32 = (l_vectorSize_i32 * l_xiMyi_f32) - (l_xi_f32 * l_yi_f32);
    l_theta1_f32 /= ((l_vectorSize_i32 * l_xiSquare_f32) - (l_xi_f32 * l_xi_f32));

    l_theta0_f32 = (1.0f / l_vectorSize_i32) * (l_yi_f32 - l_theta1_f32 * l_xi_f32);

    f_thetaPair = std::make_pair(l_theta0_f32, l_theta1_f32);
}

void FirstLaboratory::computeBetaAndRo(const std::vector<std::pair<float, float>>&           f_pointsVector,
                                       std::pair<float, float>&                              f_betaRoPair)
{
    vfc::float32_t l_beta_f32 = 0;
    vfc::float32_t l_ro_f32 = 0;
    vfc::float32_t l_xi_f32 = 0;
    vfc::float32_t l_yi_f32 = 0;
    vfc::float32_t l_xiMyi_f32 = 0;

    vfc::float32_t l_yiSquareMinusxiSquare_f32 = 0;
    vfc::int32_t l_vectorSize_i32 = f_pointsVector.size();

    for (vfc::int32_t i = 0; i < l_vectorSize_i32; ++i)
    {
        l_yiSquareMinusxiSquare_f32 += f_pointsVector.at(i).second * f_pointsVector.at(i).second -
            f_pointsVector.at(i).first * f_pointsVector.at(i).first;
        l_xi_f32 += f_pointsVector.at(i).first;
        l_yi_f32 += f_pointsVector.at(i).second;
        l_xiMyi_f32 += f_pointsVector.at(i).first * f_pointsVector.at(i).second;
    }

    l_beta_f32 = (-1 / 2.0) * atan2(2 * l_xiMyi_f32 - ((2.0 / l_vectorSize_i32) * l_xi_f32 * l_yi_f32),
        l_yiSquareMinusxiSquare_f32 + (((1.0 / l_vectorSize_i32) * (l_xi_f32 * l_xi_f32)) -
        (1.0 / l_vectorSize_i32) * (l_yi_f32 * l_yi_f32)));

    l_ro_f32 = (1.0 / l_vectorSize_i32) * (cos(l_beta_f32) * l_xi_f32 + sin(l_beta_f32) * l_yi_f32);

    f_betaRoPair = std::make_pair(l_beta_f32, l_ro_f32);
}

void FirstLaboratory::readPointsFromFile(std::vector<std::pair<float, float>>&               f_pointsVector,
                                         const Mat&                                          f_matrix)
{
    FILE* f = fopen("points0.txt", "r");
    assert(f != 0);

    vfc::int32_t   l_noPoints_i32 = 0;
    vfc::float32_t l_firstPoint_f32 = 0;
    vfc::float32_t l_secondPoint_f32 = 0;

    fscanf(f, "%d", &l_noPoints_i32);

    for (vfc::int32_t i = 0; i < l_noPoints_i32; ++i)
    {
        fscanf(f, "%f", &l_firstPoint_f32);
        fscanf(f, "%f", &l_secondPoint_f32);

        f_pointsVector.push_back(std::make_pair(l_firstPoint_f32, l_secondPoint_f32));

        Point2f l_2Dpoint = Point2f(l_firstPoint_f32, l_secondPoint_f32);

        if (l_2Dpoint.x >= 0 &&
            l_2Dpoint.y >= 0 &&
            l_2Dpoint.y <= f_matrix.rows &&
            l_2Dpoint.x <= f_matrix.cols)
        {
            cv::circle(f_matrix, l_2Dpoint, 1, Scalar(255, 0, 0));
        }
    }
    fclose(f);
}

void FirstLaboratory::firstLearningModel(const Mat&                                              f_matrix,
                                         const std::vector<std::pair<float, float>>&             f_pointsVector,
                                         std::pair<float, float>&                                f_thetaPair)
{
    vfc::float32_t l_teta0Deriv_f32 = 0.0f;
    vfc::float32_t l_teta1Deriv_f32 = 0.0f; 
    vfc::float32_t learning_rate = 0.0001f / f_pointsVector.size();

    for (int i = 0; i < f_pointsVector.size(); ++i)
    {
        vfc::float32_t prediction = f_thetaPair.first + f_thetaPair.second * f_pointsVector.at(i).first;
        l_teta0Deriv_f32 += prediction - f_pointsVector.at(i).second;
        l_teta1Deriv_f32 += (prediction - f_pointsVector.at(i).second) * f_pointsVector.at(i).first;
    }

    f_thetaPair.first  = f_thetaPair.first - learning_rate * l_teta0Deriv_f32;
    f_thetaPair.second = f_thetaPair.second - learning_rate * l_teta1Deriv_f32;

    printf("Teta0: %f\n", f_thetaPair.first);
    printf("Teta1: %f\n", f_thetaPair.second);

    drawLineFirstApproach(f_matrix, f_thetaPair); 
    waitKey(27);
}

void FirstLaboratory::matrixLearnigModel(const Mat&                                              f_matrix,
                                         const std::vector<std::pair<float, float>>&             f_pointsVector)
{
    Mat A(f_pointsVector.size(), 2, CV_32F);
    Mat B(f_pointsVector.size(), 1, CV_32F);
    Mat theta(2, 1, CV_32F);
    A.setTo(1);
    for (int i = 0; i < f_pointsVector.size(); ++i)
    {
        A.at<float>(i, 0) = f_pointsVector.at(i).first;
        B.at<float>(i, 0) = f_pointsVector.at(i).second;
    }

    theta = (A.t() * A).inv() * A.t() * B;

    Point l_point1;
    Point l_point2;

    l_point1.x = 0;
    l_point2.x = 500;
    l_point1.y = theta.at<float>(1, 0);
    l_point2.y = theta.at<float>(1, 0) + theta.at<float>(0, 0) * 500;

    line(f_matrix, l_point1, l_point2, Scalar(0, 0, 255));
    
    imshow("Matrix form", f_matrix);
    waitKey();
}
