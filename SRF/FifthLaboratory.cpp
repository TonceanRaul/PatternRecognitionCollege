#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "FifthLaboratory.h"

FifthLaboratory::FifthLaboratory()
{
}

FifthLaboratory::~FifthLaboratory()
{
}

Mat1b FifthLaboratory::constructImage()
{
    char folder[256] = "E:\\Repos\\Rainbows\\OpenCVApplication-VS2017_OCV340_basic\\prs_res_Statistics";
    char fname[256];
    Mat1b bigMat = Mat1b(400, 361);

    for (int it = 1; it <= 400; it++)
    {
        sprintf(fname, "%s/face%05d.bmp", folder, it);
        Mat1b img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

        for (int i = 0; i < img.rows; ++i)
        {
            for (int j = 0; j < img.cols; ++j)
            {
                bigMat(it - 1, i * img.cols + j) = img(i, j);
            }
        }
    }

    return bigMat;
}

std::vector<vfc::float32_t> FifthLaboratory::computeMeanScanningRows(Mat1b f_matrix)
{
    FILE *f = fopen("media.csv", "w");

    std::vector<vfc::float32_t> l_meanVector;
    vfc::float32_t l_meanValue_f32 = 0;

    for (int i = 0; i < f_matrix.cols; ++i)
    {
        vfc::float32_t l_meanValue_f32 = 0.0f;
        for (int j = 0; j < f_matrix.rows; ++j)
        {
            l_meanValue_f32 += f_matrix(j, i);
        }
        l_meanValue_f32 /= f_matrix.rows;
        fprintf(f, "%d,%f\n", i, l_meanValue_f32);

        l_meanVector.push_back(l_meanValue_f32);
    }
    fclose(f);

    return l_meanVector;
}

void FifthLaboratory::computeCovarianceMatrix(Mat1b f_intensity)
{
    FILE* f = fopen("media.csv", "r");
    FILE* g = fopen("covariance.csv", "w");
    std::vector<float> l_meanVector;

    for (int i = 0; i < f_intensity.cols; ++i)
    {
        vfc::float32_t value = 0;
        int dummy = 0;
        fscanf(f, "%d,%f\n", &dummy, &value);
        l_meanVector.push_back(value);
    }

    for (int i = 0; i < f_intensity.cols; ++i)
    {
        for (int j = 0; j < f_intensity.cols; ++j)
        {
            vfc::float32_t value = 0.0f;
            for (int k = 0; k < f_intensity.rows; k++)
            {
                value += (f_intensity(k, i) - l_meanVector.at(i)) * (f_intensity(k, j) - l_meanVector.at(j));
            }
            value = value / f_intensity.rows;
            fprintf(g, "%d,%d,%f\n", i, j, value);
        }
    }
    fclose(f);
    fclose(g);
}

void FifthLaboratory::computeCorelationCoeff(Mat1b f_covariance)
{
    FILE* g = fopen("covariance.csv", "r");
    FILE* c = fopen("corelationCoeff.csv", "w");

    std::vector<vfc::float32_t> covarianceVector;
    
    for (int index = 0; index < f_covariance.cols * f_covariance.cols; ++index)
    {
        vfc::float32_t vectorValue= 0;
        vfc::int32_t i = 0;
        vfc::int32_t j = 0;
        fscanf(g, "%d,%d,%f", &i, &j, &vectorValue);

        if (i == j)
        {
            covarianceVector.push_back(sqrt(vectorValue));
        }
    }

    fclose(g);
    g = fopen("covariance.csv", "r");

    for (int index = 0; index < f_covariance.cols * f_covariance.cols; ++index)
    {
        vfc::float32_t vectorValue = 0;
        vfc::int32_t i = 0;
        vfc::int32_t j = 0;
        fscanf(g, "%d,%d,%f", &i, &j, &vectorValue);

        float coeff = vectorValue / (covarianceVector.at(i) * covarianceVector.at(j));
        fprintf(c, "%d,%d,%f\n", i, j, coeff);
    }
    fclose(c);
}

std::vector<Mat1b> FifthLaboratory::computeMatForGraphics(Mat1b f_matrix)
{
    std::vector<Mat1b> matVector;
    Mat image = Mat(256, 256, CV_LOAD_IMAGE_GRAYSCALE);
    image.setTo(255);

    int firstCoordX  = 5;
    int firstCoordY  = 4;
    int secondCoordX = 5;
    int secondCoordY = 14;
    for (int i = 0; i < f_matrix.rows; i++)
    {
        circle(image, Point(f_matrix(i, firstCoordX * 19 + firstCoordY),
            f_matrix(i, secondCoordX * 19 + secondCoordY)), 1, Scalar(0, 0, 0), 1);
    }

    matVector.push_back(image.clone());

    image.setTo(255);
    firstCoordX = 10;
    firstCoordY = 3;
    secondCoordX = 9;
    secondCoordY = 15;

    for (int i = 0; i < f_matrix.rows; ++i)
    {
        circle(image, Point(f_matrix(i, firstCoordX * 19 + firstCoordY),
            f_matrix(i, secondCoordX * 19 + secondCoordY)), 1, Scalar(0, 0, 0), 1);
    }

    matVector.push_back(image.clone());
    
    image.setTo(255);
    firstCoordX = 5;
    firstCoordY = 4;
    secondCoordX = 18;
    secondCoordY = 0;
    
    for (int i = 0; i < f_matrix.rows; ++i)
    {
        circle(image, Point(f_matrix(i, firstCoordX * 19 + firstCoordY),
            f_matrix(i, secondCoordX * 19 + secondCoordY)), 1, Scalar(0, 0, 0), 1);
    }

    matVector.push_back(image.clone());

    return matVector;
}

void FifthLaboratory::displayGraphics(std::vector<Mat1b> f_matrixVector)
{
    int counter = 0;
    for (int i = 0; i < f_matrixVector.size(); ++i)
    {
        imshow("Mat" + std::to_string(counter), f_matrixVector.at(i));
        counter++;
    }
    waitKey(0);
}

void FifthLaboratory::gaussianDensity(Mat1b intensity)
{
    Mat distribution = Mat(400, 400, CV_LOAD_IMAGE_GRAYSCALE);
    distribution.setTo(255);

    FILE* f = fopen("media.csv", "r");
    FILE* g = fopen("covariance.csv", "r");

    std::vector<float> l_meanVector;
    std::vector<float> l_covarianceVector;

    for (int i = 0; i < intensity.cols; ++i) 
    {
        float value = 0;
        fscanf(f, "%d,%f\n", &i, &value);
        l_meanVector.push_back(value);
    }

    for (int k = 0; k < intensity.cols * intensity.cols; ++k)
    {
        float value;
        int i, j;
        fscanf(g, "%d,%d,%f\n", &i, &j, &value);
        if (i == j) 
        {
            l_covarianceVector.push_back(sqrt(value));
        }
    }

    fclose(f);
    fclose(g);

    float maxValue = 0.0f;
    std::vector<float> result;
    for (int k = 0; k < intensity.rows; ++k) 
    {
        float x = (1.0f / (sqrt(2 * CV_PI) * l_covarianceVector.at(5 * 19 + 4))) * 
            cv::exp(-pow(intensity(k, 5 * 19 + 4) - l_meanVector.at(5 * 19 + 4), 2) /
            (2 * pow(l_covarianceVector.at(5 * 19 + 4), 2)));

        if (x > maxValue)
        {
            maxValue = x;
        }
        result.push_back(x);
    }

    for (int k = 0; k < distribution.rows; k++)
    {
        //printf("%f\n", (result.at(k) / maxValue) * distribution.cols);
        circle(distribution, Point(k, (distribution.cols * result.at(k) / maxValue)),
           1, Scalar(0, 0, 0), 1);

       
        //distribution.at<uchar>(distribution.cols * result.at(k) / maxValue, k) = 0;
    }

    imshow("Probability", distribution);
    waitKey(0);
}
