#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "SeventhLaboratory.h"


SeventhLaboratory::SeventhLaboratory()
{
}


SeventhLaboratory::~SeventhLaboratory()
{
}

Mat SeventhLaboratory::computePCA(const char* path, int red_dim)
{
    FILE* f = fopen(path, "r");
    vfc::int32_t n = 0;
    vfc::int32_t d = 0;
    vfc::float32_t x = 0;
    
    std::vector<float> meanVector;

    if (NULL == f)
    {
        printf("The file doesn't exit.\n");
        return Mat::zeros(1, 1, CV_32FC1);
    }

    fscanf(f, "%d %d", &n, &d);
    Mat result(n, d, CV_32FC1);
    Mat original;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            fscanf(f, "%f", &x);
            result.at<float>(i, j) = x;
        }
    }

    for (int i = 0; i < result.cols; ++i)
    {
        float media = 0.0f;
        for (int k = 0; k < result.rows; ++k)
        {
            media += result.at<float>(k, i);
        }
        media = media / result.rows;
        meanVector.push_back(media);
    }

    // scaderea mediei din vectorul de puncte
    for (int i = 0; i < result.cols; ++i)
    {
        for (int j = 0; j < result.rows; ++j)
        {
            result.at<float>(j, i) -= meanVector.at(i);
        }
    }

    original = result.clone();
    //calcularea matricei de covarianta
    Mat C = result.t() * result / (n - 1);

    //descompunerea in valori proprii si vectori proprii
    Mat Lambda, Q; // valorile proprii imprastierea pe axe // vectorii proprii dau directia axelor
    eigen(C, Lambda, Q);
    Q = Q.t();

    std::cout << "Valori proprii: \n" << std::endl;
    for (int i = 0; i < d; i++)
    {
        printf("%f\n", Lambda.at<float>(i, 0));
    }

    // sortare elemente si aflarea punctelor in dimensiunea redusa
    Mat lambda_sort;
    Mat q_sort(Q.rows, red_dim, CV_32FC1);
    sort(Lambda, lambda_sort, CV_SORT_EVERY_COLUMN);
    
    for (int i = 0; i < d; i++)
    {
        for (int q = 0; q < red_dim; ++q)
        {
            q_sort.at<float>(i, q) = Q.at<float>(i, q);
        }
    }

    printf("\n");
    printf("\n");
    printf("\n");

    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < q_sort.cols; ++j)
        {
            printf("%f ", q_sort.at<float>(i, j));
        }
        printf("\n");
    }

    Mat new_X(n, red_dim, CV_32FC1);
    new_X = result * q_sort;

    printf("\n");
    printf("\n");
    printf("\n");

    printf("Coef\n");
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < q_sort.cols; ++j)
        {
            printf("%f ", new_X.at<float>(i, j));
        }
        printf("\n");
    }
    
    // calcularea diferentei dintre punctele originale si cele aproximate
    Mat old_X = new_X * q_sort.t();
    float media = 0.0;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            media += abs(old_X.at<float>(i, j) - original.at<float>(i, j));
        }
    }

    media = media / (n * d);
    printf("Media diferentei dintre punctele originale si cele aproximate: %f\n", media);

    // minimele si maximele pe coloanele de coeficienti
    printf("Minimele si maximele pe coloanele de coeficienti: \n");
    double minValue, maxValue;
    Point minLoc, maxLoc;

    for (int i = 0; i < red_dim; ++i)
    {
        minMaxLoc(new_X.col(i), &minValue, &maxValue, &minLoc, &maxLoc);
        printf("Minim: %lf, Maxim %lf\n", minValue, maxValue);
    }

    waitKey();
    return new_X;
}

void SeventhLaboratory::imageUsingPCA2(char* path, int k)
{
    Mat coef_mat = computePCA(path, k);
    double minValue, maxValue;
    Point minLoc, maxLoc;
    double maxColor, minColor;
    float maxx = 0.0f;
    float maxy = 0.0f;

    for (int i = 0; i < coef_mat.cols; ++i)
    {
        minMaxLoc(coef_mat.col(i), &minValue, &maxValue, &minLoc, &maxLoc);
        for (int j = 0; j < coef_mat.rows; ++j)
        {
            coef_mat.at<float>(j, i) -= (float)minValue;
        }
    }

    Mat img(500, 500, CV_8UC1, Scalar(255));

    for (int j = 0; j < coef_mat.rows; ++j)
    {
        int x = (int)coef_mat.at<float>(j, 0);
        int y = (int)coef_mat.at<float>(j, 1);
        img.at<uchar>(x, y) = 0;
    }

    imshow("Cerc", img);
    waitKey();
}

void SeventhLaboratory::imageUsingPCA3(char* path, int k)
{
    Mat coef_mat = computePCA(path, k);
    double minValue, maxValue;
    Point minLoc, maxLoc;
    double maxColor, minColor;
    float maxx = 0.0f;
    float maxy = 0.0f;

    for (int i = 0; i < coef_mat.cols - 1; ++i)
    {
        minMaxLoc(coef_mat.col(i), &minValue, &maxValue, &minLoc, &maxLoc);
        for (int j = 0; j < coef_mat.rows; ++j)
        {
            coef_mat.at<float>(j, i) -= (float)minValue;
        }
    }

    Mat img(500, 500, CV_8UC1, Scalar(255));

    for (int j = 0; j < coef_mat.rows; ++j)
    {
        int x = (int)coef_mat.at<float>(j, 1);
        int y = (int)coef_mat.at<float>(j, 0);


        float aux = 0;
        aux = coef_mat.at<float>(j, 2);
        //if (aux < 0)
        //{
        //    aux = 0;
        //}
        //if (coef_mat.at<float>(j, 2) > 255)
        //{
        //    aux = 255;
        //}
        
        img.at<uchar>(x, y) = aux;
    }

    imshow("Expected Lena <3", img);
    waitKey(0);
}
