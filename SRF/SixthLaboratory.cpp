#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <tuple>
#include "SixthLaboratory.h"

std::vector<Point2i> SixthLaboratory::getBlackPointsCoords(const Mat3b& f_bigMat)
{
    std::vector<Point2i> l_blackPoints;

    for (vfc::int32_t i = 0; i < f_bigMat.rows; ++i)
    {
        for (vfc::int32_t j = 0; j < f_bigMat.cols; ++j)
        {
            for (int k = 0; k < 3; ++k) 
            {
                if (f_bigMat(i, j)[k] == 0)
                {
                    l_blackPoints.push_back(Point2i(i, j));
                }
            }
        }
    }
    return l_blackPoints;
}

std::vector<Mat> SixthLaboratory::computeParameters(int d, const char* path)
{
    Mat img;
    if (d == 3)
    {
        img = imread(path, CV_LOAD_IMAGE_COLOR);
    }
    else
    {
        img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    }

    int counter = 0;

    switch (d)
    {
    case 1:
        counter = img.rows * img.cols;
        break;
    case 2:
        for (int i = 0; i < img.rows; ++i)
        {
            for (int j = 0; j < img.cols; ++j)
            {
                if (0 == img.at<uchar>(i, j))
                {
                    counter++;
                }
            }
        }
        break;
    case 3:
        counter = img.rows * img.cols;
        break;
    }

    Mat points(counter, d, CV_32FC1);
    Mat labels = Mat::zeros(counter, 1, CV_8UC1);

    counter = 0;
    switch (d)
    {
        case 1:
            for (int i = 0; i < img.rows; ++i)
            {
                for (int j = 0; j < img.cols; ++j)
                {
                    points.at<float>(counter, 0) = (float)img.at<uchar>(i, j);
                    counter++;
                }
            }
            break;

        case 2:
            for (int i = 0; i < img.rows; ++i)
            {
                for (int j = 0; j < img.cols; ++j)
                {
                    if (0 == img.at<uchar>(i, j))
                    {
                        points.at<float>(counter, 0) = (float)i;
                        points.at<float>(counter, 1) = (float)j;
                        counter++;
                    }
                }
            }
            break;

        case 3:
            for (int i = 0; i < img.rows; ++i)
            {
                for (int j = 0; j < img.cols; ++j)
                {
                    Vec3b color = img.at<Vec3b>(i, j);
                    points.at<float>(counter, 0) = color[0];
                    points.at<float>(counter, 1) = color[1];
                    points.at<float>(counter, 2) = color[2];
                    counter++;
                }
            }
            break;
    }
    
    std::vector<Mat> kmean;
    kmean.push_back(points);
    kmean.push_back(labels);
    return kmean;
}

Mat SixthLaboratory::computeKmean(const int nK, int d, const char* path, int noIterations)
{
    std::vector<Mat> parameters = computeParameters(d, path);
    std::default_random_engine gen(time(0));
    std::uniform_int_distribution<int> dist_img(0, (parameters.at(0).rows - 1) * 255);
    std::uniform_int_distribution<int> dist(0, 255);
    
    cv::Mat clusters = cv::Mat::zeros(nK, d, CV_32FC1);
    int step = 0;
    bool modified = true;
    Mat points = parameters.at(0);
    Mat labels = parameters.at(1);

    for (int i = 0; i < nK; ++i)
    {
        int rand = dist_img(gen) / 255;
        for (int q = 0; q < d; ++q)
        {
            clusters.at<float>(i, q) = points.at<float>(rand, q);
        }
    }

    while (step < noIterations && modified)
    {
        modified = false;
        computeDistances(points, nK, d, clusters, labels, modified);
        recomputeCenters(clusters, points, labels, d, nK);
        step++;
    }

    Mat result = colorPoints(nK, dist, gen, d, path, points, clusters, labels);

    FILE* f = fopen("labels.txt", "w");
    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            fprintf(f, "%u ", labels.at<uchar>(i, j));
        }
        fprintf(f, "\n");
    }


    imshow("Clusters", result);
    return clusters;
}

Mat SixthLaboratory::colorPoints(const int &nK, const std::uniform_int_distribution<int> &dist, 
    std::default_random_engine &gen, int d, const char * path, const cv::Mat &points, const cv::Mat &clusters, const cv::Mat &labels)
{
    Mat img;
    Mat result;
    uchar color;
    Vec3b colors[25];
    
    for (int i = 0; i < nK; ++i)
    {
        colors[i] = Vec3b(dist(gen), dist(gen), dist(gen));
    }

    if (d == 3)
    {
        img = imread(path, CV_LOAD_IMAGE_COLOR);
    }
    else
    {
        img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    }

    if (d == 2)
    {
        result = Mat(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));
    }
    else
    {
        result = img.clone();
    }

    for (int i = 0; i < points.rows; i++)
    {
        switch (d)
        {
            case 1:
            {
                result.at<uchar>(i / img.cols, i % img.cols) = (uchar)clusters.at<float>(labels.at<uchar>(i, 0) - 1, 0);
                break;
            }
            case 2:
            {
                int x = (int)points.at<float>(i, 0);
                int y = (int)points.at<float>(i, 1);
                result.at<Vec3b>(x, y) = colors[labels.at<uchar>(i, 0) - 1];
                break;
            }
            case 3:
            {
                uchar b = (uchar)clusters.at<float>(labels.at<uchar>(i, 0) - 1, 0);
                uchar g = (uchar)clusters.at<float>(labels.at<uchar>(i, 0) - 1, 1);
                uchar r = (uchar)clusters.at<float>(labels.at<uchar>(i, 0) - 1, 2);
                result.at<Vec3b>(i / img.cols, i % img.cols) = Vec3b(b, g, r);
                break;
            }
        }
    }

    return result;
}

void SixthLaboratory::computeDistances(const cv::Mat&         points, 
                                       const int&             k,
                                       int                    d,
                                       const cv::Mat&         clusters, 
                                       cv::Mat&               labels, 
                                       bool&                  modified)
{
    for (int i = 0; i < points.rows; ++i)
    {
        float min = FLT_MAX;

        int index = 0;
        for (int j = 0; j < k; ++j)
        {
            float distance = 0.0f;
            for (int q = 0; q < d; ++q)
            {
                distance += pow((points.at<float>(i, q) - clusters.at<float>(j, q)), 2);
            }

            if (sqrt(distance) < min)
            {
                min = sqrt(distance);
                index = j + 1;
            }
        }
        
        if (labels.at<uchar>(i, 0) != index)
        {
            labels.at<uchar>(i, 0) = index;
            modified = true;
        }
    }
}

void SixthLaboratory::recomputeCenters(cv::Mat &clusters, const cv::Mat &points, const cv::Mat &labels, int d, int nK)
{
    Mat center(nK, d, CV_32FC1);
    center.setTo(0);
    for (int i = 0; i < clusters.rows; ++i)
    {
        int counter = 0;
        for (int j = 0; j < points.rows; ++j)
        {
            if (labels.at<uchar>(j, 0) == i + 1)
            {
                counter++;
                for (int q = 0; q < d; ++q)
                {
                    center.at<float>(i, q) += points.at<float>(j, q);
                }
            }
        } 

        for (int q = 0; q < d; ++q)
        {
            clusters.at<float>(i, q) = center.at<float>(i, q) / counter;
        }
    }
}

void SixthLaboratory::Voronoi(const char* path, int clustersK, int noIterations)
{
    std::default_random_engine gen(time(0));
    std::uniform_int_distribution<int> dist(0, 255);

    Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    Mat result = Mat::ones(img.rows, img.cols, CV_8UC3);


    Mat labels = Mat(img.rows * img.cols, 1, CV_8UC1, Scalar(0));
    Mat points = Mat(img.rows * img.cols, 2, CV_32FC1);
    Mat c = computeKmean(clustersK, 2, path, noIterations);

    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            points.at<float>(i * img.cols + j, 0) = (float)i;
            points.at<float>(i * img.cols + j, 1) = (float)j;
        }
    }

    // clusters = Mat::zeros(k, d, CV_32FC1);

    for (int i = 0; i < points.rows; ++i)
    {
        float min = FLT_MAX;
        //int index = 0;
        for (int j = 0; j < clustersK; ++j)
        {
            float sum = 0.0f;
            for (int q = 0; q < 2; ++q)
            {
                sum += (points.at<float>(i, q) - c.at<float>(j, q)) *
                    (points.at<float>(i, q) - c.at<float>(j, q));
            }

            if (sqrt(sum) < min)
            {
                min = sqrt(sum);
                //index = j + 1;
                labels.at<uchar>(i, 0) = j + 1;
            }

        }

    }

    Vec3b colors[20];
    for (int i = 0; i < clustersK; ++i)
    {
        colors[i] = Vec3b(dist(gen), dist(gen), dist(gen));
    }

    for (int i = 0; i < points.rows; ++i)
    {
        int x = (int)points.at<float>(i, 0);
        int y = (int)points.at<float>(i, 1);

        result.at<Vec3b>(x, y) = colors[labels.at<uchar>(i, 0) - 1];
    }

    imshow("Voronoi", result);
}