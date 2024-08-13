// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <vector>
#include <unordered_set>
#include <queue>
#include <random>
#include <stack>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "FirstLaboratory.h"
#include "SecondLaboratory.h"
#include "ThirdLaboratory.h"
#include "FourthLaboratory.h"
#include "FifthLaboratory.h"
#include "SixthLaboratory.h"
#include "SeventhLaboratory.h"
#include "EightLaboratory.h"
#include "NinthLaboratory.h"
#include "TenthLaboratory.h"
#include "EleventhLaboratory.h"

using namespace std;

template<typename T>
void computeOverflows(T min, T max, T* value)
{
    if (*value > max)
    {
        *value = max;
    }
    else if (*value < min)
    {
        *value = min;
    }
}

void minFilter(const Mat1b& src, Mat1b& dst, int radius)
{

    Mat1b padded;
    copyMakeBorder(src, padded, radius, radius, radius, radius, BORDER_CONSTANT, Scalar(255));

    int rr = src.rows;
    int cc = src.cols;

    dst = Mat1b(rr, cc, uchar(0));

    for (int r = 0; r < rr; ++r)
    {
        for (int c = 0; c < cc; ++c)
        {
            uchar lowest = 255;
            for (int i = -radius; i <= radius; ++i)
            {
                for (int j = -radius; j <= radius; ++j)
                {
                    uchar val = padded(radius + r + i, radius + c + j);
                    if (val < lowest)
                    {
                        lowest = val;
                    }
                }
                dst(r, c) = lowest;
            }
        }
    }
}

void minValue3b(const Mat3b& src, Mat1b& dst)
{
    int rr = src.rows;
    int cc = src.cols;

    dst = Mat1b(rr, cc, uchar(0));

    for (int r = 0; r < rr; ++r)
    {
        for (int c = 0; c < cc; ++c)
        {
            const Vec3b& v = src(r, c);

            uchar lowest = v[0];

            if (v[1] < lowest)
            {
                lowest = v[1];
            }

            if (v[2] < lowest)
            {
                lowest = v[2];
            }

            dst(r, c) = lowest;
        }
    }
}

void imageDehazer(const Mat3b& img, Mat1b& dark, Mat3b& dehazedImage)
{
    uchar highest = 0;
    for (int i = 0; i < dark.rows; ++i)
    {
        for (int j = 0; j < dark.cols / 2; ++j)
        {
            if (dark(i, j) > highest)
            {
                highest = dark(i, j);
            }
        }
    }
    printf("highest %d", highest);
    for (int i = 0; i < dark.rows; ++i)
    {
        for (int j = 0; j < dark.cols; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                int aux = (img(i, j)[k] * 1.10 - (0.85) * dark(i, j)) / (1.0 - (0.85) * (dark(i, j) / static_cast<float>(highest)));
                
                computeOverflows<int>(0, 255, &aux);

                dehazedImage(i, j)[k] = aux;
            }
        }
    }
}

float computeStandardDeviationAndMean(const Mat1b& src, float* meanValue, float* stDeviation)
{
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            *meanValue += src(i, j);
        }
    }

    *meanValue /= (src.rows * src.cols);

    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            *stDeviation += pow((src(i, j) - (*meanValue)), 2);
        }
    }
    
    return sqrt(*stDeviation  / (src.rows * src.cols));
}

Mat1b medianFilter(const Mat1b& src, int kernelSize)
{
    Mat1b dst = src.clone();
    int k = (kernelSize - 1) / 2;
    int di[8] = { -1, 0, 1, 0, 1, -1, 1, -1 };
    int dj[8] = { 0, -1, 0, 1, 1, 1, -1, -1 };

    for (int i = k; i < src.rows - k; ++i)
    {
        for (int j = k; j < src.cols - k; ++j)
        {
            std::vector<uchar> neighbors;
            neighbors.push_back(src(i, j));
        
            for (int neighbor = 0; neighbor < 8; ++neighbor)
            {
                neighbors.push_back(src(i + di[neighbor], j + dj[neighbor]));
            }
            // qsort in C uses QuickSort vs. sort in C++ uses introsort
            // 3-part hybrid sorting algorithm: 
            // introsort is performed first (introsort itself being a hybrid of quicksort and heap sort) 
            // followed by an insertion sort on the result.
            sort(neighbors.begin(), neighbors.end());
            dst(i, j) = neighbors[4];
        }
    }

    return dst;
}


void DarkChannel(const Mat3b& img, Mat1b& dark, int patchSize)
{
    int radius = patchSize / 2;

    Mat1b W;
    minValue3b(img, W);

    imshow("MinValue", W);

    Mat1b median = medianFilter(W, 3);
    imshow("Median", median);
    float meanValueW = 0.0f;
    float stDeviation = 0.0f;
    computeStandardDeviationAndMean(W, &meanValueW, &stDeviation);

    minFilter(W, dark, radius);

    imshow("MinFilter", dark);

    Mat3b dehazedImage(dark.rows, dark.cols, CV_8UC3);
    imageDehazer(img, dark, dehazedImage); // W instead of dark

    imshow("Destination", dehazedImage);
}

int main()
{
    std::vector<std::pair<float, float>> l_pointsVector;
    std::pair<float, float> l_thetaPair;
    std::pair<float, float> l_betaRoPair;
    Mat l_matrix = Mat(500, 500, CV_8UC3);
    l_matrix.setTo(255);

    FirstLaboratory   first;
    SecondLaboratory  second;
    ThirdLaboratory   third;
    FourthLaboratory  fourth;
    FifthLaboratory   fifth;
    SixthLaboratory   sixth;
    SeventhLaboratory seventh;
    EightLaboratory   eigth;
    NinthLaboratory   ninth;
    TenthLaboratory   tenth;
    EleventhLaboratory eleventh;

    vector<Mat> trainSet;
    int op;
    int clusters;
    int noIterations;
    // prs_res_Bayes\test
    do
    {
        system("cls");
        destroyAllWindows();
        ///////////////////////////////////////////////////// LABORATORUL 1

        printf("\t\t\t\t\t\tLaboratorul I \n");
        printf(" 1 - Open image\n");
        printf(" 2 - Open BMP images from folder\n");
        printf(" 3 - Image negative - diblook style\n");
        printf(" 4 - BGR->HSV\n");
        printf(" 5 - Resize image\n");
        printf(" 6 - Canny edge detection\n");
        printf(" 7 - Edges in a video sequence\n");
        printf(" 8 - Snap frame from live video\n");
        printf(" 9 - Mouse callback demo\n");
        printf(" 10 - Aditive image\n");
        printf(" 11 - Multiplicative image\n");
        printf(" 12 - 4 color image\n");

        scanf("%d", &op);
        switch (op)
        {
            case 10:
            {
                first.readPointsFromFile(l_pointsVector, l_matrix);
                first.computeThetaOneAndTwo(l_pointsVector, l_thetaPair);
                first.computeBetaAndRo(l_pointsVector, l_betaRoPair);
                first.drawLineSecondApproach(l_matrix, l_betaRoPair);
                break;
            }
            case 11:
            {
                first.readPointsFromFile(l_pointsVector, l_matrix);

                std::default_random_engine l_gen(time(0));
                std::uniform_real_distribution<vfc::float32_t> l_d(0.0, 1.0);

                srand(NULL);
                l_thetaPair = std::make_pair(-1.0f + 2 * ((float)rand() / (float)RAND_MAX),
                    -1.0f + 2 * ((float)rand() / (float)RAND_MAX));
                std::cout << "Theta 0 = " << l_thetaPair.first << "\n";
                std::cout << "Theta 1 = " << l_thetaPair.second << "\n";

                vfc::int32_t l_noIterations = 0;
                std::cout << "Enter number of iterations: ";
                std::cin >> l_noIterations;

                for (vfc::int32_t i = 0; i < l_noIterations; ++i)
                {
                    std::cout << "At iteration " << i + 1 << "\n";
                    first.firstLearningModel(l_matrix, l_pointsVector, l_thetaPair);
                }
                break;
            }
            case 12:
            {
                first.readPointsFromFile(l_pointsVector, l_matrix);
                first.matrixLearnigModel(l_matrix, l_pointsVector);
                break;
            }
            case 21:
            {
                char fname[MAX_PATH];
                //openFileDlg(fname);
                Mat1b l_grayScaleMatrix = imread("points1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
                imshow("RANSAC points", l_grayScaleMatrix);

                std::vector<std::pair<int, int>> l_blackPoints;
                l_blackPoints = second.findPositionOfBlackPoints(l_grayScaleMatrix);

    #ifdef DEBUG_PRINT
                for (int i = 0; i < l_blackPoints.size(); ++i)
                {
                    std::cout << l_blackPoints[i].first << "  " << l_blackPoints[i].second << std::endl;
                }
    #endif
                std::tuple<float, float, float> l_parameters = second.ransacAlgorithm(l_blackPoints);
                vfc::float32_t a = std::get<0>(l_parameters);
                vfc::float32_t b = std::get<1>(l_parameters);
                vfc::float32_t c = std::get<2>(l_parameters);

                int l_x1_i32 = 0;
                int l_x2_i32 = 0;
                int l_y1_i32 = 0;
                int l_y2_i32 = 0;

                if (abs(-a / b) > 1)
                {
                    l_y1_i32 = 0;
                    l_y2_i32 = l_grayScaleMatrix.rows;
                    l_x1_i32 = -c / a;
                    l_x2_i32 = (-c - b * l_y2_i32) / a;
                }
                else
                {
                    l_x1_i32 = 0;
                    l_x2_i32 = l_grayScaleMatrix.cols;
                    l_y1_i32 = -c / b;
                    l_y2_i32 = (-c - a * l_x2_i32) / b;
                }

                line(l_grayScaleMatrix, Point(l_y1_i32, l_x1_i32), Point(l_y2_i32, l_x2_i32), Scalar(0, 0, 255));
                imshow("RANSAC line", l_grayScaleMatrix);
                waitKey();

                break;
            }
            case 31:
            {
                Mat1b l_grayScaleMatrix = imread("edge_simple.bmp", CV_LOAD_IMAGE_GRAYSCALE);
                Mat Hough = third.computeHoughMatrix(l_grayScaleMatrix);
                third.houghSinus(Hough);
                third.computePeakVector(Hough, l_grayScaleMatrix);

                waitKey(0);

                break;
            }
            case 41:
            {
                char fname[MAX_PATH];
                openFileDlg(fname);
                Mat1b l_countourMatrix = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
                Mat1b l_templateObject = imread("E:\\Repos\\Rainbows\\OpenCVApplication-VS2017_OCV340_basic\\images_DT_PM\\images_DT_PM\\PatternMatching\\template.bmp", CV_LOAD_IMAGE_GRAYSCALE);

                Mat1b dst = fourth.compute2Scans(l_templateObject);

                float rez = fourth.patternMatching(l_countourMatrix, dst);

                std::pair<int, int> centerOfMassCountourImage = fourth.computeCenterOfMass(l_countourMatrix);
                std::pair<int, int> centerOfMassTemplateImage = fourth.computeCenterOfMass(l_templateObject);

                std::pair<int, int> translation = std::make_pair(centerOfMassCountourImage.first - centerOfMassTemplateImage.first,
                    centerOfMassCountourImage.second - centerOfMassTemplateImage.second);
                //l_countourMatrix.at<uchar>(centerOfMassCountourImage.second, centerOfMassCountourImage.first) = 128;
                //l_templateObject.at<uchar>(centerOfMassTemplateImage.second, centerOfMassTemplateImage.first) = 128;

                Mat countourClone = l_countourMatrix.clone();
                //countourClone.setTo(255);

                cout << "The first match " << rez << endl;
                cout << centerOfMassCountourImage.second << "   " << centerOfMassCountourImage.first << endl;
                cout << centerOfMassTemplateImage.second << "   " << centerOfMassTemplateImage.first << endl;
                cout << endl << endl;
                cout << "TRANSLATE " << translation.first << "    " << translation.second << endl;
                imshow("Before", countourClone);
                for (int i = 0; i < countourClone.rows; ++i)
                {
                    for (int j = 0; j < countourClone.cols; ++j)
                    {
                        if (l_countourMatrix.at<uchar>(i, j) == 0)
                        {
                            if (i + translation.second > 0 &&
                                i + translation.second < countourClone.rows &&
                                j + translation.first > 0 &&
                                j + translation.first < countourClone.cols)
                            {
                                countourClone.at<uchar>(i, j) = 255;
                                countourClone.at<uchar>(i + translation.second, j + translation.first) = 0;
                            }
                        }
                    }
                }
                imshow("After", countourClone);

                std::pair<int, int> centerOfMassCountourImageClone = fourth.computeCenterOfMass(countourClone);

                rez = fourth.patternMatching(countourClone, dst);

                cout << "The second match " << rez << endl;
                cout << centerOfMassCountourImage.second << "   " << centerOfMassCountourImage.first << endl;
                cout << centerOfMassCountourImageClone.second << "   " << centerOfMassCountourImageClone.first << endl;

                imshow("Clone", countourClone);
                imshow("Source", l_countourMatrix);
                imshow("Dest", dst);
                imshow("Comb", l_templateObject & dst);
                imshow("Template object", l_templateObject);

                waitKey(0);
                break;
            }
            case 51:
            {
                Mat1b bigMat = fifth.constructImage();
                fifth.computeCovarianceMatrix(bigMat);
                imshow("Big", bigMat);
                fifth.gaussianDensity(bigMat);
                waitKey(0);
                break;
            }
            case 61:
            {
                printf("No. clusters: ");
                scanf("%d", &clusters);
                printf("No. iterations: ");
                scanf("%d", &noIterations);
                sixth.computeKmean(clusters, 2, "Images/Kmeans/points2.bmp", noIterations);
                sixth.Voronoi("Images/Kmeans/points2.bmp", clusters, noIterations);
                waitKey();
                break;
            }
            case 62:
            {
                printf("No. clusters: ");
                scanf("%d", &clusters);
                printf("No. iterations: ");
                scanf("%d", &noIterations);
                sixth.computeKmean(clusters, 3, "Images/Kmeans/img01.jpg", noIterations);
                waitKey();
                break;
               
            }
            case 63: 
            {
                printf("No. clusters: ");
                scanf("%d", &clusters);
                printf("No. iterations: ");
                scanf("%d", &noIterations);
                sixth.computeKmean(clusters, 1, "Images/Kmeans/img01.jpg", noIterations);
                waitKey();
                break;
            }
            case 71:
            {
                seventh.computePCA("prs_res_PCA/pca3d.txt", 3);
                system("pause");
                waitKey(0);
                break;
            }
            case 72: 
            {
                seventh.imageUsingPCA2("prs_res_PCA/pca2d.txt", 2);
                waitKey(0);
                break;
            }
            case 73:
            {
                seventh.imageUsingPCA3("prs_res_PCA/pca3d.txt", 3);
                waitKey(0);
                break;
            }
            case 81:
            {
                int nr_bins = 8; // m = 8 acumulatoare
                int k = 0;
                printf("Number of neighbors: ");
                scanf("%d", &k);
                printf("Number of bins: ");
                scanf("%d", &nr_bins);

                trainSet = eigth.computeTrainSets(nr_bins);

                Mat kNN = eigth.computeClassifierMatrix(trainSet, k, nr_bins, 0);
                
                printf("Confusion matrix: \n");
                for (int i = 0; i < kNN.rows; ++i)
                {
                    for (int j = 0; j < kNN.cols; ++j)
                    {
                        printf("%d  ", kNN.at<int>(i, j));
                    }
                    printf("\n");
                }
                printf("\n\n");
                printf("Accuracy: %f\n", eigth.computeAccuracy(kNN));
                system("pause");

                break;
            }
            case 91:
            {
                vector<Mat> trainingSet = ninth.computeBayesTrainSet();
                Mat priors = ninth.computeAPriori(trainingSet);
                Mat likelihood = ninth.computeLikelihood(trainingSet);

                Mat bayes = ninth.computeBayesClassifier(trainingSet, priors, likelihood);

                for (int i = 0; i < bayes.rows; ++i)
                {
                    for (int j = 0; j < bayes.cols; ++j)
                    {
                        printf("%d  ", bayes.at<int>(i, j));
                    }
                    printf("\n");
                }
                printf("Accuracy: %f\n", eigth.computeAccuracy(bayes));

                system("pause");
                break;
            }
            case 101:
            {
                trainSet = tenth.computeParams("prs_res_Perceptron/test05.bmp");
                tenth.onlinePerceptron(trainSet, "prs_res_Perceptron/test05.bmp");
                waitKey(0);
                break;
            }
            case 102:
            {
                trainSet = tenth.computeParams("prs_res_Perceptron/test06.bmp");
                tenth.batchPerceptron(trainSet, "prs_res_Perceptron/test06.bmp");
                waitKey(0);
                break;
            }
            case 111:
            {
                int dim = 0;
                trainSet = eleventh.readAdaBoostPoints("prs_res_AdaBoost/points0.bmp", dim);
                struct EleventhLaboratory::classifier adaBoostClassifier = eleventh.adaBoost(trainSet, dim);
                eleventh.drawBoundary("prs_res_AdaBoost/points0.bmp", adaBoostClassifier);
                waitKey(0);
                break;
            }
            case 310:
            {
                char fname[MAX_PATH];
                openFileDlg(fname);
                Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

                Mat1b dark;
                DarkChannel(src, dark, 3);


                imshow("Source", src);
                imshow("Darkness", dark);
                waitKey(0);
                break;
            }
            case 999:
            {
                Mat src = imread("E:\\Repos\\Rainbows\\OpenCVApplication-VS2017_OCV340_basic\\Images\\robot2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
               
                Mat grad_x, grad_y;
                Mat abs_grad_x, abs_grad_y;
                int ddepth = CV_16S;
                int delta = 0;
                int scale = 1;
                Mat grad;
                /// Gradient X
                //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
                Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
                convertScaleAbs(grad_x, abs_grad_x);

                /// Gradient Y
                //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
                Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
                convertScaleAbs(grad_y, abs_grad_y);

                /// Total Gradient (approximate)
                addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
                Mat  dst, gauss;

                double k = 0.4;
                int pH = 50;
                int pL = (int)k*pH;
                GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
                Canny(gauss, dst, pL, pH, 3);
                
                
                Mat idt = dst.clone();

                imshow("Original", src);
                
                float alfa = 0.33;
                float beta = 0.98;

                const int l_iNeighbours[] = {  1, -1, 0, 0 };
                const int l_jNeighbours[] = {  0, 0, 1, -1 };
                int kernel = 4;
                for (int i = kernel; i < idt.rows - kernel; ++i)
                {
                    for (int j = kernel; j < idt.cols - kernel; ++j)
                    {
                        int max = INT_MIN;
                        int x = 0;
                        int y = 0;

                        for(int bb = 1; bb <= kernel; ++bb)
                        {
                            for (vfc::int32_t l_neighbour = 0; l_neighbour < 4; ++l_neighbour)
                            {
                                vfc::int32_t maxNeighbour =
                                    grad.at<uchar>(i + bb * l_iNeighbours[l_neighbour],
                                        j + bb * l_jNeighbours[l_neighbour]) *
                                    pow(beta, max(abs(bb * l_iNeighbours[l_neighbour]), abs(bb * l_jNeighbours[l_neighbour])));

                                if (maxNeighbour > max)
                                {
                                    max = maxNeighbour;
                                    x = i + bb * l_iNeighbours[l_neighbour];
                                    y = j + bb * l_jNeighbours[l_neighbour];
                                }
                            }
                        }

                        int overflow = alfa * grad.at<uchar>(i, j) + (1.0 - alfa) *
                            grad.at<uchar>(x, y) * pow(beta, max(abs(x - i), abs(y - j)));

                        if (overflow > 255)
                        {
                            overflow = 255;
                        }
                        if (overflow < 0)
                        {
                            overflow = 0;
                        }

                        idt.at<uchar>(i, j) = overflow;
                    }
                }

                imshow("DST", grad);
                imshow("IDT", idt);
                waitKey(0);
                break;
            }
        default:
            break;
        }
    } while (op != 0);
}
