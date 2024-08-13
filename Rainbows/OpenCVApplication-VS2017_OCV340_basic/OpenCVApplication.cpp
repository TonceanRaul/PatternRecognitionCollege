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

#include "Simulation.h"

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

void imageDehazer(Mat3b img, Mat1b& dark, Mat3b& dehazedImage)
{
	for (int i = 0; i < dark.rows; ++i)
	{
		for (int j = 0; j < dark.cols; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				int aux = img(i, j)[k] * (1.35) - (0.85) * dark(i, j);

				computeOverflows<int>(0, 255, &aux);

				dehazedImage(i, j)[k] = aux;
			}
		}
	}

}

void DarkChannel(const Mat3b& img, Mat1b& dark, int patchSize)
{
	int radius = patchSize / 2;

	Mat1b low;
	minValue3b(img, low);

	imshow("MinValue", low);

	minFilter(low, dark, radius);

	("MinFilter", dark);

	Mat3b dehazedImage(dark.rows, dark.cols, CV_8UC3);
	imageDehazer(img, dark, dehazedImage);

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
				imshow("Big", bigMat);
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
				trainSet = eleventh.readAdaBoostPoints("prs_res_AdaBoost/points5.bmp", dim);
				struct EleventhLaboratory::classifier adaBoostClassifier = eleventh.adaBoost(trainSet, dim);
				eleventh.drawBoundary("prs_res_AdaBoost/points5.bmp", adaBoostClassifier);
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
			}
			case 999:
			{
				Simulation simulation;
				srand((unsigned int)time(NULL));

				simulation.initMatrix();

				Scalar intensity;

				Mat image = Mat(700, 700, CV_8UC3);
				Mat statImage = Mat(300, 300, CV_8UC3);

				int scaleRows = 700 / N_MAX;
				int scaleCols = 700 / N_MAX;

				do
				{
					image.setTo(cv::Scalar(255, 255, 255));
					statImage.setTo(cv::Scalar(255, 255, 255));

					int sumPredator = 0;
					int sumPray = 0;

					for (int i = 0; i < N_MAX; ++i)
					{
						for (int j = 0; j < N_MAX; ++j)
						{
							if (simulation.firstMatrix[i][j] == 1)
							{
								sumPray++;
								intensity = Scalar(255, 0, 0);
								circle(image, Point(i * scaleRows, j * scaleCols), 2, intensity, CV_FILLED);
							}
							else if (simulation.firstMatrix[i][j] == -1)
							{
								sumPredator++;
								intensity = Scalar(0, 0, 255);
								circle(image, Point(i * scaleRows, j * scaleCols), 2, intensity, CV_FILLED);
							}
							else
							{
								// empty
							}
						}
					}

					int factor = max(sumPray, sumPredator);

					for (int i = 1; i < N_MAX_STAT; ++i)
					{
						simulation.statistics[0][i - 1] = simulation.statistics[0][i];
						simulation.statistics[1][i - 1] = simulation.statistics[1][i];
						factor = max(factor, max(simulation.statistics[0][i], simulation.statistics[1][i]));
					}

					simulation.statistics[0][N_MAX_STAT - 1] = sumPray;
					simulation.statistics[1][N_MAX_STAT - 1] = sumPredator;

					double divFactor = factor / (double)statImage.size().height;

					for (int i = 0; i < N_MAX_STAT; ++i)
					{
						intensity = Scalar(255, 0, 0);
						circle(statImage, Point(i * 2, statImage.size().height - simulation.statistics[0][i] / divFactor), 2, intensity, CV_FILLED);

						intensity = Scalar(0, 0, 255);
						circle(statImage, Point(i * 2, statImage.size().height - simulation.statistics[1][i] / divFactor), 2, intensity, CV_FILLED);
					}

					putText(statImage, "Pray: " + to_string(sumPray) + "; Predator: " + to_string(sumPredator),
						Point(10, statImage.size().height - 10), 0, 0.5, Scalar(0, 0, 0));

					memcpy(simulation.secondMatrix, simulation.firstMatrix, sizeof(int) * N_MAX * N_MAX);

					simulation.reproducePrey(PRAY_PROB);
					simulation.reproducePredator(PREDATOR_PROB);
					simulation.removeThePredator();
					simulation.prayDies(PRAY_DEATH_RATE);
					simulation.predatorDies(PREDATOR_DEATH_RATE);

					memcpy(simulation.firstMatrix, simulation.secondMatrix, sizeof(int) * N_MAX * N_MAX);

					imshow("Output Window", image);
					imshow("Statistics Window", statImage);
				} while (!(waitKey() == 27));

			}
		default:
			break;
		}
	} while (op != 0);
}
