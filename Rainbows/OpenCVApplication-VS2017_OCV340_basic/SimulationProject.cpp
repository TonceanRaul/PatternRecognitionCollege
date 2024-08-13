#include "stdafx.h"

#include "Simulation.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace std;
using namespace cv;

//#define RANDOM
#define BLOB
//#define WAVE

void Simulation::initMatrix()
{
#ifdef RANDOM
	for (int i = 0; i < N_MAX; ++i)
	{
		for (int j = 0; j < N_MAX; ++j)
		{
			double random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

			if (random < 0.25)
			{
				// it's a predator
				firstMatrix[i][j] = -1;
				secondMatrix[i][j] = -1;
			}
			else if (random < 0.5)
			{
				// it's a pray
				firstMatrix[i][j] = 1;
				secondMatrix[i][j] = 1;
			}
			else
			{
				// it's an empty spot
				firstMatrix[i][j] = 0;
				secondMatrix[i][j] = 0;
			}
		}
	}
#endif

#ifdef BLOB
	
	for (int j = 25; j < 45; ++j) 
	{
		for (int i = 25; i < 45; ++i)
		{
			firstMatrix[j][i] = 1;
			secondMatrix[j][i] = 1;
		}
	}
	
	for (int i = 26; i < 45; ++i)
	{
		firstMatrix[i][44] = -1;
		secondMatrix[i][44] = -1;
	}
#endif

#ifdef WAVE
	for (int j = 0; j < N_MAX; ++j)
	{
		firstMatrix[j][13] = 1;
		firstMatrix[j][14] = 1;
		secondMatrix[j][13] = 1;
		secondMatrix[j][14] = 1;

		firstMatrix[j][15] = -1;
		secondMatrix[j][15] = -1;
	}
#endif 
}

void Simulation::reproducePrey(float prob)
{
	for (int i = 0; i < N_MAX; ++i)
	{
		for (int j = 0; j < N_MAX; ++j)
		{
			if (firstMatrix[i][j] == 1)
			{
				for (int k = 0; k < 8; ++k)
				{
					int new_x = i + dx[k];
					int new_y = j + dy[k];

					if (new_x < 0) new_x = N_MAX - 1;
					if (new_y < 0) new_y = N_MAX - 1;
					if (new_x == N_MAX) new_x = 0;
					if (new_y == N_MAX) new_y = 0;

					if (firstMatrix[new_x][new_y] == 0)
					{
						double random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
						if (random <= prob)
						{
							secondMatrix[new_x][new_y] = 1;
						}
					}
				}
			}
		}
	}
}

void Simulation::reproducePredator(float prob)
{
	for (int i = 0; i < N_MAX; ++i)
	{
		for (int j = 0; j < N_MAX; ++j)
		{
			if (firstMatrix[i][j] == -1)
			{
				for (int k = 0; k < 8; ++k)
				{
					int new_x = i + dx[k];
					int new_y = j + dy[k];

					if (new_x < 0) new_x = N_MAX - 1;
					if (new_y < 0) new_y = N_MAX - 1;
					if (new_x == N_MAX) new_x = 0;
					if (new_y == N_MAX) new_y = 0;

					if (firstMatrix[new_x][new_y] == 1)
					{
						double random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
						if (random <= prob)
						{
							secondMatrix[new_x][new_y] = -1;
						}
					}
				}
			}
		}
	}
}

void Simulation::removeThePredator()
{
	for (int i = 0; i < N_MAX; ++i)
	{
		for (int j = 0; j < N_MAX; ++j)
		{
			if (secondMatrix[i][j] == -1)
			{
				secondMatrix[i][j] = 0;
				for (int k = 0; k < 8; ++k)
				{
					int new_x = i + dx[k];
					int new_y = j + dy[k];

					if (new_x < 0) new_x = N_MAX - 1;
					if (new_y < 0) new_y = N_MAX - 1;
					if (new_x == N_MAX) new_x = 0;
					if (new_y == N_MAX) new_y = 0;

					if (secondMatrix[new_x][new_y] == 1)
					{
						secondMatrix[i][j] = -1;
						break;
					}
				}
			}
		}
	}
}

void Simulation::prayDies(float prob)
{
	for (int i = 0; i < N_MAX; ++i)
	{
		for (int j = 0; j < N_MAX; ++j)
		{
			if (secondMatrix[i][j] == 1)
			{
				double random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				if (random <= prob)
				{
					secondMatrix[i][j] = 0;
				}
			}
		}
	}
}

void Simulation::predatorDies(float prob)
{
	for (int i = 0; i < N_MAX; ++i)
	{
		for (int j = 0; j < N_MAX; ++j)
		{
			if (secondMatrix[i][j] == -1)
			{
				double random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				if (random <= prob)
				{
					secondMatrix[i][j] = 0;
				}
			}
		}
	}
}
