
#define N_MAX 100
#define N_MAX_STAT 100
#define PRAY_PROB 0.75
#define PREDATOR_PROB 0.8
#define PRAY_DEATH_RATE 0.2
#define PREDATOR_DEATH_RATE 0.1

class Simulation 
{
public:
	void removeThePredator();
	void reproducePredator(float prob);
	void reproducePrey(float prob);
	void initMatrix();
	void prayDies(float prob);
	void predatorDies(float prob);

	int firstMatrix[N_MAX][N_MAX];
	int secondMatrix[N_MAX][N_MAX];
	int statistics[2][N_MAX_STAT];
	int dx[8] = { -1,-1,-1, 0, 0, 1, 1, 1 };
	int dy[8] = { -1, 0, 1,-1, 1,-1, 0, 1 };
};