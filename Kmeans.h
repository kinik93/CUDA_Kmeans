/* *******************
 * Dependencies
 * ********************/

#include "Collections.h"
#include <cuda.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <chrono>


/* *******************
 * Costants
 * ********************/

#define DATA_PATH "./points.txt"
#define BLOCK_SIZE 256



/**
 * function-like macro
 * __LINE__ = contains the line number of the currently compiled line of code
 * __FILE__ = string that contains the name of the source file being compiled
 * # operator = turns the argument it precedes into a quoted string
 * Reference: [C the complete reference]
 * check with > nvcc -E
 */
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
static void CheckCudaErrorAux (const char *file, unsigned line,
		const char *statement, cudaError_t err);

/* Utility function to clean the centroids struct */
void eraseTempData(CentroidsCollection* centroids, int* clusterSizes);


/**
 * Initialize all the data;
 */
void initDataset(PointsCollection* dataset);


/**
 * Calculate euclideanDistance between (x2,y2) and (x1,y1)
 */
__host__ __device__ float euclideanDistance(const float& x2, const float& y2, const float& x1, const float& y1);

/**
 * Initialize k centroids choosing them from the population with tha max-min strategy
 */
void initCentroids(PointsCollection* dataset, CentroidsCollection* centroids, const int& k);


/**
 * Assign labels kernel function
 */
__global__ void assignLabels(PointsCollection* dataset, CentroidsCollection* centroids, int* n_change);


/**
 * Centroids updating kernel function
 */
__global__ void updateCentroids(PointsCollection* dataset, CentroidsCollection* centroids, int* clusterSizes);

/**
 * Sequential centroids updating phase
 */
void singleUpdateCentroids(PointsCollection* dataset, CentroidsCollection* centroids, int* clusterSizes);


/**
 * Sequential label assignment phase
 */
void singleAssignLabels(PointsCollection* dataset, CentroidsCollection* centroids, int* n_change);


/*
 * Main function of the parallelized version
 */
void parallelEvolve(PointsCollection* dataset, CentroidsCollection* centroids);

/*
 * Main function of the sequential version
 * */
void sequentialEvolve(PointsCollection* dataset, CentroidsCollection* centroids);
