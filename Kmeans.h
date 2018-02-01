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

#include <algorithm> //std::replace

/* *******************
 * Costants
 * ********************/

#define DATA_PATH "./points_k8_3kk.txt"
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
		const char *statement, cudaError_t err){
	if(err == cudaSuccess)
		return;
	std::cerr << "\n" << statement << " returned " << cudaGetErrorString(err) <<
			"(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}


/* Utility function to clean the centroids struct */
void eraseTempData(CentroidsCollection* centroids, int* clusterSizes){

	for(int i=0; i< K;i++){
		centroids->x[i] = 0;
		centroids->y[i] = 0;
		centroids->labels[i] = -1;
		clusterSizes[i] = 0;
	}
}


/**
 * Initialize all the data;
 */
void initDataset(PointsCollection* dataset){

	std::ifstream file;
	file.open(DATA_PATH);
	std::string line;
	if(file.is_open()){
		while ( getline (file,line) )
		{

			std::replace(line.begin(), line.end(), ',', ' ');  // replace ',' by ' '

			std::vector<float> array;
			std::stringstream ss(line);
			float temp;
			while (ss >> temp){
				array.push_back(temp);

			}
			dataset->addPoint(array[0],array[1]);
		}
		file.close();
	}
	else{
		printf("no");
	}

}


/**
 * Calculate euclideanDistance between (x2,y2) and (x1,y1)
 */
__host__ __device__ float euclideanDistance(const float& x2, const float& y2, const float& x1, const float& y1){
	return (float)sqrt(pow(y2-y1,2) + pow(x2-x1,2));
}


/**
 * Initialize k centroids choosing them from the population with tha max-min strategy
 */
void initCentroids(PointsCollection* dataset, CentroidsCollection* centroids, const int& k){


	int r_index = rand() % (dataset->size);

	/* Add the first random chosen centroid */
	centroids->addPoint(dataset->x[r_index], dataset->y[r_index], -1);

	/* Then I have to choose the K-1 remaining using max-min distance method */
	for (int i=1;i<k;i++){

		float max = -1;
		int max_p = 0;

		//For each point
		for (int j=0; j<dataset->size;j++)  {

			//initialize the min
			float min = euclideanDistance(dataset->x[j],dataset->y[j], centroids->x[0], centroids->y[0]);

			//For each centroid
			for (int i = 1; i < centroids->size; i++) {

				float dist = euclideanDistance(dataset->x[j],dataset->y[j], centroids->x[i], centroids->y[i]);
				if (dist < min)
					//update the min
					min = dist;
			}
			if (min > max) {

				//update the max and the index of the point which is the best candidate
				max = min;
				max_p = j;
			}
		}

		centroids->addPoint(dataset->x[max_p], dataset->y[max_p], -1);

	}
}


/**
 * Assign labels kernel function
 */
__global__ void assignLabels(PointsCollection* dataset, CentroidsCollection* centroids, int* n_change){

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < dataset->size){
		int label = 0;
		float tmp;
		float min = euclideanDistance(dataset->x[i], dataset->y[i], centroids->x[0], centroids->y[0]);
		for (int j=1;j<centroids->size;j++){
			tmp = euclideanDistance(dataset->x[i], dataset->y[i], centroids->x[j], centroids->y[j]);
			if(tmp < min){
				min = tmp;
				label = j;
			}
		}
		if(dataset->labels[i] != label)
			/* We can avoid the use of atomic because
			 * we don't really want to know the correct
			 * value of the variable but only if
			 * it's greater than 0*/
			//atomicAdd(n_change, 1);
			(*n_change)++;

		dataset->labels[i] = label;
	}
}


/**
 * Centroids updating kernel function
 */
__global__ void updateCentroids(PointsCollection* dataset, CentroidsCollection* centroids, int* clusterSizes){

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < dataset->size){

		atomicAdd(&(centroids->x[dataset->labels[i]]), dataset->x[i]);

		atomicAdd(&(centroids->y[dataset->labels[i]]), dataset->y[i]);

		atomicAdd(&(clusterSizes[dataset->labels[i]]), 1);
	}

}


/**
 * Sequential centroids updating phase
 */
void singleUpdateCentroids(PointsCollection* dataset, CentroidsCollection* centroids, int* clusterSizes){

	for(int j=0;j<centroids->size;j++){
		for(int i=0; i<dataset->size;i++){
			if(dataset->labels[i] == j){
				centroids->x[j]+=dataset->x[i];
				centroids->y[j]+=dataset->y[i];
				clusterSizes[j]++;
			}
		}
	}
}


/**
 * Sequential label assignment phase
 */
void singleAssignLabels(PointsCollection* dataset, CentroidsCollection* centroids, int* n_change){

	for (int i=0;i<dataset->size;i++){

		int label = 0;
		int tmp;
		float min = euclideanDistance(dataset->x[i], dataset->y[i], centroids->x[0], centroids->y[0]);
		for (int j=1;j<centroids->size;j++){
			tmp = euclideanDistance(dataset->x[i], dataset->y[i], centroids->x[j], centroids->y[j]);
			if(tmp < min){
				min = tmp;
				label = j;
			}
		}
		if(dataset->labels[i] != label)
			(*n_change)++;

		dataset->labels[i] = label;
	}
}


/*
 * Main function of the parallelized version
 */
void parallelEvolve(PointsCollection* dataset, CentroidsCollection* centroids){

		/* Initialize random seed */
		srand(time(NULL));

		/* Host */
		int* clusterSizes = (int*)malloc(K*sizeof(int)); //For the centroids updating phase
		int* nChange = (int*)malloc(sizeof(int)); //Keep track of how many points changed their cluster for the termination of the algorithm

		/* Device data */
		PointsCollection* d_dataset;
		CentroidsCollection* d_centroids;
		int* d_clusterSizes;
		int* d_nChange;

		/* Allocate device data */
		CUDA_CHECK_RETURN(cudaMalloc((void**) &d_dataset,sizeof(PointsCollection)));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &d_centroids,sizeof(CentroidsCollection)));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &d_clusterSizes,K*sizeof(int)));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &d_nChange,sizeof(int)));

		/* Initialize dataset and centroids */
		initDataset(dataset);
		initCentroids(dataset, centroids, K);
		centroids->displayData();

		/* Copy dataset to device */
		CUDA_CHECK_RETURN(cudaMemcpy(d_dataset,dataset, sizeof(PointsCollection),cudaMemcpyHostToDevice));

		/* Start the chronometer */
		auto start = std::chrono::high_resolution_clock::now();

		do{

			/* Initialize device variable to 0 before the kernel call */
			*nChange = 0;
			CUDA_CHECK_RETURN(cudaMemcpy(d_nChange,nChange, sizeof(int),cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(d_centroids,centroids, sizeof(CentroidsCollection),cudaMemcpyHostToDevice));

			/* Assign labels phase */
			assignLabels<<<ceil((double)(dataset->size)/BLOCK_SIZE),BLOCK_SIZE>>>(d_dataset, d_centroids, d_nChange);
			cudaDeviceSynchronize(); //force the host application to wait for all kernels to complete

			/* Centroids updating phase */
			eraseTempData(centroids,clusterSizes);
			CUDA_CHECK_RETURN(cudaMemcpy(d_centroids,centroids, sizeof(CentroidsCollection),cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(d_clusterSizes, clusterSizes, K*sizeof(int),cudaMemcpyHostToDevice));
			updateCentroids<<<ceil((double)(dataset->size)/BLOCK_SIZE),BLOCK_SIZE>>>(d_dataset,d_centroids,d_clusterSizes);
			cudaDeviceSynchronize();

			/* Copy all to host */
			CUDA_CHECK_RETURN(cudaMemcpy(centroids,d_centroids, sizeof(CentroidsCollection),cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaMemcpy(clusterSizes,d_clusterSizes, sizeof(int)*K,cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaMemcpy(nChange,d_nChange, sizeof(int),cudaMemcpyDeviceToHost));


			/* Then the host terminate the updating phase */
			for (int i=0; i<K; i++){
				centroids->x[i] = centroids->x[i]/clusterSizes[i];
				centroids->y[i] = centroids->y[i]/clusterSizes[i];
				centroids->labels[i] = i;
			}

			/* If you want to see the convergence trend */
			printf("\nnchange: %d",*nChange);

		}
		while(*nChange > 0); /* Terminate if there's no point which changed its cluster */

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> d = end-start;
		printf("\ntime: %f s\n",d.count());

		/* Copy back the labeled dataset */
		CUDA_CHECK_RETURN(cudaMemcpy(dataset,d_dataset, sizeof(PointsCollection),cudaMemcpyDeviceToHost));

		/* Free device memory */
		cudaFree(d_dataset);
		cudaFree(d_centroids);
		cudaFree(d_clusterSizes);
		cudaFree(d_nChange);

		/* Free temp host memory */
		free(clusterSizes);
		free(nChange);
}


/*
 * Main function of the sequential version
 * */
void sequentialEvolve(PointsCollection* dataset, CentroidsCollection* centroids){

	/* Host */
	int* clusterSizes = (int*)malloc(K*sizeof(int)); //For the centroids updating phase
	int* nChange = (int*)malloc(sizeof(int)); //Keep track of how many points changed their cluster for the termination of the algorithm

	/* Initialize dataset and centroids */
	initDataset(dataset);
	initCentroids(dataset, centroids, K);
	centroids->displayData();

	/* Start taking time */
	auto start = std::chrono::high_resolution_clock::now();

	/* Main loop */
	do{

		*nChange = 0;

		/* Assignment phase */
		singleAssignLabels(dataset,centroids,nChange);

		/* Updating phase */
		eraseTempData(centroids,clusterSizes);
		singleUpdateCentroids(dataset,centroids,clusterSizes);
		for (int i=0; i<K; i++){
			centroids->x[i] = centroids->x[i]/clusterSizes[i];
			centroids->y[i] = centroids->y[i]/clusterSizes[i];
			centroids->labels[i] = i;
		}

		/* If you want to see the convergence trend */
		printf("\nnchange: %d",*nChange);
	}
	while(*nChange > 0);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> d = end-start;
	printf("\ntime: %f s\n",d.count());

	/* Free temp host memory */
	free(clusterSizes);
	free(nChange);

}
