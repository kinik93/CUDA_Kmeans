
#include "Kmeans.h"

/* Pameters of the algorithm are in the Collections and Kmeans headers */

int main(int argc, char** argv){


	/* Host data */
	PointsCollection* dataset = (PointsCollection*) malloc(sizeof(PointsCollection));
	CentroidsCollection* centroids = (CentroidsCollection*) malloc(sizeof(CentroidsCollection));

	/* Uncomment to choose sequential or parallel version */
	parallelEvolve(dataset,centroids);
	//sequentialEvolve(dataset,centroids);

	printf("\n\nFinal centroids:");
	centroids->displayData();

	/* Free used memory */
	free(dataset);
	free(centroids);

	return 0;
}
