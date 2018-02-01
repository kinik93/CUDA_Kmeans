/*
 * Collections.h
 *
 *  Created on: 12/gen/2018
 *      Author: tommasoaldinucci
 */

#ifndef COLLECTIONS_H_
#define COLLECTIONS_H_

#include <iostream>

#define MAX_POINTS 3000000
#define K 5


/* Struct that wraps the dataset using a coalesced memory access pattern */
struct PointsCollection{

	float x[MAX_POINTS];
	float y[MAX_POINTS];
	int labels[MAX_POINTS];
	int size;

	PointsCollection(){size = 0;}

	void addPoint(float xC, float yC){
		if(size < MAX_POINTS){
			x[size] = xC;
			y[size] = yC;
			labels[size] = -1;
			size++;
		}
	}

	void addPoint(float xC, float yC, int label){
		if(size < MAX_POINTS){
			x[size] = xC;
			y[size] = yC;
			labels[size] = label;
			size++;
		}
	}

	void displayData(){
		for(int i=0; i<size; i++){
			printf("\nx[%d] = %f ; y[%d] = %f ; labels[%d] = %d",i,x[i],i,y[i],i,labels[i]);
		}
	}
};


/* Struct that wraps the dataset using a coalesced memory access pattern */
struct CentroidsCollection{

	float x[K];
	float y[K];
	int labels[K];
	int size;

	CentroidsCollection(){size = 0;}

	void addPoint(float xC, float yC){
		if(size < MAX_POINTS){
			x[size] = xC;
			y[size] = yC;
			labels[size] = -1;
			size++;
		}
	}

	void addPoint(float xC, float yC, int label){
		if(size < MAX_POINTS){
			x[size] = xC;
			y[size] = yC;
			labels[size] = label;
			size++;
		}
	}

	void displayData(){
		for(int i=0; i<size; i++){
			printf("\nx[%d] = %f ; y[%d] = %f ; labels[%d] = %d",i,x[i],i,y[i],i,labels[i]);
		}
	}
};
#endif /* COLLECTIONS_H_ */
