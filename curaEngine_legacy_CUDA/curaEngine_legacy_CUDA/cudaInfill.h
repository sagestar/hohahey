#pragma once
#include "fffProcessor.h"
#include <vector>
#include <cuda.h>
#define blockNUM 8

namespace cura {
	/*
	* Realize Line Infill in GPU
	* @Ma Chuangxin
	*/
	void cudaLineInfill(vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints, vector< vector<Point> >& scanlineStorage,
		vector< vector<int> >& scanlineStorageInfo, int scanlineSpacing, int modelMin, int direction);

	/*
	* Realize Grid Infill in GPU
	* @Ma Chuangxin
	*/
	void cudaGridInfill(vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints, vector< vector<Point> >& scanlineStorage,
		vector< vector<int> >& scanlineStorageInfo, int scanlineSpacing, int modelMin);

	/*
	* Realize Concentric Infill By Pixel Structure in GPU
	* @Ma Chuangxin
	*/
	void cudaConcentricInfillByPixel(vector<int>& polygonsNum, vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints,
		vector< vector<Point> >& scanlineStorage, vector< vector<int> >& scanlineStorageInfo, int scanlineSpacing, int3 modelMin3D, int3 modelMax3D);

	/*
	* Realize Grid Infill By Pixel Structure in GPU
	* @Ma Chuangxin
	*/
	void cudaGridInfillByPixel(vector<int>& polygonsNum, vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints,
		vector< vector<Point> >& scanlineStorage, vector< vector<int> >& scanlineStorageInfo, int scanlineSpacing, int3 modelMin3D, int3 modelMax3D);
	
	/*
	* Get Boundary Values
	* @Ma Chuangxin
	*/
	int getModelMin(Point3 boundryMin, Point3 boundryMax, int infillPattern);

	/*
	* Convert the Data to Facilitate GPU Operations
	* @Ma Chuangxin
	*/
	void preProcess(fffProcessor& processor, SliceDataStorage& storage, vector<int>& polygonsNum, vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints);

	/*
	* Complete the Filling According to the Parameter
	* @Ma Chuangxin
	*/
	bool cudaInfill(fffProcessor& processor, SliceDataStorage& storage, vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum);
	//void cudaConcentricInfillByLDNI(vector<int>& polygonsNum, vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints, 
	//	vector< vector<Point> >& scanlineStorage,vector< vector<int> >& scanlineStorageInfo, 
	//	int scanlineSpacing, int2 modelSize);
}