#include "cudaInfill.h"
#include <device_launch_parameters.h>
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "device_functions.h"
#include <ctime>
#define threadNUM 256
namespace cura {
	void cudaGridInfill(vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints, vector< vector<Point> >& scanlineStorage,
		vector< vector<int> >& scanlineStorageInfo, int scanlineSpacing, int modelMin) {
		cudaLineInfill(polygonInfo, polygonPoints, scanlineStorage, scanlineStorageInfo, scanlineSpacing, modelMin, 0);
		cudaLineInfill(polygonInfo, polygonPoints, scanlineStorage, scanlineStorageInfo, scanlineSpacing, modelMin, 60);
		cudaLineInfill(polygonInfo, polygonPoints, scanlineStorage, scanlineStorageInfo, scanlineSpacing, modelMin, 120);
	}
}