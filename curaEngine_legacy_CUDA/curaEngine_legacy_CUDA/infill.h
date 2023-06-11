/** Copyright (C) 2013 David Braam - Released under terms of the AGPLv3 License */
#ifndef INFILL_H
#define INFILL_H

#include "utils/polygon.h"

namespace cura {

	void generateConcentricInfill_GPU(vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, Polygons outline, Polygons& result, int inset_value, long long polysCnt, int layerNr);
	void generateGridInfill_GPU(Polygons& result, int extrusionWidth, double rotation, vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum, long long polysCnt, int layerNr, vector<Point> modelSizeM);
	void generateLineInfill_GPU(Polygons& result, int extrusionWidth, double rotation, vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum, long long polysCnt, int layerNr, vector<Point> modelSizeM);
	void generateAutomaticInfill(const Polygons& in_outline, Polygons& result, int extrusionWidth, int lineSpacing, int infillOverlap, double rotation);
	void generateGridInfill(const Polygons& in_outline, Polygons& result, int extrusionWidth, int lineSpacing, int infillOverlap, double rotation);
	void generateLineInfill(const Polygons& in_outline, Polygons& result, int extrusionWidth, int lineSpacing, int infillOverlap, double rotation);

}//namespace cura

#endif//INFILL_H
