#include "cudaInfill.h"

namespace cura {

	int getModelMin(Point3 boundryMin, Point3 boundryMax, int infillPattern) {
		int modelMin;
		switch (infillPattern) {
		case INFILL_CONCENTRIC:
			modelMin = (boundryMin.x < boundryMin.y) ? boundryMin.x : boundryMin.y;
			break;
		default:
			modelMin = boundryMin.y;
			Point3* pointIdx[2];
			pointIdx[0] = &boundryMin;
			pointIdx[1] = &boundryMax;
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					for (int k = 0; k <= 2; ++k) {
						PointMatrix matrix(45 + k * 90);
						Point p = matrix.apply(Point(pointIdx[i]->x, pointIdx[j]->y));
						modelMin = (p.Y < modelMin) ? p.Y : modelMin;
					}
				}
			}
		}
		return modelMin;
	}
	int getModelMax(Point3 boundryMin, Point3 boundryMax, int infillPattern) {
		int modelMax;
		switch (infillPattern) {
		case INFILL_CONCENTRIC:
			modelMax = (boundryMax.x > boundryMax.y) ? boundryMax.x : boundryMax.y;
			break;
		default:
			modelMax = boundryMax.y;
			Point3* pointIdx[2];
			pointIdx[0] = &boundryMin;
			pointIdx[1] = &boundryMax;
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					for (int k = 0; k < 2; ++k) {
						PointMatrix matrix(45 + k * 90);
						Point p = matrix.apply(Point(pointIdx[i]->x, pointIdx[j]->y));
						modelMax = (p.Y > modelMax) ? p.Y : modelMax;
					}
				}
			}
		}
		return modelMax;
	}

	void preProcess(fffProcessor& processor, SliceDataStorage& storage, vector<int>& polygonsNum, vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints) {
		unsigned int totalLayers = storage.volumes[0].layers.size();

		int volumeIdx = 0;
		int polygonsIdx = 0;
		long long storageLength = 0;
		int fillAngle = 0;
		for (unsigned int layerNr = 0; layerNr < totalLayers; layerNr++) {
			if (processor.config.infillPattern != INFILL_CONCENTRIC) 
				fillAngle = 45 + layerNr % 2 * 90;

			for (unsigned int volumeCnt = 0; volumeCnt < storage.volumes.size(); volumeCnt++) {
				if (volumeCnt > 0)
					volumeIdx = (volumeIdx + 1) % storage.volumes.size();
				SliceLayer* layer = &storage.volumes[volumeIdx].layers[layerNr];

				for (unsigned int partCounter = 0; partCounter < layer->parts.size(); partCounter++, polygonsIdx++) {
					Polygons outline = layer->parts[partCounter].sparseOutline;
					polygonsNum.push_back(polygonInfo.size());
					for (unsigned int polyNr = 0; polyNr < outline.size(); polyNr++) {
						PolygonRef polyOutRef = outline[polyNr];
						polygonInfo.push_back(pointPtr(storageLength, fillAngle, polygonsIdx));
						if (polyOutRef.size()) 
							polyOutRef.add(polyOutRef[0]);
						polygonPoints.insert(polygonPoints.end(), polyOutRef.begin(), polyOutRef.end());
						storageLength += polyOutRef.size();
					}
					if(!outline.size()) 
						polygonInfo.push_back(pointPtr(storageLength, fillAngle, polygonsIdx));
				}
			}
		}
		polygonInfo.push_back(pointPtr(storageLength, 0, 0));
		polygonsNum.push_back(polygonInfo.size());
	}

	bool cudaInfill(fffProcessor& processor, SliceDataStorage& storage, vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum) {
		vector<pointPtr> polygonInfo;//轮廓起始点在数组中的位置
		ClipperLib::Path polygonPoints;//存放轮廓信息
		preProcess(processor, storage, polygonsNum, polygonInfo, polygonPoints);
		int modelMin1D;
		int3 modelMin3D;
		int3 modelMax3D;
		switch (processor.config.infillPattern) {
		case INFILL_AUTOMATIC:
			break;
		case INFILL_GRID:
			modelMin1D = getModelMin(storage.modelMin, storage.modelMax, processor.config.infillPattern);
			cudaGridInfill(polygonInfo, polygonPoints, scanlineStorage, ptrForPoints, processor.config.sparseInfillLineDistance, modelMin1D);
			break;
		case INFILL_LINES:
			modelMin1D = getModelMin(storage.modelMin, storage.modelMax, processor.config.infillPattern);
			cudaLineInfill(polygonInfo, polygonPoints, scanlineStorage, ptrForPoints, processor.config.sparseInfillLineDistance, modelMin1D, 0);
			break;
		case INFILL_CONCENTRIC:
			modelMin3D.x = storage.modelMin.x;
			modelMin3D.y = storage.modelMin.y;
			modelMax3D.x = storage.modelMax.x;
			modelMax3D.y = storage.modelMax.y;
			modelMax3D.z = polygonsNum.size();
			cudaGridInfillByPixel(polygonsNum, polygonInfo, polygonPoints, scanlineStorage, ptrForPoints, processor.config.sparseInfillLineDistance, modelMin3D, modelMax3D);
			break;
		default:
			fprintf(stderr, "InfillPattern error\n");
		}
	}

}