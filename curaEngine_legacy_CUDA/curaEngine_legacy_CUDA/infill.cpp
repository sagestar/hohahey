/** Copyright (C) 2013 David Braam - Released under terms of the AGPLv3 License */
#include "infill.h"
#include "polygonPlot.h"
namespace cura {
	void generateConcentricInfill_GPU(vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, Polygons outline, Polygons& result, int inset_value, long long polysCnt, int layerNr)
	{
			for (long long k = 1; k < scanlineStorage[polysCnt].size(); k = k + scanlineStorage[polysCnt][k].X) {
				//printf("%lld %lld\n", scanlineStorage[polysCnt][k].X, scanlineStorage[polysCnt][k].Y);
				long long num = scanlineStorage[polysCnt][k].X;
				ClipperLib::Path temp(&scanlineStorage[polysCnt][k + 1], &scanlineStorage[polysCnt][k + num - 1]);
				result.add(PolygonRef(temp));
				if (850==layerNr) {
					for (int i = 0; i < temp.size(); i += 1) {
						if (i + 1 == temp.size()) {
							linePlotMain(temp[i], temp[0], 500);
						}
						else {
							linePlotMain(temp[i], temp[i + 1], 500);
						}
					}
				}
			}
	}

	void generateAutomaticInfill(const Polygons& in_outline, Polygons& result,
		int extrusionWidth, int lineSpacing,
		int infillOverlap, double rotation)
	{
		if (lineSpacing > extrusionWidth * 4)
		{
			generateGridInfill(in_outline, result, extrusionWidth, lineSpacing,
				infillOverlap, rotation);
		}
		else
		{
			generateLineInfill(in_outline, result, extrusionWidth, lineSpacing,
				infillOverlap, rotation);
		}
	}
	void generateGridInfill_GPU(Polygons& result, int extrusionWidth, double rotation, vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum, long long polysCnt, int layerNr, vector<Point> modelSizeM)
	{
		if (layerNr == 500) {
			int b = abs(modelSizeM[0].Y - modelSizeM[1].Y) / 350;
			for (int k = 0; k < ptrForPoints.size(); ++k) {
				int j = ptrForPoints[k][polygonsNum[polysCnt + 1]];
				for (int i = ptrForPoints[k][polygonsNum[polysCnt]]; i < j; i += 2)
				{
					if (abs(scanlineStorage[k][i + 1].Y - scanlineStorage[k][i].Y) < extrusionWidth / 5)
						continue;
					//printf("(%lld,%lld)-(%lld,%lld)\n", scanlineStorage[0][i].X, scanlineStorage[0][i].Y, scanlineStorage[0][i + 1].X, scanlineStorage[0][i + 1].Y);
					linePlotMain(scanlineStorage[k][i] - modelSizeM[1], scanlineStorage[k][i + 1] - modelSizeM[1], b);
				}
			}
		}
	}
	void generateGridInfill(const Polygons& in_outline, Polygons& result,
		int extrusionWidth, int lineSpacing, int infillOverlap,
		double rotation)
	{
		generateLineInfill(in_outline, result, extrusionWidth, lineSpacing * 2,
			infillOverlap, rotation);
		generateLineInfill(in_outline, result, extrusionWidth, lineSpacing * 2,
			infillOverlap, rotation + 90);
	}

	int compare_int64_t(const void* a, const void* b)
	{
		int64_t n = (*(int64_t*)a) - (*(int64_t*)b);
		if (n < 0) return -1;
		if (n > 0) return 1;
		return 0;
	}

	int compare_intPoint_t(const void* a, const void* b)
	{
		ClipperLib::IntPoint n;
		n.X = (*(ClipperLib::IntPoint*)a).X - (*(ClipperLib::IntPoint*)b).X;
		n.Y = (*(ClipperLib::IntPoint*)a).Y - (*(ClipperLib::IntPoint*)b).Y;
		if (n.X < 0) return -1;
		if (n.X == 0 && n.Y < 0) return -1;
		if (n.Y == 0) return 0;
		return 1;
	}

	void generateLineInfill(const Polygons& in_outline, Polygons& result, int extrusionWidth, int lineSpacing, int infillOverlap, double rotation)
	{
		Polygons outline = in_outline.offset(extrusionWidth * infillOverlap / 100);
		PointMatrix matrix(rotation);
		outline.applyMatrix(matrix);

		AABB boundary(outline);

		boundary.min.X = ((boundary.min.X / lineSpacing) - 1) * lineSpacing;
		int lineCount = (boundary.max.X - boundary.min.X + (lineSpacing - 1)) / lineSpacing;
		vector<vector<int64_t> > cutList;
		for (int n = 0; n < lineCount; n++)
			cutList.push_back(vector<int64_t>());

		for (unsigned int polyNr = 0; polyNr < outline.size(); polyNr++)
		{
			Point p1 = outline[polyNr][outline[polyNr].size() - 1];
			for (unsigned int i = 0; i < outline[polyNr].size(); i++)
			{
				Point p0 = outline[polyNr][i];
				int idx0 = (p0.X - boundary.min.X) / lineSpacing;
				int idx1 = (p1.X - boundary.min.X) / lineSpacing;
				int64_t xMin = p0.X, xMax = p1.X;
				if (p0.X > p1.X) { xMin = p1.X; xMax = p0.X; }
				if (idx0 > idx1) { int tmp = idx0; idx0 = idx1; idx1 = tmp; }
				for (int idx = idx0; idx <= idx1; idx++)
				{
					int x = (idx * lineSpacing) + boundary.min.X + lineSpacing / 2;
					if (x < xMin) continue;
					if (x >= xMax) continue;
					int y = p0.Y + (p1.Y - p0.Y) * (x - p0.X) / (p1.X - p0.X);
					cutList[idx].push_back(y);
				}
				p1 = p0;
			}
		}
		int idx = 0;
		for (int64_t x = boundary.min.X + lineSpacing / 2; x < boundary.max.X; x += lineSpacing)
		{
			qsort(cutList[idx].data(), cutList[idx].size(), sizeof(int64_t), compare_int64_t);
			// for (unsigned int i = 0; i + 1 < cutList[idx].size(); i += 2)
			// {
			// 	if (cutList[idx][i + 1] - cutList[idx][i] < extrusionWidth / 5)
			// 		continue;
			// 	PolygonRef p = result.newPoly();
			// 	p.add(matrix.unapply(Point(x, cutList[idx][i])));
			// 	p.add(matrix.unapply(Point(x, cutList[idx][i + 1])));
			// }
			idx += 1;
		}
	}

	void generateLineInfill_GPU(Polygons& result, int extrusionWidth, double rotation, vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum, long long polysCnt, int layerNr, vector<Point> modelSizeM)
	{
		if (layerNr == 1450) {
			linePlotMaina(Point(300, 300), Point(300, 800), 2);
			linePlotMaina(Point(800, 300), Point(300, 300), 2);
			int b = abs(modelSizeM[0].Y - modelSizeM[1].Y) / 300;
			int j = ptrForPoints[0][polygonsNum[polysCnt + 1]];
			//printf("pointSize:%d\n", j - ptrForPoints[polysCnt]);
			for (int i = ptrForPoints[0][polygonsNum[polysCnt]]; i < j; i += 2)
			{
				if (abs(scanlineStorage[0][i + 1].Y - scanlineStorage[0][i].Y) < extrusionWidth / 5)
					continue;
				//printf("(%lld,%lld)-(%lld,%lld)\n", scanlineStorage[0][i].X, scanlineStorage[0][i].Y, scanlineStorage[0][i + 1].X, scanlineStorage[0][i + 1].Y);
				linePlotMain(scanlineStorage[0][i]-modelSizeM[1], scanlineStorage[0][i + 1] - modelSizeM[1], b);
			}


		}
		
	}


}//namespace cura
