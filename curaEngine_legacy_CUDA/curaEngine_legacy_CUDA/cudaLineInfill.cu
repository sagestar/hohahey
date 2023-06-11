#include "cudaInfill.h"
#include <device_launch_parameters.h>
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "device_functions.h"
#include <ctime>
#include <cmath>
#define threadNUM 256

namespace cura {
	__host__ __device__ int computeScanSegmentIdxKernel(Point p, int lineSpacing)
	{
		return (p.X < 0) ? (p.X + 1) / lineSpacing - 1 : p.X / lineSpacing;
		// - 1 because -1 belongs to scansegment -1
		// + 1 because -line_width belongs to scansegment -1
	}

	/*
	* 计算扫描线数据所需占用的内存大小
	*/
	__global__ void cudaComputeScanlineStorageInfo(int polygonCnt, pointPtr* dPolygonInfo, ClipperLib::IntPoint* dPolygonPoints, int lineSpacing, int* dScanlineStorageInfo, int direction) {
		unsigned int polygonIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (polygonIdx < polygonCnt - 1) {
			PointMatrix matrix(dPolygonInfo[polygonIdx].rotation + direction);
			int startLocation = dPolygonInfo[polygonIdx].location;
			int endLocation = dPolygonInfo[polygonIdx + 1].location;
			ClipperLib::IntPoint startPoint;
			ClipperLib::IntPoint endPoint;
			int startScanSegmentIdx;
			int endScanSegmentIdx;

			int scanPointsCnt = 0;
			startPoint = matrix.apply(dPolygonPoints[startLocation++]);
			for (; startLocation < endLocation; ++startLocation) {
				endPoint = matrix.apply(dPolygonPoints[startLocation]);
				startScanSegmentIdx = computeScanSegmentIdxKernel(startPoint, lineSpacing);
				endScanSegmentIdx = computeScanSegmentIdxKernel(endPoint, lineSpacing);
				scanPointsCnt += abs(startScanSegmentIdx - endScanSegmentIdx);
				startPoint = endPoint;
			}
			dScanlineStorageInfo[polygonIdx + 1] = scanPointsCnt;
		}
	}

	/*
	* 计算扫描线数据
	*/
	__global__ void cudaComputeScanline(int polygonCnt, pointPtr* dPolygonInfo, ClipperLib::IntPoint* dPolygonPoints, int lineSpacing,
		int* dScanlineStorageInfo, ClipperLib::IntPoint* dScanlineStorage, int modelMin, int direction) {
		int polygonIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (polygonIdx < polygonCnt - 1) {
			PointMatrix matrix(dPolygonInfo[polygonIdx].rotation + direction);
			int startLocation = dPolygonInfo[polygonIdx].location;
			int endLocation = dPolygonInfo[polygonIdx + 1].location;
			int scanPointsCnt = dScanlineStorageInfo[polygonIdx];
			ClipperLib::IntPoint startPoint;
			ClipperLib::IntPoint endPoint;

			int startScanSegmentIdx;
			int endScanSegmentIdx;
			ClipperLib::IntPoint currPoint;
			long long tempx;
			long long tempy;
			startPoint = matrix.apply(dPolygonPoints[startLocation++]);
			for (; startLocation < endLocation; ++startLocation) {
				endPoint = matrix.apply(dPolygonPoints[startLocation]);
				startScanSegmentIdx = computeScanSegmentIdxKernel(startPoint, lineSpacing) + 1;
				endScanSegmentIdx = computeScanSegmentIdxKernel(endPoint, lineSpacing);
				if (startPoint.X >= endPoint.X) {
					int temp = startScanSegmentIdx - 1;
					startScanSegmentIdx = endScanSegmentIdx + 1;
					endScanSegmentIdx = temp;
				}
				for (; startScanSegmentIdx <= endScanSegmentIdx; ++startScanSegmentIdx, ++scanPointsCnt) {
					tempx = startScanSegmentIdx * lineSpacing;
					tempy = (-modelMin + endPoint.Y + (startPoint.Y - endPoint.Y) * (tempx - endPoint.X) / (startPoint.X - endPoint.X)) & 0x00000000ffffffff;
					currPoint.X = dPolygonInfo[polygonIdx].layerNr;
					tempx = tempx & 0x00000000ffffffff;
					tempx = tempx << 32;
					currPoint.Y = tempx | tempy;
					dScanlineStorage[scanPointsCnt] = currPoint;
				}
				startPoint = endPoint;
			}
		}
	}

	/*
	* 对扫描线数据进行还原
	*/
	__global__ void cudaTranslatePoint(pointPtr* dPolygonInfo, int* dScanlineStorageInfo, ClipperLib::IntPoint* dScanlineStorage, int modelMin, int direction) {
		int polygonIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
		PointMatrix matrix(dPolygonInfo[polygonIdx].rotation + direction);
		ClipperLib::IntPoint p;

		int tempx;
		int tempy;
		int startLocation = dScanlineStorageInfo[polygonIdx];
		int endLocation = dScanlineStorageInfo[polygonIdx + 1];
		for (; startLocation < endLocation; ++startLocation) {
			p = dScanlineStorage[startLocation];
			tempx = (p.Y & 0xffffffff00000000) >> 32;
			tempy = p.Y & 0x00000000ffffffff;
			tempy += modelMin;
			p.X = tempx;
			p.Y = tempy;
			dScanlineStorage[startLocation] = matrix.unapply(p);
		}
	}

	void cudaLineInfill(vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints, vector< vector<Point> >& scanlineStorage, 
		vector< vector<int> >& scanlineStorageInfo, int scanlineSpacing, int modelMin, int direction) {
		cudaError_t cudaStatus;
		
		int dPolygonCnt = (polygonInfo.size() + threadNUM - 1) / threadNUM * threadNUM;
		pointPtr *dPolygonInfo;
		cudaStatus = cudaMalloc(&dPolygonInfo, dPolygonCnt * sizeof(pointPtr));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc fail!\n");
		}
		cudaStatus = cudaMemcpy(dPolygonInfo, &polygonInfo[0], polygonInfo.size() * sizeof(pointPtr), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy fail!\n");
		}

		int *dScanlineStorageInfo;
		cudaStatus = cudaMalloc(&dScanlineStorageInfo, dPolygonCnt * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc fail!\n");
		}

		int dPolygonPointsCnt = (polygonPoints.size() + threadNUM - 1) / threadNUM * threadNUM;
		ClipperLib::IntPoint* dPolygonPoints;
		cudaStatus = cudaMalloc(&dPolygonPoints, sizeof(ClipperLib::IntPoint) * dPolygonPointsCnt);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
		}
		cudaStatus = cudaMemcpy(dPolygonPoints, &polygonPoints[0], sizeof(ClipperLib::IntPoint) * polygonPoints.size(), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
		}

		/*Step 1*/
		cudaComputeScanlineStorageInfo << <(polygonInfo.size() + threadNUM - 1) / threadNUM, threadNUM >> > (polygonInfo.size(), dPolygonInfo, dPolygonPoints, scanlineSpacing, dScanlineStorageInfo, direction);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			printf("%s\n", cudaGetErrorString(cudaStatus));
		}

		/*Step 2*/
		thrust::inclusive_scan(thrust::device_pointer_cast(dScanlineStorageInfo), thrust::device_pointer_cast(dScanlineStorageInfo + dPolygonCnt), thrust::device_pointer_cast(dScanlineStorageInfo));
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			printf("%s\n", cudaGetErrorString(cudaStatus));
		}

		int scanlineStorageSize;
		cudaMemcpy(&scanlineStorageSize, &(dScanlineStorageInfo[polygonInfo.size()]), sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
		}

		int dScanlineStorageSize = (scanlineStorageSize + threadNUM - 1) / threadNUM * threadNUM;
		ClipperLib::IntPoint* dScanlineStorage;
		cudaStatus = cudaMalloc(&dScanlineStorage, sizeof(ClipperLib::IntPoint) * dScanlineStorageSize);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
		}

		/*Step 3*/
		cudaComputeScanline << <(polygonInfo.size() + threadNUM - 1) / threadNUM, threadNUM >> > (polygonInfo.size(), dPolygonInfo, dPolygonPoints, scanlineSpacing, dScanlineStorageInfo, dScanlineStorage, modelMin, direction);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			printf("%s\n", cudaGetErrorString(cudaStatus));
		}

		thrust::sort(thrust::device_pointer_cast(dScanlineStorage), thrust::device_pointer_cast(dScanlineStorage + scanlineStorageSize));
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			printf("%s\n", cudaGetErrorString(cudaStatus));
		}

		/*Step 4*/
		cudaTranslatePoint << <(polygonInfo.size() + threadNUM - 1) / threadNUM, threadNUM >> > (dPolygonInfo, dScanlineStorageInfo, dScanlineStorage, modelMin, direction);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			printf("%s\n", cudaGetErrorString(cudaStatus));
		}

		scanlineStorageInfo.push_back(vector<int>(polygonInfo.size()));
		cudaStatus = cudaMemcpy(&scanlineStorageInfo.back()[0], dScanlineStorageInfo, polygonInfo.size() * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
		}

		scanlineStorage.push_back(vector<Point>(scanlineStorageSize));
		cudaStatus = cudaMemcpy(&scanlineStorage.back()[0], dScanlineStorage, scanlineStorageSize * sizeof(ClipperLib::IntPoint), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaFree(dPolygonPoints);
		cudaFree(dPolygonInfo);
		cudaFree(dScanlineStorage);
		cudaFree(dScanlineStorageInfo);
	}
}