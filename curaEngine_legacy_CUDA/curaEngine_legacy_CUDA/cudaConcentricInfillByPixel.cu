#include <device_launch_parameters.h>
#include <device_functions.h>
#include<fstream>
#include <ctime>
#include "cudaInfill.h"
#define threadNUM 256
#define CUDA_CALL(x) {const cudaError a = (x); if(a != cudaSuccess){printf("\nCUDA Error:%s(err_num=%d)\n", cudaGetErrorString(a), a);cudaDeviceReset();assert(0);}}
namespace cura {
	__device__ int computeScanSegmentIdxKernel(int x, int line_width) {

		return (x < 0) ? (x + 1) / line_width - 1 : x / line_width;
		// - 1 because -1 belongs to scansegment -1
		// + 1 because -line_width belongs to scansegment -1
	}

	__global__ void initModel(ClipperLib::IntPoint* model, int3 modelSize) {
		int point_idx = (blockIdx.x * blockDim.x) + threadIdx.x;//��point_idx��������
		if (point_idx < modelSize.x * modelSize.y * modelSize.z) {
			model[point_idx].X = 0;
			model[point_idx].Y = 0;
		}
	}
	__global__ void computePointGPU(Point* d_poly, ClipperLib::IntPoint* model, int lineSpacing, pointPtr* d_ptrForPoints, int* polyonsNum, int3 modelSize, int3 modelMin3D, int polygonsSize, int modelMax)
	{
		int point_idx = (blockIdx.x * blockDim.x) + threadIdx.x;//��point_idx��������
		if (point_idx < modelSize.z && point_idx + polygonsSize < modelMax) {
			int lineCnt = 2;
			for (int dx = polyonsNum[polygonsSize + point_idx]; dx < polyonsNum[polygonsSize + point_idx + 1]; ++dx) {
				pointPtr start_location = d_ptrForPoints[dx];
				pointPtr end_location = d_ptrForPoints[dx + 1];

				int layerNr = start_location.layerNr % modelSize.z;//�������������
				int originalPoint = modelSize.x * modelSize.y * layerNr;//��ǰ����ʼλ��

				ClipperLib::IntPoint p0 = d_poly[start_location.location];
				ClipperLib::IntPoint p1;

				int scanline_idx0;
				int scanline_idx1;

				ClipperLib::IntPoint cut_point;
				long long boundaryPixelLocation;
				for (int location = start_location.location + 1; location < end_location.location; ++location, ++lineCnt) {
					p1 = d_poly[location];

					scanline_idx0 = computeScanSegmentIdxKernel(p0.X, lineSpacing) + 1;
					scanline_idx1 = computeScanSegmentIdxKernel(p1.X, lineSpacing);
					if (p0.X >= p1.X)
					{
						int temp = scanline_idx0 - 1;
						scanline_idx0 = scanline_idx1 + 1;
						scanline_idx1 = temp;
					}
					if (scanline_idx0 <= scanline_idx1) {
						boundaryPixelLocation = (p0.Y + (p0.Y - p1.Y) * (scanline_idx0 * lineSpacing - p0.X) / (p0.X - p1.X)) / lineSpacing;
						boundaryPixelLocation += originalPoint + (scanline_idx0 - modelMin3D.x / lineSpacing) * modelSize.y - modelMin3D.y / lineSpacing;
						model[boundaryPixelLocation].X += 1;
						model[boundaryPixelLocation].Y = 1;
						++scanline_idx0;
					}
					for (; scanline_idx0 < scanline_idx1; ++scanline_idx0) {
						boundaryPixelLocation = (p0.Y + (p0.Y - p1.Y) * (scanline_idx0 * lineSpacing - p0.X) / (p0.X - p1.X)) / lineSpacing;
						boundaryPixelLocation += originalPoint + (scanline_idx0 - modelMin3D.x / lineSpacing) * modelSize.y - modelMin3D.y / lineSpacing;
						model[boundaryPixelLocation].X += 1;
						model[boundaryPixelLocation].Y = lineCnt;
					}
					if (scanline_idx0 == scanline_idx1) {
						boundaryPixelLocation = (p0.Y + (p0.Y - p1.Y) * (scanline_idx0 * lineSpacing - p0.X) / (p0.X - p1.X)) / lineSpacing;
						boundaryPixelLocation += originalPoint + (scanline_idx0 - modelMin3D.x / lineSpacing) * modelSize.y - modelMin3D.y / lineSpacing;
						model[boundaryPixelLocation].X += 1;
						model[boundaryPixelLocation].Y = 1;
					}

					p0 = p1;
				}
			}
		}
	}
	__global__ void computePointGPUY(Point* d_poly, ClipperLib::IntPoint* model, int lineSpacing, pointPtr* d_ptrForPoints, int* polygonsNum, int3 modelSize, int3 modelMin3D, int polygonsSize, int modelMax) {
		int point_idx = (blockIdx.x * blockDim.x) + threadIdx.x;//��point_idx��������
		if (point_idx < modelSize.z && point_idx + polygonsSize < modelMax) {
			int lineCnt = 2;
			for (int dx = polygonsNum[polygonsSize + point_idx]; dx < polygonsNum[polygonsSize + point_idx + 1]; ++dx) {
				pointPtr start_location = d_ptrForPoints[dx];
				pointPtr end_location = d_ptrForPoints[dx + 1];

				int layerNr = start_location.layerNr % modelSize.z;//�������������
				int originalPoint = modelSize.x * modelSize.y * layerNr;//��ǰ����ʼλ��

				ClipperLib::IntPoint p0 = d_poly[start_location.location];
				ClipperLib::IntPoint p1;

				int scanline_idx0;
				int scanline_idx1;

				ClipperLib::IntPoint cut_point;
				long long boundaryPixelLocation;
				for (int location = start_location.location + 1; location < end_location.location; ++location, ++lineCnt) {
					p1 = d_poly[location];

					scanline_idx0 = computeScanSegmentIdxKernel(p0.Y, lineSpacing) + 1;
					scanline_idx1 = computeScanSegmentIdxKernel(p1.Y, lineSpacing);
					if (p0.Y >= p1.Y) {
						int temp = scanline_idx0 - 1;
						scanline_idx0 = scanline_idx1 + 1;
						scanline_idx1 = temp;
					}
					if (scanline_idx0 <= scanline_idx1) {
						boundaryPixelLocation = (p0.X + (p0.X - p1.X) * (scanline_idx0 * lineSpacing - p0.Y) / (p0.Y - p1.Y)) / lineSpacing;
						boundaryPixelLocation = originalPoint + scanline_idx0 - modelMin3D.y / lineSpacing + (boundaryPixelLocation - modelMin3D.x / lineSpacing) * modelSize.y;
						model[boundaryPixelLocation].Y = 1;
						++scanline_idx0;
					}
					for (; scanline_idx0 <= scanline_idx1; ++scanline_idx0) {
						boundaryPixelLocation = (p0.X + (p0.X - p1.X) * (scanline_idx0 * lineSpacing - p0.Y) / (p0.Y - p1.Y)) / lineSpacing;
						boundaryPixelLocation = originalPoint + scanline_idx0 - modelMin3D.y / lineSpacing + (boundaryPixelLocation - modelMin3D.x / lineSpacing) * modelSize.y;
						model[boundaryPixelLocation].Y = lineCnt;
					}
					if (scanline_idx0 <= scanline_idx1) {
						boundaryPixelLocation = (p0.X + (p0.X - p1.X) * (scanline_idx0 * lineSpacing - p0.Y) / (p0.Y - p1.Y)) / lineSpacing;
						boundaryPixelLocation = originalPoint + scanline_idx0 - modelMin3D.y / lineSpacing + (boundaryPixelLocation - modelMin3D.x / lineSpacing) * modelSize.y;
						model[boundaryPixelLocation].Y = 1;
					}
					p0 = p1;
				}

			}
		}
	}
	__global__ void infillPolygonGPU(ClipperLib::IntPoint* model, int3 modelSize) {
		int point_idx = ((blockIdx.x * blockDim.x) + threadIdx.x) * modelSize.y;//��ʼλ��
		if (point_idx < modelSize.x * modelSize.y * modelSize.z) {
			int cnt = model[point_idx].X;
			int previer = model[point_idx].X;
			long long formatTrans;
			int curr;
			for (int a = point_idx + 1; a < point_idx + modelSize.y; ++a) {
				curr = model[a].X;
				if (model[a - 1].Y) {
					formatTrans = model[a - 1].Y & 0x00000000ffffffff;
					formatTrans = formatTrans << 32;
					model[a - 1].Y = formatTrans | 1;
					model[a - 1].X = 0;
				}
				if (previer < 56 && previer)
					cnt += previer;
				if ((cnt % 2) && !model[a].Y)
					model[a].X = 56;
				previer = curr;
			}
			if (curr < 56 && curr) {
				formatTrans = model[point_idx + modelSize.y - 1].Y & 0x00000000ffffffff;
				formatTrans = formatTrans << 32;
				model[point_idx + modelSize.y - 1].Y = formatTrans | 1;
				model[point_idx + modelSize.y - 1].X = 0;
			}
		}
	}

	__global__ void getBoundryStart(ClipperLib::IntPoint* model, int3 modelSize, int n, int *d_status) {
		int point_idx = ((blockIdx.x * blockDim.x) + threadIdx.x);//��ʼλ��
		int3 location;
		location.z = point_idx / (modelSize.x * modelSize.y);
		location.x = point_idx % (modelSize.x * modelSize.y) / modelSize.y;
		location.y = point_idx % (modelSize.x * modelSize.y) % modelSize.y;
		int2 size;
		size.x = location.z * (modelSize.x * modelSize.y);
		size.y = (location.z + 1) * (modelSize.x * modelSize.y);
		int cnt = 0;
		if (point_idx < modelSize.x * modelSize.y * modelSize.z && model[point_idx].Y) {
			//*d_status = 1;
			if (point_idx - modelSize.y >= size.x && model[point_idx - modelSize.y].X)
				cnt += 1;
			else if (point_idx + modelSize.y < size.y && model[point_idx + modelSize.y].X)
				cnt += 1;
			else if (location.y > 0 && model[point_idx - 1].X)
				cnt += 1;
			else if (location.y + 1 < modelSize.y && model[point_idx + 1].X)
				cnt += 1;
			if (!cnt)
				model[point_idx].Y = 0;
		}
	}

	__global__ void getBoundry(ClipperLib::IntPoint* model, int3 modelSize, int n, int *d_status) {
		int point_idx = ((blockIdx.x * blockDim.x) + threadIdx.x);//��ʼλ��
		int3 location;
		location.z = point_idx / (modelSize.x * modelSize.y);
		location.x = point_idx % (modelSize.x * modelSize.y) / modelSize.y;
		location.y = point_idx % (modelSize.x * modelSize.y) % modelSize.y;
		int2 size;
		size.x = location.z * (modelSize.x * modelSize.y);
		size.y = (location.z + 1) * (modelSize.x * modelSize.y);
		int cnt = 0;
		ClipperLib::IntPoint currPoint = model[point_idx];
		currPoint.X = currPoint.Y & 0x00000000ffffffff;
		++n;
		if (point_idx < modelSize.x * modelSize.y * modelSize.z && !currPoint.X && currPoint.Y) {
			//*d_status = 1;
			if (point_idx - modelSize.y >= size.x && model[point_idx - modelSize.y].X)
				cnt += 1;
			else if (point_idx + modelSize.y < size.y && model[point_idx + modelSize.y].X)
				cnt += 1;
			else if (location.y > 0 && model[point_idx - 1].X)
				cnt += 1;
			else if (location.y + 1 < modelSize.y && model[point_idx + 1].X)
				cnt += 1;

			if (cnt) {
				model[point_idx].Y = model[point_idx].Y | n;
				*d_status = 1;
			}
			else
				model[point_idx].Y = 0;
		}
	}

	__device__ int dist(float a, float b, float c) {
		return sqrtf((a - c) * (a - c) + (b - c) * (b - c)) + 0.5;
	}

	__global__ void initOffsetCricle(int* offsetCircle, int offset) {
		int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
		float a = point_idx / (2 * offset + 1);
		float b = point_idx % (2 * offset + 1);
		if (dist(a, b, offset) <= offset && point_idx < (2 * offset + 1) * (2 * offset + 1))
			offsetCircle[point_idx] = 1;
		else if (point_idx < (2 * offset + 1) * (2 * offset + 1))
			offsetCircle[point_idx] = 0;
	}

	__global__ void getBoundryCorrode(ClipperLib::IntPoint* model, int3 modelSize, int* offsetCricle, int offset, int n) {
		int point_idx = ((blockIdx.x * blockDim.x) + threadIdx.x);//��ʼλ��
		int3 location;
		//��ǰ�������
		location.z = point_idx / (modelSize.x * modelSize.y);
		location.x = point_idx % (modelSize.x * modelSize.y) / modelSize.y;
		location.y = point_idx % (modelSize.x * modelSize.y) % modelSize.y;

		//��ά�߽�
		int4 boundary;
		boundary.w = (location.x - offset < 0) ? -location.x : -offset;
		boundary.x = (location.x + offset >= modelSize.x) ? modelSize.x - 1 - location.x : offset;
		boundary.y = (location.y - offset < 0) ? -location.y : -offset;
		boundary.z = (location.y + offset >= modelSize.y) ? modelSize.y - 1 - location.y : offset;
		int status = 0;
		if ((model[point_idx].Y & 0x00000000ffffffff) == n) {
			for (int i = boundary.w; i <= boundary.x; ++i) {
				for (int j = boundary.y; j <= boundary.z; ++j) {
					if (model[point_idx + i * modelSize.y + j].Y == 0 && offsetCricle[(2 * offset + 1) * (offset + i) + offset + j]) {
						model[point_idx + i * modelSize.y + j].X = 0;
						model[point_idx + i * modelSize.y + j].Y = model[point_idx].Y & 0xffffffff00000000;
					}
				}
			}
		}
	}
	__constant__ static const  int2 searchRoute[8] = { 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1 };//�ϣ����ϣ�...�����ϣ���ʱ�����ͨ�����飻
	__device__ bool generatePolygon(ClipperLib::IntPoint *model, int3 modelSize, int storageNum, int i, int polygonIdx) {
		int startNum = storageNum;
		ClipperLib::IntPoint startPoint;//�궨��ʼԪ�ص�x��yֵ
		ClipperLib::IntPoint currPoint;
		int startDirect = 0;
		bool findStartPoint = false;
		long long temp;
		int cnt;
		ClipperLib::IntPoint tempP;//�궨��ǰԪ�ص�x��yֵ����currPoint�ظ���
		tempP.X = startPoint.X = i % (modelSize.x * modelSize.y) / modelSize.y;
		tempP.Y = startPoint.Y = i % (modelSize.x * modelSize.y) % modelSize.y;
		ClipperLib::IntPoint previerPoint;//ǰһ������߶���ź�������
		previerPoint.X = model[tempP.X * modelSize.y + tempP.Y].Y & 0x00000000ffffffff;
		previerPoint.Y = model[tempP.X * modelSize.y + tempP.Y].Y & 0xffffffff00000000;

		currPoint.X = startPoint.X;
		currPoint.Y = startPoint.Y;
		int cntLine = 1;
		//û�ҵ���ʼ�㣬˵��������βδ���ӣ�����ѭ��
		while (!findStartPoint) {
			bool findPoint = false;
			cnt = 0;
			while (!findPoint) {//Ѱ�ҵ�ǰ�����һ�����ӵ�
				++cnt;
				int ay = currPoint.Y + searchRoute[startDirect].y;//��һ�����ӵ��y����
				int ax = currPoint.X + searchRoute[startDirect].x;//��һ�����ӵ��x����
				ClipperLib::IntPoint a;
				a.X = model[ax * modelSize.y + ay].Y & 0x00000000ffffffff;//��һ�����ӵ���߶����
				a.Y = model[ax * modelSize.y + ay].Y & 0xffffffff00000000;//��һ�����ӵ����ڵ�a.Y������
				if (a.X == previerPoint.X) {//��ͬһ�㣬��...��
					findPoint = true;
					tempP.X = currPoint.X = ax;//������ǰ��
					tempP.Y = currPoint.Y = ay;

					temp = tempP.X << 32;//����ǰ��x&y����϶�Ϊһ
					temp |= tempP.Y;
					if (a.Y == 0x0000000100000000) {//��a�ǹսǴ��ĵ�
						model[++storageNum].X = temp;
						cntLine = 1;
					}
					else if (a.Y == previerPoint.Y && cntLine > 1) {
						model[storageNum].X = temp;
					}
					else if (a.Y == previerPoint.Y) {//��a�ǵ�ǰ�߶εĵڶ�����
						++cntLine;
						model[++storageNum].X = temp;
					}
					else {//��a�ǵ�ǰ�߶ε���ʼ��
						model[++storageNum].X = temp;
						cntLine = 1;
					}
					previerPoint = a;
					model[ax * modelSize.y + ay].Y = 0;//��ʹ�ù��ĵ���0

					if (currPoint.X == startPoint.X && currPoint.Y == startPoint.Y) {
						findStartPoint = true;//���ҵ���ʼ��
					}
					startDirect = (startDirect - 1) & 7;//����������ʱ����1
				}
				else {//û�ҵ���ʱ�����Ѱ��
					startDirect++;
					if (startDirect == 8) {
						startDirect = 0;
					}
				}
				if (cnt == 9) {//����ͨ�������޿����ӵ㣬��������ʧ��
					break;
				}
			}
			if (cnt == 9) {
				break;
			}

		}
		if (findStartPoint) {//���ӳɹ����¼��ǰ����
			++storageNum;
			model[startNum].X = storageNum - startNum;
			model[startNum].Y = 0xffffffffffffffff;
			return true;
		}
		else {//���򲻼�¼
			storageNum = startNum;
			model[startNum].X = 0;
			return false;
		}
	}
	__global__ void resultCollect(ClipperLib::IntPoint *model, int3 modelSize) {
		int point_idx = blockDim.x * blockIdx.x + threadIdx.x;
		long long currLoca = 1;
		long long polygonsNum;
		int polygonNum;
		bool judgeP;
		if (point_idx < modelSize.z) {
			for (int i = point_idx * modelSize.x * modelSize.y; i < (point_idx + 1) * modelSize.x * modelSize.y; ++i) {
				if (model[i].Y) {
					judgeP = generatePolygon(&model[point_idx * modelSize.x * modelSize.y], modelSize, currLoca, i, point_idx);
					model[i].Y = 0;
					if (judgeP) {
						polygonNum = model[point_idx * modelSize.x * modelSize.y + currLoca].X;
						currLoca += polygonNum;
						++polygonsNum;
					}
				}
			}
			model[point_idx * modelSize.x * modelSize.y].X = currLoca;
			model[point_idx * modelSize.x * modelSize.y].Y = 0xffffffffffffffff;
		}
	}

	__global__ void transformPoints(ClipperLib::IntPoint *model, int3 modelSize, int3 modelMin3D, int lineSpacing) {
		int point_idx = blockDim.x * blockIdx.x + threadIdx.x;
		ClipperLib::IntPoint p;
		p = model[point_idx];
		if (p.Y != 0xffffffffffffffff) {
			p.Y = p.X & 0x00000000ffffffff;
			p.X = p.X >> 32;
			p.Y = p.Y * lineSpacing + modelMin3D.y;
			p.X = p.X * lineSpacing + modelMin3D.x;
		}
		model[point_idx] = p;
	}

	void cudaConcentricInfillByPixel(vector<int>& polygonsNum, vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints,
		vector< vector<Point> >& scanlineStorage, vector< vector<int> >& scanlineStorageInfo, int scanlineSpacing, int3 modelMin3D, int3 modelMax3D) {
		cudaError_t cudaStatus;

		//Copy polygonsNum to GPU
		int d_polygonsCnt = (polygonsNum.size() + threadNUM - 1) / threadNUM * threadNUM;
		int *d_polygonsNum;
		CUDA_CALL(cudaMalloc((void**)&d_polygonsNum, d_polygonsCnt * sizeof(int)));
		CUDA_CALL(cudaMemcpy(d_polygonsNum, &polygonsNum[0], polygonsNum.size() * sizeof(int), cudaMemcpyHostToDevice));

		//Copy polygonInfo to GPU
		int d_polysCnt = (polygonInfo.size() + threadNUM - 1) / threadNUM * threadNUM;
		pointPtr *d_ptrForPoints;
		CUDA_CALL(cudaMalloc((void**)&d_ptrForPoints, d_polysCnt * sizeof(pointPtr)));
		CUDA_CALL(cudaMemcpy(d_ptrForPoints, &polygonInfo[0], polygonInfo.size() * sizeof(pointPtr), cudaMemcpyHostToDevice));

		//Copy polygonPoints to GPU
		int d_poly_size = (polygonPoints.size() + threadNUM - 1) / threadNUM * threadNUM;
		ClipperLib::IntPoint* d_poly;
		CUDA_CALL(cudaMalloc((void**)&d_poly, sizeof(ClipperLib::IntPoint) * d_poly_size));
		CUDA_CALL(cudaMemcpy(d_poly, &polygonPoints[0], sizeof(ClipperLib::IntPoint) * polygonPoints.size(), cudaMemcpyHostToDevice));

		//Caculate memory used to malloc pixel structure
		int lineSpacing = scanlineSpacing / 3;
		//	lineSpacing = lineSpacing < pixelSpacing ? lineSpacing : pixelSpacing;
		int3 modelSize = { (modelMax3D.x - modelMin3D.x) / lineSpacing + 1, (modelMax3D.y - modelMin3D.y) / lineSpacing + 1, modelMax3D.z };

		size_t avail, total;
		cudaMemGetInfo(&avail, &total);
		int n = (avail / (8 * sizeof(ClipperLib::IntPoint))) / (modelSize.x * modelSize.y) - 1;
		if (n <= 3) {
			fprintf(stderr, "cudaMemory is limited!\n");
			return;
		}
		n = n > modelSize.z ? modelSize.z : n;
		int3 GPUModelSize = { modelSize.x, modelSize.y, n };

		//Malloc pixel array
		ClipperLib::IntPoint *model;
		CUDA_CALL(cudaMalloc((void**)&model, modelSize.x * modelSize.y * n * sizeof(ClipperLib::IntPoint)));
		ClipperLib::IntPoint *h_model;
		h_model = (ClipperLib::IntPoint*)malloc(modelSize.x * modelSize.y * GPUModelSize.z * sizeof(ClipperLib::IntPoint));

		//Generate offset circle
		int offset = scanlineSpacing / lineSpacing;
		int *offsetCricle;
		int offsetCricleSize = 2 * offset + 1;
		CUDA_CALL(cudaMalloc((void**)&offsetCricle, pow(offsetCricleSize, 2) * sizeof(int)));
		initOffsetCricle << <(offsetCricleSize + threadNUM - 1) / threadNUM, threadNUM >> > (offsetCricle, offset);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "initOffsetCricle launch failed: %s\n", cudaGetErrorString(cudaStatus));

		int currI = 0;
		int *d_status;
		CUDA_CALL(cudaMalloc((void**)&d_status, sizeof(int)));
		while (currI <= modelSize.z) {
			//ģ�ͳ�ʼ��
			initModel << <(modelSize.x * modelSize.y * GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (model, GPUModelSize);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "initModel launch failed: %s\n", cudaGetErrorString(cudaStatus));

			//������ʽת��
			computePointGPU << <(GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (d_poly, model, lineSpacing, d_ptrForPoints, d_polygonsNum, GPUModelSize, modelMin3D, currI, modelSize.z);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "computePointGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
			computePointGPUY << <(GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (d_poly, model, lineSpacing, d_ptrForPoints, d_polygonsNum, GPUModelSize, modelMin3D, currI, modelSize.z);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "computePointGPUY launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//�������ڲ����
			infillPolygonGPU << <(modelSize.x * GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (model, GPUModelSize);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "infillPolygonGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));

			int status = 1, cnt = 1;
			//�޳��߽��ϵ������
			getBoundryStart << <(modelSize.x * modelSize.y * GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (model, GPUModelSize, cnt, d_status);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "getBoundryStart launch failed: %s\n", cudaGetErrorString(cudaStatus));
			CUDA_CALL(cudaDeviceSynchronize());
			//��ѭ��ʵ���������ڲ���䣬ֱ�������ڲ��޴�������򣬼�status == 0��
			while (status) {
				status = 0;
				CUDA_CALL(cudaMemcpy(d_status, &status, sizeof(int), cudaMemcpyHostToDevice));
				//���߽�������������ÿ��������ΪԲ�ģ�ƫ�þ���Ϊ�뾶��Բ�ڵ����зǱ����ص�ֵ��Ϊ0��
				getBoundryCorrode << <(modelSize.x * modelSize.y * GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (model, GPUModelSize, offsetCricle, offset, cnt);
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "getBoundryCorrode launch failed: %s\n", cudaGetErrorString(cudaStatus));
				//�±߽��ʶ��
				getBoundry << <(modelSize.x * modelSize.y * GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (model, GPUModelSize, cnt, d_status);
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "getBoundry launch failed: %s\n", cudaGetErrorString(cudaStatus));
				CUDA_CALL(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
				CUDA_CALL(cudaDeviceSynchronize());
				++cnt;
			}
			int *dSearchRoute;
			//���߽��������������ȡ������
			resultCollect << <(GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (model, GPUModelSize);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "resultCollect launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//��ÿ�����ؽ��и�ʽת����
			transformPoints << < (modelSize.x * modelSize.y * GPUModelSize.z + threadNUM - 1) / threadNUM, threadNUM >> > (model, GPUModelSize, modelMin3D, lineSpacing);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "transformPoints launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//���ݿ�����host�ˣ�����vector��
			CUDA_CALL(cudaMemcpy(h_model, model, modelSize.x * modelSize.y * GPUModelSize.z * sizeof(ClipperLib::IntPoint), cudaMemcpyDeviceToHost));
			for (int i = 0; i < GPUModelSize.z && currI + i < modelSize.z; ++i) {
				ClipperLib::IntPoint *be = h_model + modelSize.x * modelSize.y * i;
				long long polygonsSize = be[0].X;
				scanlineStorage.push_back(vector<Point>(polygonsSize));
				scanlineStorage.back().assign(be, be + polygonsSize);
			}
			currI += n;
		}
		CUDA_CALL(cudaFree(d_status));
		CUDA_CALL(cudaFree(model));
		CUDA_CALL(cudaFree(offsetCricle));
		CUDA_CALL(cudaFree(d_poly));
		CUDA_CALL(cudaFree(d_ptrForPoints));
		free(h_model);
	}
}