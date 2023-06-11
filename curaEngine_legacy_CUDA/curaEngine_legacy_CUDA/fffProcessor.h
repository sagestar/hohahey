#ifndef FFF_PROCESSOR_H
#define FFF_PROCESSOR_H

#include <algorithm>
#include <vector>
#include "utils/socket.h"
#include "utils/gettime.h"
#include "gcodeExport.h"
#include "settings.h"
#include "sliceDataStorage.h"

namespace cura {

	//FusedFilamentFabrication processor.
	class fffProcessor
	{
	private:
		int maxObjectHeight;
		int fileNr;
		GCodeExport gcode;
		ConfigSettings& config;
		TimeKeeper timeKeeper;
		ClientSocket guiSocket;

		GCodePathConfig skirtConfig;
		GCodePathConfig inset0Config;
		GCodePathConfig insetXConfig;
		GCodePathConfig infillConfig;
		GCodePathConfig skinConfig;
		GCodePathConfig supportConfig;
	public:
		fffProcessor(ConfigSettings& config)
			: config(config)
		{
			fileNr = 1;
			maxObjectHeight = 0;
		}

		void guiConnect(int portNr)
		{
			guiSocket.connectTo("127.0.0.1", portNr);
		}

		void sendPolygonsToGui(const char* name, int layerNr, int32_t z, Polygons& polygons);

		bool setTargetFile(const char* filename);

		bool processFile(const std::vector<std::string> &files);

		void finalize()
		{
			if (!gcode.isOpened())
				return;
			gcode.finalize(maxObjectHeight, config.moveSpeed, config.endCode.c_str());
		}

		friend bool cudaInfill(fffProcessor& processor, SliceDataStorage& storage, vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum);

		friend 	void preProcess(fffProcessor& processor, SliceDataStorage& storage, vector<int>& polygonsNum, vector<pointPtr>& polygonInfo, ClipperLib::Path& polygonPoints);
	private:
		void preSetup();

		bool prepareModel(SliceDataStorage& storage, const std::vector<std::string> &files);

		void processSliceData(SliceDataStorage& storage);

		void writeGCode(SliceDataStorage& storage);

		//Add a single layer from a single mesh-volume to the GCode
		void addVolumeLayerToGCode(SliceDataStorage& storage, GCodePlanner& gcodeLayer, int volumeIdx, int layerNr, 
			vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum, long long& polysCnt);

		void addInfillToGCode(SliceLayerPart* part, GCodePlanner& gcodeLayer, int layerNr, int extrusionWidth, int fillAngle, 
			vector< vector<Point> >& scanlineStorage, vector< vector<int> >& ptrForPoints, vector<int>& polygonsNum, long long polysCnt, vector<Point> modelSizeM);

		void addInsetToGCode(SliceLayerPart* part, GCodePlanner& gcodeLayer, int layerNr);

		void addSupportToGCode(SliceDataStorage& storage, GCodePlanner& gcodeLayer, int layerNr);

		void addWipeTower(SliceDataStorage& storage, GCodePlanner& gcodeLayer, int layerNr, int prevExtruder);

		friend class PolygonRef;

	};

}//namespace cura

#endif//FFF_PROCESSOR_H
