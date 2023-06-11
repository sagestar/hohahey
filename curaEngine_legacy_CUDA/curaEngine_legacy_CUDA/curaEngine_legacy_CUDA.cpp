/** Copyright (C) 2013 David Braam - Released under terms of the AGPLv3 License */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <windows.h>

#include <signal.h>

#include <stddef.h>
#include <vector>

#include "utils/gettime.h"
#include "utils/logoutput.h"
#include "sliceDataStorage.h"

#include "modelFile/modelFile.h"
#include "settings.h"
#include "optimizedModel.h"
#include "multiVolumes.h"
#include "polygonOptimizer.h"
#include "slicer.h"
#include "layerPart.h"
#include "inset.h"
#include "skin.h"
#include "infill.h"
#include "bridge.h"
#include "support.h"
#include "pathOrderOptimizer.h"
#include "skirt.h"
#include "raft.h"
#include "comb.h"
#include "gcodeExport.h"
#include "fffProcessor.h"
#include <ctime>


void print_usage()
{
	cura::logError("usage: CuraEngine [-h] [-v] [-m 3x3matrix] [-c <config file>] [-s <settingkey>=<value>] -o <output.gcode> <model.stl>\n");
}

//Signal handler for a "floating point exception", which can also be integer division by zero errors.
void signal_FPE(int n)
{
	(void)n;
	cura::logError("Arithmetic exception.\n");
	exit(1);
}

using namespace cura;

int main(int argc, char **argv)
{
#if defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
	//Lower the process priority on linux and mac. On windows this is done on process creation from the GUI.
	setpriority(PRIO_PROCESS, 0, 10);
#endif
	//Register the exception handling for arithmic exceptions, this prevents the "something went wrong" dialog on windows to pop up on a division by zero.
	ConfigSettings config;
	if (!config.setSetting("layerThickness", "100"))
		cura::logError("Setting not found\n");
	if (!config.setSetting("infillPattern", "3"))
		cura::logError("Setting not found\n");
	if (!config.setSetting("sparseInfillLineDistance", "12000"))
		cura::logError("Setting not found\n");

	cura::fffProcessor processor(config);
	std::string fileName = "../test/testModel.stl";
	std::vector<std::string> files;
	files.push_back(fileName);
	processor.setTargetFile("../test/bwl.gcode");
	clock_t start, finish;
	double duration;
	start = clock();

	processor.processFile(files);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("%f seconds\n", duration);
	return 0;
}
