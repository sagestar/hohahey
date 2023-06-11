/** Copyright (C) 2013 David Braam - Released under terms of the AGPLv3 License */
#ifndef GETTIME_H
#define GETTIME_H

#if defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
#include <sys/time.h>
#include <stddef.h>
#else
#include <windows.h>
#endif

static inline double getTime()
{
#if defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return double(tv.tv_sec) + double(tv.tv_usec) / 1000000.0;
#else
	return double(GetTickCount()) / 1000.0;
#endif
}

class TimeKeeper
{
private:
    double startTime;
public:
    TimeKeeper();
    
    double restart();
};

#endif//GETTIME_H
