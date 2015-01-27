/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Data Dictionary (e.g. for loading transient data)
//
// CGL ETH Zuerich
// Ronald Peikert and Filip Sadlo

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <algorithm>
#include "dataDict.h"

// TODO: support mmap() for win32 (non-POSIX ?)
#ifdef WIN32
#include <windows.h>
#else
#include <dirent.h>
//#   if USE_MMAP
#include <unistd.h>
#include <sys/mman.h>
//#   endif
#endif

using namespace std;

bool operator<(const TimeStep &a, const TimeStep &b) { return a.time < b.time; }

// Class methods
DataDict::DataDict()
{
    // Initialize
    nTimeSteps = 0;
    nBytes = 0;
    min = 1e19f;
    max = -1e19f;
    cacheSize = 0;
    cache = 0;
    cached = 0;
    fileDescs = 0;
    useMMap = false;
}

DataDict::~DataDict()
{
    if (useMMap)
    {
        for (int i = 0; i < cacheSize; i++)
        {
            if (cached[i] >= 0)
            {
                munmap(cache[i], cacheSize);
                //close(fileDescs[i]);
            }
        }
        delete[] fileDescs;
    }
    else
    {
        for (int i = 0; i < cacheSize; i++)
            delete[] cache[i];
    }
    delete[] cache;
    delete[] cached;
}

void DataDict::setDataDict(char *dirName, int size, bool verb, bool mmap)
{
    verbose = verb;
    useMMap = mmap;
    strcpy(directory, dirName);

    timeStep.resize(0);

#ifdef WIN32
    HANDLE handle;
    char dirSpec[400];
    sprintf(dirSpec, "%s\\*", dirName);
    WIN32_FIND_DATA fileData;
    bool first = true;
#else
    struct dirent *dent;
    DIR *dir = opendir(dirName);
    if (!dir)
    {
        fprintf(stderr, "Cannot open directory %s\n", dirName);
        return;
    }
#endif

    while (true)
    {
        char *fileName;

// Get a new file name
#ifdef WIN32
        if (first)
        {
            first = false;
            handle = FindFirstFile(dirSpec, &fileData);
            if (handle == INVALID_HANDLE_VALUE)
            {
                printf("Invalid file handle. Error is %u\n", GetLastError());
            }
        }
        else
        {
            if (FindNextFile(handle, &fileData) == 0)
            {
                FindClose(handle);
                break;
            }
        }
        fileName = fileData.cFileName;
#else
        dent = readdir(dir);
        if (!dent)
        {
            closedir(dir);
            break;
        }
        fileName = dent->d_name;
#endif

        // Convert the file name into a time
        float time;
        if (sscanf(fileName, "%f", &time) != 1)
            continue;
        if (time < min)
            min = time;
        if (time > max)
            max = time;

        // Add to time steps
        timeStep.push_back(TimeStep(time, fileName));
        nTimeSteps++;

// Get byte size
#ifdef WIN32
        nBytesFile = fileData.nFileSizeLow + fileData.nFileSizeHigh * MAXDWORD;
#else
        struct stat status;
        char fullName[400];
        sprintf(fullName, "%s/%s", directory, fileName);
        int ret = stat(fullName, &status);
        nBytesFile = ret >= 0 ? status.st_size : 0;
#endif

        if (nBytes == 0)
        {
            nBytes = nBytesFile;
            nFloats = nBytes / sizeof(float);
        }
    }

    // Sort time steps
    sort(timeStep.begin(), timeStep.end());
    epsilon = (maxTime() - minTime()) * 1e-6f;

    if (verbose)
    {
        fprintf(stderr, "%d timesteps, range: %g .. %g\n", timeStep.size(), minTime(), maxTime());
    }

    // Set up chache
    cacheSize = size;
    cache = new float *[cacheSize];
    cached = new int[cacheSize];
    if (useMMap)
    {
        fileDescs = new int[cacheSize];
    }
    for (int i = 0; i < cacheSize; i++)
    { // allocate
        if (useMMap)
        {
            cache[i] = NULL;
            fileDescs[i] = -1;
        }
        else
        {
            cache[i] = new float[nFloats];
        }

        cached[i] = -1;
    }
    cachePos = 0;
}

bool DataDict::setByteSize(int n)
{
    if (n > nBytesFile)
    {
        fprintf(stderr, "DataDict::setByteSize: error: dictionary files are too small\n");
        nBytes = nBytesFile;
        nFloats = nBytes / sizeof(float);
        return false;
    }
    nBytes = n;
    nFloats = nBytes / sizeof(float);
    return true;
}

TimeStep *DataDict::getTimeStep(int timeStepNr)
{
    TimeStep *ts = &timeStep[timeStepNr];

    if (!ts->data)
    { // need to cache it
        int oldTimeStepNr = cached[cachePos];
        cached[cachePos] = timeStepNr;
        if (oldTimeStepNr >= 0)
            timeStep[oldTimeStepNr].data = 0;

        if (useMMap)
        {
            // unmap and close old file
            if (verbose)
                fprintf(stderr, "unmapping and closing\n");
            if (oldTimeStepNr >= 0)
            {
                munmap(cache[cachePos], nBytes);
                //close(fileDescs[cachePos]);
            }

            // open new file
            char fileName[400];
            sprintf(fileName, "%s/%s", directory, ts->fileName);
            if (verbose)
                fprintf(stderr, "opening\n");
            fileDescs[cachePos] = open(fileName, O_RDONLY);
            if (fileDescs[cachePos] == -1)
            {
                fprintf(stderr, "Cannot open %s\n", fileName);
                return 0;
            }

            // map new file
            if (verbose)
                fprintf(stderr, "Cache mmap: %s (%d bytes)\n", ts->fileName, nBytes);
            cache[cachePos] = (float *)mmap(0, nBytes, PROT_READ, MAP_SHARED, fileDescs[cachePos], 0);
            //cache[cachePos] = (float *) mmap(0, nBytes, PROT_READ, MAP_SHARED|MAP_NORESERVE, fileDescs[cachePos], 0);
            if (cache[cachePos] == MAP_FAILED)
            {
                fprintf(stderr, "Cache mmap failed\n");
                exit(1); // ####
            }

            // tell the virtual memory system about expected accesses
            //madvise(cache[cachePos], nBytes, MADV_RANDOM);
            //madvise(cache[cachePos], nBytes, MADV_SEQUENTIAL); // probably use this
            //madvise(cache[cachePos], nBytes, MADV_WILLNEED);
            //madvise(cache[cachePos], nBytes, MADV_DONTNEED);

            // mmap does not need the file to be open
            close(fileDescs[cachePos]);

            ts->data = cache[cachePos];
            cachePos = (cachePos + 1) % cacheSize;
        }
        else
        { // no mmap()
            ts->data = cache[cachePos];
            cachePos = (cachePos + 1) % cacheSize;

            if (verbose)
                fprintf(stderr, "Cache load: %s (%d bytes)\n", ts->fileName, nBytes);

            // Read the data
            char fileName[400];
            sprintf(fileName, "%s/%s", directory, ts->fileName);
            FILE *fp = fopen(fileName, "rb");
            if (!fp)
            {
                fprintf(stderr, "Cannot read %s\n", fileName);
                return 0;
            }
            fread(ts->data, 1, nBytes, fp);
            fclose(fp);
        }
    }

    return &timeStep[timeStepNr];
}

void DataDict::setTime(float t)
{
    if (t < minTime() + epsilon)
    { // first time step
        ts0 = getTimeStep(0);
        ts1 = getTimeStep(1);
        wt0 = 1;
        wt1 = 0;
        time = t;
    }
    else if (t > maxTime() - epsilon)
    { // last time step
        ts1 = getTimeStep(nTimeSteps - 1);
        ts0 = getTimeStep(nTimeSteps - 2);
        wt0 = 0;
        wt1 = 1;
        time = t;
    }
    else
    { // binary search for time
        int lo = 0;
        int hi = nTimeSteps - 1;
        bool found = false;
        while (hi > lo + 1)
        {
            int mid = (lo + hi) / 2;
            if (t < timeStep[mid].time - epsilon)
                hi = mid;
            else if (t > timeStep[mid].time + epsilon)
                lo = mid;
            else
            { // intermediate time step
                ts0 = getTimeStep(mid);
                ts1 = getTimeStep(mid + 1);
                wt0 = 1;
                wt1 = 0;
                time = t;
                found = true;
                break;
            }
        }
        if (!found)
        {
// between two time steps
#ifdef OLD
            if (t > time)
            {
                ts0 = getTimeStep(lo);
                ts1 = getTimeStep(hi);
            }
            else
            {
                ts1 = getTimeStep(hi);
                ts0 = getTimeStep(lo);
            }
#endif
            ts0 = getTimeStep(lo);
            ts1 = getTimeStep(hi);
            ts0 = getTimeStep(lo); // Load (lo) again, if kicked out by previous loading of (hi)

            wt0 = (ts1->time - t) / (ts1->time - ts0->time);
            wt1 = (t - ts0->time) / (ts1->time - ts0->time);
            time = t;
        }
    }

    // Set weights for temporal derivative
    dwt1 = 1. / (ts1->time - ts0->time);
    dwt0 = -dwt1;
}

float *DataDict::interpolate(float time)
{
    setTime(time);
    float *p = new float[nFloats];

    float *d0 = data0();
    float *d1 = data1();
    float wt0 = weight0();
    float wt1 = weight1();

    for (int i = 0; i < nFloats; i++)
    {
        p[i] = wt0 * *d0++ + wt1 * *d1++;
    }

    return p;
}

float DataDict::interpolate(float time, int floatIdx)
{
    setTime(time);

    float *d0 = data0();
    float *d1 = data1();
    float wt0 = weight0();
    float wt1 = weight1();

    return wt0 * d0[floatIdx] + wt1 * d1[floatIdx];
}

float *DataDict::temporalDerivative(float time)
{
    setTime(time);
    float *p = new float[nFloats];

    float *d0 = data0();
    float *d1 = data1();
    float dwt0 = dweight0();
    float dwt1 = dweight1();

    for (int i = 0; i < nFloats; i++)
    {
        p[i] = dwt0 * *d0++ + dwt1 * *d1++;
    }

    return p;
}
