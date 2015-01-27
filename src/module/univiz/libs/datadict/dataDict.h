/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Data Dictionary (e.g. for loading transient data)
//
// CGL ETH Zuerich
// Ronald Peikert and Filip Sadlo

#ifndef DATA_DICT_H
#define DATA_DICT_H

#include <vector>
#include <cstring>

struct TimeStep
{
    float time;
    char fileName[20]; // Decimal representation of time
    float *data; // NULL or pointer to cache

    TimeStep(float t = 0, char *f = "0.0")
    {
        time = t;
        strcpy(fileName, f);
        data = 0;
    }
};

class DataDict
{
public:
    DataDict();
    ~DataDict();

    void setDataDict(char *dirName, int cacheSize = 2, bool verbose = false, bool useMMap = false);

    int byteSize()
    {
        return nBytes;
    }
    bool setByteSize(int n);

    float minTime()
    {
        return min;
    }
    float maxTime()
    {
        return max;
    }

    float *interpolate(float time);
    float interpolate(float time, int floatIdx);
    float *temporalDerivative(float time);

    void setTime(float t);
    float getTime()
    {
        return time;
    }

    char *getDirectory()
    {
        return directory;
    }

    float *data0()
    {
        return ts0->data;
    }
    float *data1()
    {
        return ts1->data ? ts1->data : ts0->data;
    }
    float weight0()
    {
        return wt0;
    }
    float weight1()
    {
        return wt1;
    }
    float dweight0()
    {
        return dwt0;
    }
    float dweight1()
    {
        return dwt1;
    }

private:
    TimeStep *getTimeStep(int timeStepNr);

    char directory[300];
    std::vector<TimeStep> timeStep;
    int nTimeSteps;
    int nFloats;
    int nBytes; // bytes used
    int nBytesFile; // bytes present in file

    int cacheSize; // Max number of simult. cache data sets
    int cachePos; // Position to overwrite next
    float **cache; // Cyclic buffer
    int *cached;
    int *fileDescs;

    float min, max;
    float time;
    TimeStep *ts0, *ts1;
    float wt0, wt1; // Weights for linear interpolation
    float dwt0, dwt1; // Weights for temporal derivative
    float epsilon;
    bool verbose;
    bool useMMap;
};
#endif // DATA_DICT_H
