/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadMP.h"

#include <iostream>
#include <sstream>

#include <util/coVector.h>

using namespace std;

ReadMP::ReadMP()
    : coModule("Measurement Report reader")
    , points(0)
    , lines(0)
{

    set_module_description("Reader for a Measurement Reports");

    mpPointsPort = addOutputPort("points", "coDoPoints|coDoSet", "tcp points");
    mpLinesPort = addOutputPort("lines", "coDoLines|coDoSet", "tcp path");

    mpFileParam = addFileBrowserParam("mpFile", "MP file");
    mpFileParam->setValue("/mnt/raid/cc/users/ak_te/Work/ISW/ProSi/K_GROSS-2", "*");
}

ReadMP::~ReadMP()
{
}

int ReadMP::compute()
{

    const char *filename = mpFileParam->getValue();

    list<coDoPoints *> pointsList;
    list<coDoLines *> linesList;

    if (filename == 0)
    {

        sendError("Please specify a filename");

        return FAIL;
    }

    ifstream file(filename, ios::in);

    if (!file)
    {

        ostringstream info;

        info << "Cannot open file " << filename;
        sendError(info.str().c_str());

        return FAIL;
    }
    else
    {

        ostringstream info;

        info << "Opened file " << filename;
        sendInfo(info.str().c_str());
    }

    const int INBUFSIZE = 1024;
    char *buffer = new char[INBUFSIZE];

    int ctr = 0;

    char **endptr = 0;

    list<float> locationValues;

    bool inSet = false;
    bool inData = false;

    int dataCount = 0;

    for (file.getline(buffer, INBUFSIZE, '\n'); file.good(); file.getline(buffer, INBUFSIZE, '\n'))
    {

        string line(buffer);

        if (line.find("BEGINSET") != string::npos)
        {
            inData = false;
            inSet = true;
        }
        else if ((line.find("MDI") != string::npos) && inSet)
        {
            int cStart = line.rfind("/");
            int cEnd = line.rfind(",") - 1;
            dataCount = atoi(line.substr(cStart, cEnd - cStart).c_str());

            {
                ostringstream info;

                info << "Reading " << dataCount << " values";
                sendInfo(info.str().c_str());
            }

            inData = true;
        }
        else if (line.find("ENDSET") != string::npos)
        {
            inData = inSet = false;
        }
        else if (inData && inSet)
        {

            istringstream strline(buffer);
            string token;

            strline >> token;
            locationValues.push_back(strtod(token.c_str(), endptr));
            strline >> token;
            locationValues.push_back(strtod(token.c_str(), endptr));
            strline >> token;
            locationValues.push_back(strtod(token.c_str(), endptr));
        }
    }

    delete[] buffer;

    if (file.fail() && !file.eof())
    {

        ostringstream info;

        info << "Line[" << ctr << "] too long (<1024) in file " << filename;
        sendError(info.str().c_str());

        return FAIL;
    }

    {
        ostringstream info;
        info << "Read " << (locationValues.size() / 3) << " locationValues";
        sendInfo(info.str().c_str());
    }

    int valueCount = locationValues.size() / 3;

    points = new coDoPoints(mpPointsPort->getObjName(), valueCount);
    lines = new coDoLines(mpLinesPort->getObjName(), valueCount, valueCount, 1);

    makeGeos(locationValues, points, lines);

    mpPointsPort->setCurrentObject(points);
    mpLinesPort->setCurrentObject(lines);

    return SUCCESS;
}

void ReadMP::makeGeos(const std::list<float> &vertices, coDoPoints *points, coDoLines *lines, long limit)
{

    float *xPoint;
    float *yPoint;
    float *zPoint;

    float *xLine;
    float *yLine;
    float *zLine;
    int *cornerList;
    int *lineList;

    if (limit < 0)
        limit = vertices.size();

    points->getAddresses(&xPoint, &yPoint, &zPoint);
    lines->getAddresses(&xLine, &yLine, &zLine, &cornerList, &lineList);

    lineList[0] = 0;

    list<float>::const_iterator i = vertices.begin();

    for (int ctr = 0; (i != vertices.end() && ctr < limit); ++ctr)
    {

        xPoint[ctr] = *i;
        xLine[ctr] = *i;
        ++i;
        yPoint[ctr] = *i;
        yLine[ctr] = *i;
        ++i;
        zPoint[ctr] = *i;
        zLine[ctr] = *i;
        ++i;

        cornerList[ctr] = ctr;
    }
}

int main(int argc, char *argv[])
{

    ReadMP app;
    app.start(argc, argv);
    return 0;
}
