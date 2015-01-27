/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadHyperMill.h"

#include <iostream>
#include <sstream>
#include <list>

#include <util/coVector.h>

using namespace std;

ReadHyperMill::ReadHyperMill()
    : coModule("HyperMill program reader")
    , points(0)
    , lines(0)
    , orientations(0)
{

    set_module_description("Simple Reader for HyperMill Data");

    hmPointsPort = addOutputPort("points", "coDoPoints", "tcp points");
    hmLinesPort = addOutputPort("lines", "coDoLines", "tcp path");

    hmOrientationPort = addOutputPort("tcp orientation", "DO_Unstructured_V3D_Normals|coDoVec3", "tcp orientation");

    hmOrientationAnglePort = addOutputPort("tcp angle", "coDoFloat", "tcp angle to z");

    //   hmFeedPort =
    //     addOutputPort("feed", "coDoFloat", "tcp feed");

    hmFileParam = addFileBrowserParam("hmFile", "HyperMill file");
    //hmFileParam->setValue("/mnt/raid/cc/users/ak_te/Work/ISW/ProSi/Data01/T2_3D_Profilschlichten_2.nc","*.nc");
    hmFileParam->setValue("/mnt/raid/cc/users/ak_te/Work/ISW/ProSi/Data01/Test01_2.POF", "*.POF");
}

ReadHyperMill::~ReadHyperMill()
{
}

int ReadHyperMill::compute()
{

    const char *filename = hmFileParam->getValue();

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
    list<float> orientationValues;
    list<float> feedValues;

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;

    float feed = 0.0f;

    for (file.getline(buffer, INBUFSIZE, '\n'); file.good(); file.getline(buffer, INBUFSIZE, '\n'))
    {

        ++ctr;

        istringstream line(buffer);
        string token;

        line >> token;
        line >> token;

        if (token[0] != 'o')
            continue;

        do
        {
            line >> token;
            switch (token[0])
            {
            case 'X':
            case 'x':
                x = strtod(token.substr(3, token.length() - 1).c_str(), endptr);
                break;
            case 'Y':
            case 'y':
                y = strtod(token.substr(3, token.length() - 1).c_str(), endptr);
                break;
            case 'Z':
            case 'z':
                z = strtod(token.substr(3, token.length() - 1).c_str(), endptr);
                break;
            case 'U':
            case 'u':
                u = strtod(token.substr(3, token.length() - 1).c_str(), endptr);
                break;
            case 'V':
            case 'v':
                v = strtod(token.substr(3, token.length() - 1).c_str(), endptr);
                break;
            case 'W':
            case 'w':
                w = strtod(token.substr(3, token.length() - 1).c_str(), endptr);
                break;
            case 'F':
            case 'f':
                feed = strtod(token.substr(3, token.length() - 1).c_str(), endptr);
            default:
                ;
            }
        } while (!line.eof());

        locationValues.push_back(x);
        locationValues.push_back(y);
        locationValues.push_back(z);

        orientationValues.push_back(u);
        orientationValues.push_back(v);
        orientationValues.push_back(w);

        feedValues.push_back(feed);
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

    points = new coDoPoints(hmPointsPort->getObjName(), valueCount);
    lines = new coDoLines(hmLinesPort->getObjName(), valueCount, valueCount, 1);
    orientations = new coDoVec3(hmOrientationPort->getObjName(), valueCount);
    angles = new coDoFloat(hmOrientationAnglePort->getObjName(), valueCount);
    //feeds = new coDoFloat(hmFeedPort->getObjName(), valueCount);

    float *xPoint;
    float *yPoint;
    float *zPoint;

    float *xLine;
    float *yLine;
    float *zLine;
    int *cornerList;
    int *lineList;

    float *data1;
    float *data2;
    float *data3;

    float *angle;

    //float * spd;

    points->getAddresses(&xPoint, &yPoint, &zPoint);
    lines->getAddresses(&xLine, &yLine, &zLine, &cornerList, &lineList);
    orientations->getAddresses(&data1, &data2, &data3);
    angles->getAddress(&angle);
    //feeds->getAddress(&spd);

    lineList[0] = 0;
    coVector zAxis(0.0f, 0.0f, 1.0f);

    list<float>::const_iterator i = locationValues.begin();

    for (int ctr = 0; i != locationValues.end(); ++ctr)
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

    list<float>::const_iterator ov = orientationValues.begin();
    for (int ctr = 0; ov != orientationValues.end(); ++ctr)
    {

        data1[ctr] = *ov;
        ++ov;
        data2[ctr] = *ov;
        ++ov;
        data3[ctr] = *ov;
        ++ov;

        angle[ctr] = coVector(data1[ctr], data2[ctr], data3[ctr]).enclosedAngle(zAxis);

        //cerr << "[" << data1[ctr] << "|" << data2[ctr] << "|" << data3[ctr] << "]" << endl;
    }

    //   list<float>::const_iterator sv = feedValues.begin();
    //   for (int ctr = 0; sv != feedValues.end(); ++ctr) {
    //     spd[ctr] = *sv;
    //     ++sv;
    //   }

    hmPointsPort->setCurrentObject(points);
    hmLinesPort->setCurrentObject(lines);
    hmOrientationPort->setCurrentObject(orientations);
    hmOrientationAnglePort->setCurrentObject(angles);
    //hmFeedPort->setCurrentObject(feeds);

    return SUCCESS;
}

int main(int argc, char *argv[])
{

    ReadHyperMill app;
    app.start(argc, argv);
    return 0;
}
