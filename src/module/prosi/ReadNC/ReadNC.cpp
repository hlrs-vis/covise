/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadNC.h"

#include <iostream>
#include <sstream>

#include <util/coVector.h>

using namespace std;

ReadNC::ReadNC()
    : coModule("NC program reader")
    , points(0)
    , lines(0)
    , pointSet(0)
    , lineSet(0)
    , orientations(0)
    , angles(0)
    , feeds(0)
    , spindlespeeds(0)
    , gfunctions(0)
    , forces(0)
{

    set_module_description("Simple Reader for a NC Machine Program");

    ncPointsPort = addOutputPort("points", "coDoPoints|coDoSet", "TCP Points");
    ncLinesPort = addOutputPort("lines", "coDoLines|coDoSet", "TCP Path");

    ncOrientationPort = addOutputPort("tcp orientation", "DO_Unstructured_V3D_Normals|coDoVec3", "TCP Orientation");

    ncOrientationAnglePort = addOutputPort("tcp angle", "coDoFloat", "TCP Angle to Z");

    ncFeedPort = addOutputPort("feed", "coDoFloat", "TCP Feed");

    ncSpindleSpeedPort = addOutputPort("spindle speed", "coDoFloat", "Spindle RPM");

    ncForcesPort = addOutputPort("forces", "coDoFloat", "Forces");

    ncGFunctionPort = addOutputPort("G function", "coDoFloat", "G function");

    ncFileParam = addFileBrowserParam("ncFile", "NC file");
    //ncFileParam->setValue("/mnt/raid/cc/users/ak_te/Work/ISW/ProSi/Data01/T2_3D_Profilschlichten_2.nc","*.nc");
    ncFileParam->setValue("/mnt/raid/cc/users/ak_te/Work/ISW/ProSi/NC_FTK2003_V2.nc", "*.nc");

    ncTimestepParam = addBooleanParam("ncTimestep", "Create animated data");
    ncTimestepParam->setValue(false);

    ncNoTimestepsParam = addInt32Param("ncNoTimesteps", "Number of timesteps");
    ncNoTimestepsParam->setValue(1);
}

ReadNC::~ReadNC()
{
}

int ReadNC::compute()
{

    const char *filename = ncFileParam->getValue();
    bool isTimestep = ncTimestepParam->getValue();

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

    char *objectName = new char[max(strlen(ncPointsPort->getObjName()), strlen(ncLinesPort->getObjName())) + 64];

    int ctr = 0;

    char **endptr = 0;

    list<float> locationValues;
    list<float> orientationValues;
    list<float> feedValues;
    list<float> spindleValues;
    list<float> gfunctionValues;
    list<float> forceValues;

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;

    float feed = 0.0f;
    float spindle = 0.0f;
    float force = 0.0f;
    int g = 0;

    float spindleDirection = 0.0f;

    const string startTokens("GXYZUVWFSMK");
    bool programRuns = true;

    for (file.getline(buffer, INBUFSIZE, '\n'); (programRuns && file.good()); file.getline(buffer, INBUFSIZE, '\n'))
    {

        string line(buffer);

        // Skip masked entries
        if ((line.find("/") == line.find_first_not_of(" \t")) && (line.find("/") != string::npos))
            continue;

        // Cut comments
        string::size_type pos;
        while ((pos = line.rfind("(")) != string::npos)
        {
            line.erase(pos, line.find(")"));
        }

        bool isLineParsed = false;

        const char *lineCStr = line.c_str();

        for (int i = 0; i < startTokens.length(); ++i)
        {

            string::size_type index = line.find(startTokens[i]);

            // Covers string::npos and comment to end of set (;)
            if (index >= line.find(';'))
                continue;

            float value = strtod(&lineCStr[index + 1], endptr);

            isLineParsed = true;

            switch (line[index])
            {
            case 'G':
                g = (int)value;
                if (g > 1)
                {
                    ostringstream info;
                    info << "G function " << g << " read, but not interpreted";
                    sendWarning(info.str().c_str());
                }
                break;
            case 'X':
                x = value;
                break;
            case 'Y':
                y = value;
                break;
            case 'Z':
                z = value;
                break;
            case 'U':
                u = value;
                break;
            case 'V':
                v = value;
                break;
            case 'W':
                w = value;
                break;
            case 'F':
                feed = value;
                break;
            case 'S':
                spindle = spindleDirection * value;
                break;
            case 'K':
                force = value;
                break;
            case 'M':
                if (value == 2.0f)
                {
                    programRuns = false; // M02 program end
                }
                else if (value == 3.0f)
                {
                    spindleDirection = 1.0f; // M03 CW spin
                }
                else if (value == 4.0f)
                {
                    spindleDirection = -1.0f; // M04 CCW spin
                }
                else if (value == 5.0f)
                {
                    spindleDirection = 0.0f; // M05 spin off
                }
                else if (value == 30.0f)
                {
                    programRuns = false; // M30 program end
                }
                else
                {
                    ostringstream info;
                    info << "Unknown M function " << ((int)value);
                    sendWarning(info.str().c_str());
                }
                break;
            default:
                ;
            }
        }

        if (isLineParsed)
        {

            ++ctr;

            locationValues.push_back(x);
            locationValues.push_back(y);
            locationValues.push_back(z);

            orientationValues.push_back(u);
            orientationValues.push_back(v);
            orientationValues.push_back(w);

            feedValues.push_back(feed);
            spindleValues.push_back(spindle);
            gfunctionValues.push_back(g);
            forceValues.push_back(force);
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

    if (!isTimestep) // already created the data
    {

        points = new coDoPoints(ncPointsPort->getObjName(), valueCount);
        lines = new coDoLines(ncLinesPort->getObjName(), valueCount, valueCount, 1);

        makeGeos(locationValues, points, lines);
    }
    else
    {

        int noOfTimesteps = ncNoTimestepsParam->getValue();
        float stepping = ((float)valueCount) / ((float)noOfTimesteps);

        for (int ctr = 1; ctr <= noOfTimesteps; ++ctr)
        {

            int limit = (int)(ctr * stepping);

            sprintf(objectName, "%s_%d", ncPointsPort->getObjName(), ctr);
            points = new coDoPoints(objectName, limit);
            sprintf(objectName, "%s_%d", ncLinesPort->getObjName(), ctr);
            lines = new coDoLines(objectName, limit, limit, 1);

            makeGeos(locationValues, points, lines, limit);

            pointsList.push_back(points);
            linesList.push_back(lines);
        }
    }

    orientations = new coDoVec3(ncOrientationPort->getObjName(), valueCount);
    angles = new coDoFloat(ncOrientationAnglePort->getObjName(), valueCount);
    feeds = new coDoFloat(ncFeedPort->getObjName(), valueCount);
    spindlespeeds = new coDoFloat(ncSpindleSpeedPort->getObjName(), valueCount);
    gfunctions = new coDoFloat(ncGFunctionPort->getObjName(), valueCount);
    forces = new coDoFloat(ncForcesPort->getObjName(), valueCount);

    float *data1Array;
    float *data2Array;
    float *data3Array;

    float *angleArray;

    float *spdArray;

    float *spindleSpeedArray;

    float *forceArray;

    float *gFunctionArray;

    orientations->getAddresses(&data1Array, &data2Array, &data3Array);
    angles->getAddress(&angleArray);
    feeds->getAddress(&spdArray);
    spindlespeeds->getAddress(&spindleSpeedArray);
    gfunctions->getAddress(&gFunctionArray);
    forces->getAddress(&forceArray);

    coVector zAxis(0.0f, 0.0f, 1.0f);

    list<float>::const_iterator ov = orientationValues.begin();
    for (int ctr = 0; ov != orientationValues.end(); ++ctr)
    {

        data1Array[ctr] = *ov;
        ++ov;
        data2Array[ctr] = *ov;
        ++ov;
        data3Array[ctr] = *ov;
        ++ov;

        angleArray[ctr] = coVector(data1Array[ctr], data2Array[ctr], data3Array[ctr]).enclosedAngle(zAxis);
        //cerr << angleArray[ctr] << endl;
        //cerr << "[" << data1[ctr] << "|" << data2[ctr] << "|" << data3[ctr] << "]" << endl;
    }

    list<float>::const_iterator sv = feedValues.begin();
    for (int ctr = 0; sv != feedValues.end(); ++ctr)
    {
        spdArray[ctr] = *sv;
        ++sv;
    }

    list<float>::const_iterator ssv = spindleValues.begin();
    for (int ctr = 0; ssv != spindleValues.end(); ++ctr)
    {
        spindleSpeedArray[ctr] = *ssv;
        ++ssv;
    }

    list<float>::const_iterator gf = gfunctionValues.begin();
    for (int ctr = 0; gf != gfunctionValues.end(); ++ctr)
    {
        gFunctionArray[ctr] = *gf;
        ++gf;
    }

    list<float>::const_iterator fc = forceValues.begin();
    for (int ctr = 0; fc != forceValues.end(); ++ctr)
    {
        forceArray[ctr] = *fc;
        ++fc;
    }

    if (isTimestep)
    {

        coDistributedObject **pointSteps = new coDistributedObject *[pointsList.size() + 1];
        coDistributedObject **lineSteps = new coDistributedObject *[linesList.size() + 1];

        list<coDoPoints *>::iterator pi = pointsList.begin();
        list<coDoLines *>::iterator li = linesList.begin();

        int ctr;

        for (ctr = 0; pi != pointsList.end() && li != linesList.end(); ++ctr)
        {
            pointSteps[ctr] = *pi;
            lineSteps[ctr] = *li;
            ++pi;
            ++li;
        }

        pointSteps[ctr] = 0;
        lineSteps[ctr] = 0;

        pointSet = new coDoSet(ncPointsPort->getObjName(), pointSteps);
        lineSet = new coDoSet(ncLinesPort->getObjName(), lineSteps);

        for (int ctr2 = 0; ctr2 < ctr; ++ctr2)
        {
            delete pointSteps[ctr];
            delete lineSteps[ctr];
        }

        delete[] pointSteps;
        delete[] lineSteps;

        ostringstream range;
        range << "0 " << ctr;

        pointSet->addAttribute("TIMESTEP", range.str().c_str());
        lineSet->addAttribute("TIMESTEP", range.str().c_str());
    }

    if (isTimestep)
    {
        ncPointsPort->setCurrentObject(pointSet);
        ncLinesPort->setCurrentObject(lineSet);
    }
    else
    {
        ncPointsPort->setCurrentObject(points);
        ncLinesPort->setCurrentObject(lines);
    }

    ncOrientationPort->setCurrentObject(orientations);
    ncOrientationAnglePort->setCurrentObject(angles);
    ncFeedPort->setCurrentObject(feeds);
    ncSpindleSpeedPort->setCurrentObject(spindlespeeds);
    ncGFunctionPort->setCurrentObject(gfunctions);
    ncForcesPort->setCurrentObject(forces);

    return SUCCESS;
}

void ReadNC::makeGeos(const std::list<float> &vertices, coDoPoints *points, coDoLines *lines, long limit)
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

    ReadNC app;
    app.start(argc, argv);
    return 0;
}
