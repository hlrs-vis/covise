/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <util/byteswap.h>

#include "Read2DM.h"

#include <string>
#include <vector>
#include <cstdio>

Read2DM::Read2DM(int argc, char *argv[])
    : coModule(argc, argv, "Read 2DM files")
    , dataValuesExpected(0)
{

    this->choiceValues.push_back("(none)");
    char *choiceInitVal[] = { (char *)"(none)" };

    this->dataParameter = new coChoiceParam *[4];
    this->dataOutputPort = new coOutputPort *[4];

    this->twodmFilenameParameter = addFileBrowserParam("_2DMFilePath", "2DM file");
    this->twodmFilenameParameter->setValue("data/", "*.2dm");

    this->srhFilenameParameter = addFileBrowserParam("_SRHFilePath", "SRH file");
    this->srhFilenameParameter->setValue("data/", "*.dat");

    this->dataParameter[0] = addChoiceParam("select_data_0", "Select Output Data");
    this->dataParameter[0]->setValue(1, choiceInitVal, 0);

    this->dataParameter[1] = addChoiceParam("select_data_1", "Select Output Data");
    this->dataParameter[1]->setValue(1, choiceInitVal, 0);

    this->dataParameter[2] = addChoiceParam("select_data_2", "Select Output Data");
    this->dataParameter[2]->setValue(1, choiceInitVal, 0);

    this->dataParameter[3] = addChoiceParam("select_data_3", "Select Output Data");
    this->dataParameter[3]->setValue(1, choiceInitVal, 0);

    // Ports.
    this->polygonOutputPort = addOutputPort("polygons", "Polygons", "Geometry polygons");
    this->dataOutputPort[0] = addOutputPort("data_out0", "Float", "Data");
    this->dataOutputPort[1] = addOutputPort("data_out1", "Float", "Data");
    this->dataOutputPort[2] = addOutputPort("data_out2", "Float", "Data");
    this->dataOutputPort[3] = addOutputPort("data_out3", "Float", "Data");
}

Read2DM::~Read2DM()
{
}

void Read2DM::param(const char *paramname, bool inMapLoading)
{

    (void)inMapLoading;

    if (strcmp("_SRHFilePath", paramname) == 0)
    {

        FILE *file;
        char line[2048];
        const char *filename = this->srhFilenameParameter->getValue();

        if (!(file = fopen(filename, "r")))
        {
            sendError("Unable to open file %s", filename);
            return;
        }

        if (fgets(line, 2048, file) == 0)
        {
            sendError("Cannot read SRH header");
            return;
        }

        this->choiceValues.clear();
        this->choiceValues.push_back("(none)");

        int loc1 = 0, loc2 = 0;

        for (loc1 = 0; loc1 < strlen(line); ++loc1)
        {
            // Squeeze multiple spaces
            if (line[loc1] == ' ')
                continue;
            line[loc2++] = line[loc1];
        }

        line[loc2] = '\0';

        char *param = strtok(line, ",");

        if (param == 0)
            return;
        else
            this->choiceValues.push_back(param);

        while ((param = strtok(0, ",")) != 0)
        {
            if (strcmp("\n", param) != 0)
                this->choiceValues.push_back(param);
        }

        this->dataValuesExpected = this->choiceValues.size() - 1;

        if (!inMapLoading)
            for (int ctr = 0; ctr < 4; ++ctr)
                updateChoice(ctr);
    }

    if (strcmp("select_data_0", paramname) == 0)
    {
        updateChoice(0);
    }
    else if (strcmp("select_data_1", paramname) == 0)
    {
        updateChoice(1);
    }
    else if (strcmp("select_data_2", paramname) == 0)
    {
        updateChoice(2);
    }
    else if (strcmp("select_data_3", paramname) == 0)
    {
        updateChoice(3);
    }
}

void Read2DM::updateChoice(int number)
{

    const char **values = new const char *[this->choiceValues.size()];

    for (int ctr = 0; ctr < this->choiceValues.size(); ++ctr)
    {
        values[ctr] = this->choiceValues[ctr].c_str();
    }

    int choicePreset = this->dataParameter[number]->getValue();
    const char *oldParameter = this->dataParameter[number]->getActLabel();

    if (oldParameter != 0)
    {
        for (int pos = 0; pos < this->choiceValues.size(); ++pos)
        {
            if (this->choiceValues[pos] == oldParameter)
            {
                choicePreset = pos;
            }
        }
    }

    this->dataParameter[number]->setValue(this->choiceValues.size(), values, choicePreset);
    delete[] values;
}

int Read2DM::compute(const char *)
{

    if (!read2DMFile())
        return STOP_PIPELINE;

    readSRHFile();

    return CONTINUE_PIPELINE;
}

bool Read2DM::read2DMFile()
{
    coDoPolygons *polygonObject;

    std::string polygonObjectName;
    std::string filename = this->twodmFilenameParameter->getValue();

    std::vector<int> polygonList;
    std::vector<int> cornerList;
    std::vector<float> xCoord;
    std::vector<float> yCoord;
    std::vector<float> zCoord;

    int currentPolygon = 0;

    char line[2048];
    FILE *file;

    if (!(file = fopen(filename.c_str(), "r")))
    {
        sendError("Unable to open file %s", filename.c_str());
        return false;
    }

    if (fgets(line, 2048, file) == 0 || strncmp(line, "MESH2D", strlen("MESH2D")) != 0)
    {
        sendError("Cannot read MESH2D header");
        return false;
    }

    while (fgets(line, 2048, file) != 0)
    {
        if (strncmp(line, "E4Q", 3) == 0 || strncmp(line, "E3T", 3) == 0)
        {
            polygonList.push_back(currentPolygon);
            int numEdges;

            if (strncmp(line, "E4Q", 3) == 0)
                numEdges = 4;
            else
                numEdges = 3;

            if (strtok(line, " \t") == 0 || strtok(0, " \t") == 0)
            {
                sendError("Error in parsing elements");
                return false;
            }

            int ctr = 0;
            char *cornerString;

            while ((cornerString = strtok(0, " \t")) != 0 && ctr < numEdges)
            {
                ++ctr;
                int corner = atoi(cornerString) - 1;
                cornerList.push_back(corner);
            }

            if (ctr < numEdges)
            {
                sendError("Error in parsing elements");
                return false;
            }

            currentPolygon += numEdges;
        }
        else if (strncmp(line, "ND", 2) == 0)
        {
            if (strtok(line, " \t") == 0 || strtok(0, " \t") == 0)
            {
                sendError("Error in parsing nodes");
                return false;
            }

            const char *xCoordString = strtok(0, " \t");
            const char *yCoordString = strtok(0, " \t");
            const char *zCoordString = strtok(0, " \t");

            if (xCoordString == 0 || yCoordString == 0 || zCoordString == 0)
            {
                sendError("Error in parsing nodes");
                return false;
            }

            xCoord.push_back((float)atof(xCoordString));
            yCoord.push_back((float)atof(yCoordString));
            zCoord.push_back((float)atof(zCoordString));
        }
    }

    int *polygonListArray = new int[polygonList.size()];
    int *cornerListArray = new int[cornerList.size()];
    float *xCoordListArray = new float[xCoord.size()];
    float *yCoordListArray = new float[yCoord.size()];
    float *zCoordListArray = new float[zCoord.size()];

    copy(polygonList.begin(), polygonList.end(), polygonListArray);
    copy(cornerList.begin(), cornerList.end(), cornerListArray);
    copy(xCoord.begin(), xCoord.end(), xCoordListArray);
    copy(yCoord.begin(), yCoord.end(), yCoordListArray);
    copy(zCoord.begin(), zCoord.end(), zCoordListArray);

    polygonObjectName = this->polygonOutputPort->getObjName();
    polygonObject = new coDoPolygons(polygonObjectName.c_str(),
                                     xCoord.size(), xCoordListArray, yCoordListArray, zCoordListArray,
                                     cornerList.size(), cornerListArray,
                                     polygonList.size(), polygonListArray);

    this->polygonOutputPort->setCurrentObject(polygonObject);

    delete[] polygonListArray;
    delete[] cornerListArray;
    delete[] xCoordListArray;
    delete[] yCoordListArray;
    delete[] zCoordListArray;

    return true;
}

bool Read2DM::readSRHFile()
{
    std::vector<std::vector<float> > data(this->dataValuesExpected);

    std::string filename = this->srhFilenameParameter->getValue();

    char line[2048];
    int currentLine = 0;
    FILE *file;

    if (!(file = fopen(filename.c_str(), "r")) || fgets(line, 2048, file) == 0)
    {
        sendWarning("Unable to open file %s", filename.c_str());
        return false;
    }

    while (fgets(line, 2048, file) != 0)
    {

        ++currentLine;
        char *token = strtok(line, " \t");

        if (token != 0)
        {
            data[0].push_back((float)atof(token));
        }
        else
        {
            sendError("Not enough elements in file %s in line %d (epxected %d, found 0)", filename.c_str(), currentLine, this->dataValuesExpected);
            return false;
        }

        for (int currentData = 1; currentData < this->dataValuesExpected; ++currentData)
        {

            token = strtok(0, " \t");

            if (token != 0)
            {
                data[currentData].push_back((float)atof(token));
            }
            else
            {
                sendError("Not enough elements in file %s in line %d (epxected %d, found %d)", filename.c_str(), currentLine, this->dataValuesExpected, currentData);
                return false;
            }
        }
    }

    for (int parameterNumber = 0; parameterNumber < 4; ++parameterNumber)
    {
        int selected = this->dataParameter[parameterNumber]->getValue() - 1;

        if (selected < 0)
            continue;

        coDoFloat *floatData = new coDoFloat(this->dataOutputPort[parameterNumber]->getObjName(), data[selected].size());

        float *floatDataValues;
        floatData->getAddress(&floatDataValues);

        std::copy(data[selected].begin(), data[selected].end(), floatDataValues);

        this->dataOutputPort[parameterNumber]->setCurrentObject(floatData);
    }

    return true;
}

MODULE_MAIN(IO, Read2DM)
