/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***********************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 ************************************************************************/

/************************************************************************/

#include "ReadITA.h"

#include <util/coviseCompat.h>
#include <ctype.h>
#include <sstream>

ReadITA::ReadITA(int argc, char *argv[])
    : coModule(argc, argv, "Read ITA field data")
{

    filename = 0;

    gridPort = 0;
    dataPort = 0;
    fileParam = 0;

    gridDataObject = 0;
    scalarDataObject = 0;

    set_module_description("Read ITA field data");

    // the output port
    gridPort = addOutputPort("surface", "Polygons", "Surface Polygons");
    dataPort = addOutputPort("data", "Float", "Amplitude");

    // select the OBJ file name with a file browser
    fileParam = addFileBrowserParam("data_file", "ITA File");
    fileParam->setValue("/data/ITA/test.txt", "*.txt");
}

ReadITA::~ReadITA()
{
}

void ReadITA::quit(void)
{
}

int ReadITA::compute(const char *)
{

    // get the file name

    filename = fileParam->getValue();

    if (filename == NULL)
    {
        sendError("An input file has to be specified");
        return FAIL;
    }
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        sendError("could not open file");
        return FAIL;
    }
    bool firstTime = true;
    int numTimesteps = 0;
    int numValues = 0;
    while (!feof(fp))
    {
        if (fgets(line, 20000, fp))
        {
            for (char *c = line; *c != '\0'; c++)
            {
                if (*c == ',')
                    *c = '.';
            }
            if (numTimesteps == 0)
            {
                stringstream s(line);
                float phi;
                float theta;
                s >> phi;
                s >> theta;
                while (!s.eof())
                {
                    float Amplitude;
                    s >> Amplitude;
                    numTimesteps++;
                    if (numTimesteps > 10000)
                    {
                        sendError("parse error in first line, expected float values only");
                        return FAIL;
                    }
                }
                numTimesteps--;
            }
            Amplitudes a;
            a.values.resize(numTimesteps);

            stringstream s(line);
            s >> a.phi;
            s >> a.theta;
            a.phi = a.phi / 180.0 * M_PI;
            a.theta = a.theta / 180.0 * M_PI;
            for (int i = 0; i < numTimesteps; i++)
            {
                s >> a.values[i];
            }
            table.push_back(a);
        }
        else
        {
            break;
        }
    }
    fclose(fp);
    int numPhi = 0;
    int numTheta = 0;
    for (int i = 1; i < table.size(); i++)
    {
        if (table[0].phi == table[i].phi)
        {
            numPhi = i;
            break;
        }
    }
    numTheta = table.size() / numPhi;
    coDistributedObject **polygons = new coDistributedObject *[numTimesteps + 1];
    for (int t = 0; t < numTimesteps; t++)
    {
        char *objName = new char[strlen(gridPort->getObjName()) + 100];
        sprintf(objName, "%s_%d", gridPort->getObjName(), t);
        coDoPolygons *polygonObject;
        polygonObject = new coDoPolygons(objName, table.size(), (numPhi - 1) * (numTheta - 1) * 4, (numPhi - 1) * (numTheta - 1));
        float *x_c, *y_c, *z_c;
        int *v_l;
        int *p_l;
        polygonObject->getAddresses(&x_c, &y_c, &z_c, &v_l, &p_l);
        int index = 0;
        for (int i = 0; i < table.size(); i++)
        {
            x_c[i] = table[i].values[t] * cos(table[i].phi) * cos(table[i].theta);
            y_c[i] = table[i].values[t] * cos(table[i].phi) * sin(table[i].theta);
            z_c[i] = table[i].values[t] * sin(table[i].phi);
        }
        int polygonNumber = 0;
        for (int p = 0; p < (numPhi - 1); p++)
        {
            for (int t = 0; t < (numTheta - 1); t++)
            {
                int polIndex = (p * (numTheta - 1) + t) * 4;
                v_l[polIndex] = p * (numTheta) + t;
                v_l[polIndex + 1] = p * (numTheta) + t + 1;
                v_l[polIndex + 2] = (p + 1) * (numTheta) + t + 1;
                v_l[polIndex + 3] = (p + 1) * (numTheta) + t;
                p_l[polygonNumber] = polIndex;
                polygonNumber++;
            }
        }
        polygons[t] = polygonObject;
        polygons[t + 1] = 0;
    }
    coDoSet *polygonSet = new coDoSet(gridPort->getObjName(), polygons);
    polygonSet->addAttribute("TIMESTEP", "1 100");
    coDistributedObject **dataObjs = new coDistributedObject *[numTimesteps + 1];
    for (int t = 0; t < numTimesteps; t++)
    {
        char *objName = new char[strlen(dataPort->getObjName()) + 100];
        sprintf(objName, "%s_%d", dataPort->getObjName(), t);
        coDoFloat *dataObject;
        dataObject = new coDoFloat(objName, table.size());
        float *f_v;
        dataObject->getAddress(&f_v);
        int index = 0;
        for (int i = 0; i < table.size(); i++)
        {
            f_v[i] = table[i].values[t];
        }
        dataObjs[t] = dataObject;
        dataObjs[t + 1] = 0;
    }
    coDoSet *dataSet = new coDoSet(dataPort->getObjName(), dataObjs);
    dataSet->addAttribute("TIMESTEP", "1 100");

    gridPort->setCurrentObject(polygonSet);
    dataPort->setCurrentObject(dataSet);

    return SUCCESS;
}

MODULE_MAIN(IO, ReadITA)
