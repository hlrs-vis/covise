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

#include "ReadSeis.h"

#include <util/coviseCompat.h>
#include <ctype.h>

// Our input file was created on an intel system,
// so byteswapping has to be reversed.
#ifdef BYTESWAP
#undef BYTESWAP
#else
#define BYTESWAP
#endif

//#define VERBOSE

ReadSeis::ReadSeis(int argc, char *argv[])
    : coModule(argc, argv, "Read IHF volume data")
{

    filename = 0;

    gridPort = 0;
    dataPort = 0;
    fileParam = 0;

    gridDataObject = 0;
    scalarDataObject = 0;

    xDimension = 0;
    yDimension = 0;
    zDimension = 0;

    gridspacing = 1.0f;

    set_module_description("ASCII file reader for SEIS data");

    // the output port
    gridPort = addOutputPort("grid", "UniformGrid", "grid data");
    dataPort = addOutputPort("data", "Float", "scalar data");

    // select the OBJ file name with a file browser
    fileParam = addFileBrowserParam("data_file", "SEIS File");
    fileParam->setValue("/mnt/pro/cod/hlrs/seis/seisdata-small.dat", "*.dat");

    cerr << "ReadSeis::<init> info: done" << endl;
}

ReadSeis::~ReadSeis()
{
}

void ReadSeis::quit(void)
{
}

int ReadSeis::compute(const char *)
{

    // get the file name

    filename = fileParam->getValue();

    if (filename == NULL)
    {
        sendError("An input file has to be specified");
        return FAIL;
    }
    xDimension = 0;
    yDimension = 0;
    zDimension = 0;
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        sendError("could not open file");
        return FAIL;
    }
    bool firstTime = true;
    while (!feof(fp))
    {
        if (fgets(line, 20000, fp))
        {
            float dummy;
            int xd = 0, yd = 0, zd = 0;
            if (sscanf(line, "%f %f %f %f %f %f %d %d %d", &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &xd, &yd, &zd) == 9)
            {
                if (firstTime)
                {

                    int n = 0;
                    char *c = line;
                    while (*c != '\0')
                    {
                        n++;
                        while ((*c != '\0') && (*c != ' ' && *c != '\t' && *c != '\r' && *c != '\n'))
                        {
                            c++;
                        }
                        while ((*c != '\0') && (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n'))
                        {
                            c++;
                        }
                    }
                    xDimension = n - 9;
                    firstTime = false;
                }
                if (yd > yDimension)
                {
                    yDimension = yd;
                }
                if (zd > zDimension)
                {
                    zDimension = zd;
                }
            }
        }
        else
        {
            break;
        }
    }
    yDimension++;
    zDimension++;

    fclose(fp);

    cerr << "ReadSeis::compute info: done parsing" << endl;

    gridspacing = 1.0f;

    gridDataObject = new coDoUniformGrid(gridPort->getObjName(),
                                         xDimension, yDimension, zDimension,
                                         0.0f, gridspacing * (float)xDimension,
                                         0.0f, gridspacing * (float)yDimension,
                                         0.0f, gridspacing * (float)zDimension);

    scalarDataObject = new coDoFloat(dataPort->getObjName(),
                                     xDimension * yDimension * zDimension);

    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        sendError("could not open file");
        return FAIL;
    }
    float Value = 0.0;

    float *xValues = NULL;

    scalarDataObject->getAddress(&xValues);

    int i, j, k;
    for (j = 0; j < yDimension; j++)
    {
        for (k = 0; k < zDimension; k++)
        {
            if (fgets(line, 20000, fp))
            {
                float dummy;
                int xd = 0, yd = 0, zd = 0;
                if (sscanf(line, "%f %f %f %f %f %f %d %d %d", &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &xd, &yd, &zd) == 9)
                {
                    int n = 0;
                    char *c = line;
                    while (*c != '\0')
                    {
                        n++;
                        while ((*c != '\0') && (*c != ' ' && *c != '\t' && *c != '\r' && *c != '\n'))
                        {
                            c++;
                        }
                        while ((*c != '\0') && (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n'))
                        {
                            c++;
                        }
                        if (n == 9)
                            break;
                    }
                    for (i = 0; i < xDimension; i++)
                    {
                        sscanf(c, "%f", &Value);
                        int index = k + j * zDimension + i * zDimension * yDimension;
                        //int index = i + j * xDimension + k * xDimension * yDimension;

                        xValues[index] = (Value);
                        while ((*c != '\0') && (*c != ' ' && *c != '\t' && *c != '\r' && *c != '\n'))
                        {
                            c++;
                        }
                        while ((*c != '\0') && (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n'))
                        {
                            c++;
                        }
                        if (*c == '\0')
                            break;
                    }
                }
            }
            else
            {
                break;
            }
        }
    }
    fclose(fp);

    gridPort->setCurrentObject(gridDataObject);
    dataPort->setCurrentObject(scalarDataObject);

    cerr << "ReadSeis::compute info: finished" << endl;

    return SUCCESS;
}

MODULE_MAIN(IO, ReadSeis)
