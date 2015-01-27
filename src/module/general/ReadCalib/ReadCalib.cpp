/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
 **                                                   	                  **
 **                                                                        **
 ** Description: READ imk KA maps             	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <do/coDoStructuredGrid.h>
#include "ReadCalib.h"

ReadCalib::ReadCalib(int argc, char *argv[])
    : coModule(argc, argv, "Calib file Reader")
{
    // the output ports
    p_meshSoll = addOutputPort("mesh", "StructuredGrid", "mesh");
    p_meshIst = addOutputPort("measurement", "StructuredGrid", "measurement");

    //files
    p_GridPath = addFileBrowserParam("CalibFile", "Calibration file");
    p_GridPath->setValue("data/calib.asc", "*.asc*");
}

ReadCalib::~ReadCalib()
{
}

// =======================================================

int ReadCalib::compute(const char *)
{
    // open the file
    int i, j, k;
    int nx, ny, nz;
    float minP[3], maxP[3];

    //int n=0;
    int line = 0;
    bool hasDim = false;
    bool hasMin = false;
    bool hasMax = false;
    float *xCoord, *yCoord, *zCoord;
    float *xCoordS, *yCoordS, *zCoordS;
    coDoStructuredGrid *sollGrid = NULL;
    coDoStructuredGrid *istGrid = NULL;

    FILE *fp = fopen(p_GridPath->getValue(), "r");
    if (fp == NULL)
    {
        return FAIL;
    }
    char buf[1000];
    // read header
    while ((!feof(fp)) && (hasDim == false || hasMin == false || hasMax == false))
    {
        if (fgets(buf, 1000, fp) == NULL)
            cerr << "fgets returned with error " << endl;
        else
        {
            line++;
            if ((buf[0] != '%') && (strlen(buf) > 5))
            {
                if (strncasecmp(buf, "DIM", 3) == 0)
                {
                    int iret = sscanf(buf + 3, "%d %d %d", &nx, &ny, &nz);
                    if (iret != 3)
                        cerr << "sscanf has read " << iret << "elements" << endl;
                    hasDim = true;
                }
                else if (strncasecmp(buf, "MIN", 3) == 0)
                {
                    int iret = sscanf(buf + 3, "%f %f %f", &minP[0], &minP[1], &minP[2]);
                    if (iret != 3)
                        cerr << "sscanf2 has read " << iret << "elements" << endl;
                    hasMin = true;
                }
                else if (strncasecmp(buf, "MAX", 3) == 0)
                {
                    int iret = sscanf(buf + 3, "%f %f %f", &maxP[0], &maxP[1], &maxP[2]);
                    if (iret != 3)
                        cerr << "sscanf3 has read " << iret << "elements" << endl;
                    hasMax = true;
                }
                else
                {
                    cerr << "Unknown statement in line " << line << endl;
                }
            }
        }
    }

    if (!(nx == 0 || ny == 0 || nz == 0))
    {

        sollGrid = new coDoStructuredGrid(p_meshSoll->getObjName(), nx, ny, nz);
        istGrid = new coDoStructuredGrid(p_meshIst->getObjName(), nx, ny, nz);
        sollGrid->getAddresses(&xCoord, &yCoord, &zCoord);
        sollGrid->addAttribute("COLOR", "green");
        istGrid->getAddresses(&xCoordS, &yCoordS, &zCoordS);
        p_meshSoll->setCurrentObject(sollGrid);
        p_meshIst->setCurrentObject(istGrid);
        float px, py, pz;
        float orientation[9];
        //int num = nx*ny*nz;
        while (!feof(fp))
        {
            if (fgets(buf, 1000, fp) == NULL)
                cerr << "fgets returned with error " << endl;
            else
            {
                line++;
                if ((buf[0] != '%') && (strlen(buf) > 5))
                {
                    int ntok = sscanf(buf, "%d %d %d", &i, &j, &k);
                    if (ntok == 3) // all three numbers parsed
                    {
                        //n = i*ny*nz+j*nz+k;
                        int nums = sscanf(buf, "%d %d %d    %f %f %f    %f %f %f  %f %f %f  %f %f %f",
                                          &i, &j, &k,
                                          &px, &py, &pz,
                                          &orientation[0], &orientation[1], &orientation[2],
                                          &orientation[3], &orientation[4], &orientation[5],
                                          &orientation[6], &orientation[7], &orientation[8]);

                        if (nums != 15)
                        {
                            fprintf(stderr, "error parsing calib file\n");
                        }
                        int index = i * ny * nz + ((ny - 1) - j) * nz + k;
                        xCoord[index] = px;
                        yCoord[index] = py;
                        zCoord[index] = pz;
                        xCoordS[index] = minP[0] + i * (maxP[0] - minP[0]) / (nx - 1);
                        yCoordS[index] = minP[1] + ((ny - 1) - j) * (maxP[1] - minP[1]) / (ny - 1);
                        zCoordS[index] = minP[2] + k * (maxP[2] - minP[2]) / (nz - 1);
                    }
                }
            }
        }
    }

    return SUCCESS;
}

MODULE_MAIN(IO, ReadCalib)
