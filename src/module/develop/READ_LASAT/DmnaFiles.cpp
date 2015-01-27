/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS   DmnaFiles
//
// Description:
//
//
// Initial version: 11.12.2002 (CS)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// All Rights Reserved.
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// $Id: DmnaFiles.cpp,v 1.3 2002/12/17 13:22:21 ralf Exp $
//
#include "DmnaFiles.h"
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#ifdef __linux
#include <unistd.h>
#endif
#include <string.h>
//#include "coDoFloat.h"
DmnaFiles::DmnaFiles(const char *firstFile, const char *zFile, const char *o_name, char *&errMsg)
{
    char *filename;
    int i, j, k;

    errMsg = NULL;
    obj_name = new char[strlen(o_name) + 1];
    strcpy(obj_name, o_name);
    xDim = yDim = zDim = 0;

    sdata = new float **[DIM];
    if (!sdata)
    {
        errMsg = new char[1024];
        sprintf(errMsg, "could not allocate float");
        return;
    }

    for (i = 0; i < DIM; i++)
    {
        sdata[i] = new float *[DIM];
        if (!sdata[i])
            return;
        for (j = 0; j < DIM; j++)
        {
            sdata[i][j] = new float[DIM];
            if (!sdata[i][j])
                return;
            for (k = 0; k < DIM; k++)
            {
                sdata[i][j][k] = 0.;
            }
        }
    }

    files = new coStepFile(firstFile);

    char buffer[MAXLINE];
    char *dig;

    files->get_nextpath(&filename);
    while (filename)
    {
        //fd = ::open(filename, O_RDONLY);
        ifstream input(filename); // input file stream
        if (input.fail())
        {
            errMsg = new char[1024];
            sprintf(errMsg, "sorry couldn't open file %s", filename);
            return;
        }

        while (input.getline(buffer, MAXLINE))
        {
            if (strstr(buffer, "delta") != NULL)
            {
                strtok(buffer, "\t");
                sscanf(strtok(NULL, "\t"), "%f", &delta);
            }
            if (strstr(buffer, "ymin") != NULL)
            {
                strtok(buffer, "\t");
                sscanf(strtok(NULL, "\t"), "%f", &yMin);
            }
            if (strstr(buffer, "xmin") != NULL)
            {
                strtok(buffer, "\t");
                sscanf(strtok(NULL, "\t"), "%f", &xMin);
            }
            if (strstr(buffer, "hghb") != NULL)
            {
                strtok(buffer, "\t");
                sscanf(strtok(NULL, "\t"), "%d", &xDim);
                sscanf(strtok(NULL, "\t"), "%d", &yDim);

                input.getline(buffer, MAXLINE); //skip

                for (i = 0; i < xDim; i++)
                {
                    input.getline(buffer, MAXLINE);
                    strtok(buffer, " "); // skip tab
                    dig = strtok(NULL, " ");
                    dig[strlen(dig) - 1] = '\0';
                    sscanf(dig, "%f", &sdata[i][0][zDim]);

                    for (j = 1; j < yDim; j++)
                    {
                        dig = strtok(NULL, " ");
                        if (!dig)
                        {
                            char buf[400];
                            sprintf(buf, "error at %d in %s", j, filename);
                            coModule::sendError(buf);
                        }
                        else
                        {
                            dig[strlen(dig) - 1] = '\0';
                            sscanf(dig, "%f", &sdata[i][j][zDim]);
                        }
                    }
                }
            }
        }
        zDim++;
        input.close();
        files->get_nextpath(&filename);
    }
}

coDistributedObject *
DmnaFiles::getData()
{
    coDoFloat *strdata = new coDoFloat(obj_name, xDim, yDim, zDim);

    if (strdata && strdata->objectOk())
    {
        float *sd;
        strdata->getAddress(&sd);
        int i, j, k;

        for (k = 0; k < zDim; k++)
        {
            for (j = 0; j < yDim; j++)
            {
                for (i = 0; i < xDim; i++)
                {
                    sd[i * yDim * zDim + j * zDim + k] = sdata[i][j][k];
                }
            }
        }
        return strdata;
    }
    return NULL;
}

DmnaFiles::~DmnaFiles()
{
    int i, j;
    delete[] obj_name;

    for (i = 0; i < DIM; i++)
    {
        for (j = 0; j < DIM; j++)
        {
            delete[] sdata[i][j];
        }
        delete[] sdata[i];
    }
    delete[] sdata;
}

//
// History:
//
// $Log: DmnaFiles.cpp,v $
// Revision 1.3  2002/12/17 13:22:21  ralf
// - made it WIN compliant
//
// Revision 1.2  2002/12/16 14:26:35  cs_te
// -
//
// Revision 1.1  2002/12/12 11:58:58  cs_te
// initial version
//
//
