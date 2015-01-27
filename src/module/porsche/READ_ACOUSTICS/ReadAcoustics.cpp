/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                   (C)2000 VirCinity IT-Consulting GmbH **
 **                                                                        **
 ** Description: Simple Reader for Head Audio Akustik	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: A. Wierse                                                      **
 **                                                                        **
 ** History:                                                               **
 ** May 00           v1                                                    **
 **                                                                        **
\**************************************************************************/
#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

//lenght of a line
#define LINE_SIZE 8192

// portion for resizing data
#define CHUNK_SIZE 4096

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadAcoustics.h"

void main(int argc, char *argv[])
{
    ReadAcoustics *application = new ReadAcoustics();
    application->start(argc, argv);
}

ReadAcoustics::ReadAcoustics()
{

    // this info appears in the module setup window
    set_module_description("Reader for Head Acoustics data ASCII");

    // the output port
    matrix1Port = addOutputPort("matrix1", "coDoStructuredGrid", "grid of first data set");
    matrix2Port = addOutputPort("matrix2", "coDoStructuredGrid", "grid of second data set");
    data1Port = addOutputPort("data1", "coDoFloat", "first data set");
    data2Port = addOutputPort("data2", "coDoFloat", "second data set");

    // select the OBJ file name with a file browser
    objFileParam = addFileBrowserParam("objFile", "OBJ file");
    objFileParam->setValue("objFile", "data/ *.obj");
}

ReadAcoustics::~ReadAcoustics()
{
}

void ReadAcoustics::quit()
{
}

int ReadAcoustics::compute()
{

    char infobuf[500]; // buffer for COVISE info and error messages

    // get the file name
    filename = objFileParam->getValue();

    if (filename != NULL)
    {
        // open the file
        if (openFile())
        {
            sprintf(infobuf, "File %s open", filename);
            sendInfo(infobuf);

            // read the file, create the lists and create a COVISE polygon object
            readFile();
            return SUCCESS;
        }
        else
        {
            sprintf(infobuf, "Error opening file %s", filename);
            sendError(infobuf);
            return FAIL;
        }
    }
    else
    {
        sendError("ERROR: fileName is NULL");
        return FAIL;
    }
}

void
ReadAcoustics::readFile()
{
    int i, j, k;
    int idummy;
    float fdummy;
    float *xc, *yc, *zc, *sd;
    char buffer[255 * 20];

    coDoStructuredGrid *matrix1Object; // output object
    coDoStructuredGrid *matrix2Object; // output object
    coDoFloat *data1Object;
    coDoFloat *data2Object;

    const char *matrix1ObjectName; // output object name assigned by the controller
    const char *matrix2ObjectName; // output object name assigned by the controller
    const char *data1ObjectName; // output object name assigned by the controller
    const char *data2ObjectName; // output object name assigned by the controller

    fscanf(fp, "%d %d\n", &xdim, &ydim);
    cerr << "reading " << xdim << " columns and " << ydim << " rows\n";

    // get the COVISE output object name from the controller
    matrix1ObjectName = matrix1Port->getObjName();
    data1ObjectName = data1Port->getObjName();

    // create the COVISE output object
    matrix1Object = new coDoStructuredGrid(matrix1ObjectName, xdim, ydim, 1);
    matrix1Object->getAddresses(&xc, &yc, &zc);
    data1Object = new coDoFloat(data1ObjectName, xdim, ydim, 1);
    data1Object->getAddress(&sd);

    fgets(buffer, 255 * 20, fp);

    float dy = 1.0 / (ydim - 1.0);
    float dx = 1.0 / (xdim - 1.0);

    for (i = 0; i < ydim; i++)
    {
        fscanf(fp, "%d", &idummy);
        cerr << "idummy[" << i << ", 0]: " << idummy << endl;
        for (j = 0; j < xdim; j++)
        {
            fscanf(fp, "%f", &fdummy);
            //	    cerr << " " << fdummy << " ";
            xc[j * (ydim) + i] = j * dx;
            yc[j * (ydim) + i] = i * dy;
            zc[j * (ydim) + i] = fdummy;
            sd[j * (ydim) + i] = fdummy;
            //	    sd[i * (xdim) + j] = fdummy;
        }
        //	cerr << endl;
    }
    matrix1Port->setCurrentObject(matrix1Object);
    data1Port->setCurrentObject(data1Object);
}

int ReadAcoustics::openFile()
{
    char infobuf[300];

    strcpy(infobuf, "Opening file ");
    strcat(infobuf, filename);
    sendInfo(infobuf);

    // open the obj file
    if ((fp = fopen((char *)filename, "r")) == NULL)
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, filename);
        sendError(infobuf);
        return (FALSE);
    }
    else
    {
        return (TRUE);
    }
}
