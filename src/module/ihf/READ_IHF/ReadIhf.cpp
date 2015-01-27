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

#define ProgrammName "Reader for IHF Volume Data"
#define Kurzname "ReadIhf"
#define Copyright "(c) 2002 RUS Rechenzentrum der Uni Stuttgart"
#define Autor "Andreas Kopecki"
#define letzteAenderung "ako - 26.08.2002"

/************************************************************************/

#include "ReadIhf.h"

#include <stdlib.h>
#include <ctype.h>

#include <covise/covise_byteswap.h>

// Our input file was created on an intel system,
// so byteswapping has to be reversed.
#ifdef BYTESWAP
#undef BYTESWAP
#else
#define BYTESWAP
#endif

//#define VERBOSE

int main(int argc, char *argv[])
{
    ReadIhf *application = new ReadIhf();
    application->start(argc, argv);
    return 0;
}

void ReadIhf::param(const char *name)
{
}

ReadIhf::ReadIhf()
{

    cerr << "ReadIhf::<init> info: starting" << endl;

    filename = 0;

    gridPort = 0;
    dataPort = 0;
    fileParam = 0;

    gridDataObject = 0;
    vectorDataObject = 0;
    vectorDataName = 0;

    xDimension = 0;
    yDimension = 0;
    zDimension = 0;

    gridspacing = 0.0f;

    set_module_description("Binary file reader for IHF volume data");

    cerr << "ReadIhf::<init> info: adding ports" << endl;
    // the output port
    gridPort = addOutputPort("grid", "coDoUniformGrid", "grid data");
    dataPort = addOutputPort("data", "coDoVec3", "vector data");

    // select the OBJ file name with a file browser
    fileParam = addFileBrowserParam("data_file", "IHF File");
    fileParam->setValue("/mnt/pro/cod/ihf/binary/vector_jdataABz.bin", "*.bin");

    cerr << "ReadIhf::<init> info: done" << endl;
}

ReadIhf::~ReadIhf()
{
}

void ReadIhf::quit(void)
{
}

int ReadIhf::compute(void)
{

    // get the file name

    cerr << "ReadIhf::compute info: called" << endl;

    filename = fileParam->getValue();

    if (filename == NULL)
    {
        sendError("An input file has to be specified");
        return FAIL;
    }

#ifdef _STANDARD_C_PLUS_PLUS
    ifstream dataFile(filename, ios::in | ios::binary);
#else
    ifstream dataFile(filename, ios::in);
#endif

    if (!dataFile)
    {
        sprintf(infobuf, "Could not open file %s", filename);
        sendError(infobuf);
        return FAIL;
    }

    cerr << "ReadIhf::compute info: reading file " << filename << endl;

    dataFile.seekg(0, ios::end);
    long fileSize = dataFile.tellg();
    dataFile.seekg(0, ios::beg);

    dataFile.read(reinterpret_cast<char *>(&xDimension), 2);
    dataFile.read(reinterpret_cast<char *>(&yDimension), 2);
    dataFile.read(reinterpret_cast<char *>(&zDimension), 2);
    dataFile.read(reinterpret_cast<char *>(&gridspacing), 4);

#ifdef BYTESWAP
    byteSwap(xDimension);
    byteSwap(yDimension);
    byteSwap(zDimension);
    byteSwap(gridspacing);
#endif

    gridspacing *= 1000.0f;

    long calculatedSize = (xDimension * yDimension * zDimension) * 12 + 10;

    cerr << "ReadIhf::compute info: dimensions ["
         << xDimension << "|" << yDimension << "|"
         << zDimension << "] -> cs=" << calculatedSize
         << " fs=" << fileSize << endl;

    cerr << "ReadIhf::compute info: gridspacing is "
         << gridspacing << " mm" << endl;

    if (calculatedSize != fileSize)
    {
        sprintf(infobuf,
                "Dimensions in header don't match filesize %d <-> %d",
                calculatedSize, fileSize);
        sendError(infobuf);
        return FAIL;
    }

    gridDataObject = new coDoUniformGrid(gridPort->getObjName(),
                                         xDimension, yDimension, zDimension,
                                         0.0f, gridspacing * (float)xDimension,
                                         0.0f, gridspacing * (float)yDimension,
                                         0.0f, gridspacing * (float)zDimension);

    vectorDataObject = new coDoVec3(dataPort->getObjName(),
                                    xDimension, yDimension, zDimension);

    float xValue = 0.0;
    float yValue = 0.0;
    float zValue = 0.0;

    float *xValues = NULL;
    float *yValues = NULL;
    float *zValues = NULL;

    vectorDataObject->getAddresses(&xValues, &yValues, &zValues);

    for (int k = 0; k < zDimension; k++)
    {
        for (int j = 0; j < yDimension; j++)
        {
            for (int i = 0; i < xDimension; i++)
            {

                dataFile.read(reinterpret_cast<char *>(&xValue), 4);
                dataFile.read(reinterpret_cast<char *>(&yValue), 4);
                dataFile.read(reinterpret_cast<char *>(&zValue), 4);

#ifdef BYTESWAP
                byteSwap(xValue);
                byteSwap(yValue);
                byteSwap(zValue);
#endif

                int index = k + j * zDimension + i * zDimension * yDimension;
                //int index = i + j * xDimension + k * xDimension * yDimension;

                xValues[index] = (xValue);
                yValues[index] = (yValue);
                zValues[index] = (zValue);
            }
        }
    }

    gridPort->setCurrentObject(gridDataObject);
    dataPort->setCurrentObject(vectorDataObject);

    cerr << "ReadIhf::compute info: exit" << endl;

    return SUCCESS;
}
