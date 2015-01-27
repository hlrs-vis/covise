/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2005 GER  **
 **                                                                        **
 ** Description: Read ASCII FIles.							                     **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     	 Oliver Krause			                           **
 **     			Zentrum fuer angewandte Informatik Koeln				         **
 **                     University  of Cologne						            **
 **                                                   					      **
 **                                                                        **
 ** Cration Date: May 2005                                                 **
\**************************************************************************/

#ifndef _READ_HEIGHTFIELD_H_
#define _READ_HEIGHTFIELD_H_

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

struct params
{
    float *xcoords;
    float *ycoords;
    float *zcoords;

    float *xdata;
    float *ydata;
    float *zdata;
    float *scalardata;
};

bool strContains(const string &, const string &);
void expandPattern(string &pattern, int bufferSize);

class ReadASCII : public coSimpleModule
{
    COMODULE

private:
    // Ports:
    coOutputPort *poGeometry;
    coOutputPort *poScalar;
    coOutputPort *poVector;

    // Parameters:
    coFileBrowserParam *pbrFile; // name of first file of a sequence
    coBooleanParam *pHeader; // does an header exist?

    coIntScalarParam *pBufferSize; // Size of the read buffer
    coChoiceParam *pGeom;
    //coStringParam*       pSedPattern;
    coIntScalarParam *pHeaderByteOffset;
    coIntScalarParam *pHeaderLineSkip;

    coStringParam *pDimPattern;
    coIntScalarParam *pDimX; // dim X
    coFloatParam *pUniDistX; // uniform distance in X direction
    coIntScalarParam *pDimY; // dim Y
    coFloatParam *pUniDistY; // uniform distance in Y direction
    coIntScalarParam *pDimZ; // dim Z
    coFloatParam *pUniDistZ; // uniform distance in Z direction

    coBooleanParam *pInterl; // reading the data (record, planar)

    coIntScalarParam *pPointsNum;
    coIntScalarParam *pDataByteOffset;
    coIntScalarParam *pDataLineOffset;
    coStringParam *pDataFormat;
    coChoiceParam *pCoordSequence;
    coBooleanParam *pOutputResult;
    coBooleanParam *pOutputDebug;

    coFloatParam *pScale; // Scale factor for data

    // Methods:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    bool readDataFromFile(params para);
    void printDebug(void);
    void printResult(params para);

public:
    ReadASCII(int argc, char **argv);
    virtual ~ReadASCII();
};

#endif
