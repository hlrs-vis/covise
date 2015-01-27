/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_VISTEON_H
#define _READ_VISTEON_H
/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Simple Reader for Wavefront OBJ Format	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: U.Woessner                                                     **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

#define LINELENGTH 1000

class ReadVisteon : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();
    char infobuf[1000]; // buffer for COVISE info and error messages
    char buf[LINELENGTH + 1];

    int numLines(const char *filename);
    int *polygonCoordNumbers;
    int *gridCoordNumbers;
    int numPolygonCoords;
    int numGridCoords;
    int numPolygons;
    int numElements;
    int numGrids;
    char **grids;
    char **surfaces;
    int numSurfaces;
    int tNumCoords;
    float *x_g, *y_g, *z_g;
    float *x_p, *y_p, *z_p;

    int readSurfaces();
    int readGrid();

    //void parseString(const char *string,char **&values,int &num);
    void parseCfg(const char *filename);

    bool hasTemperature; // true, if data file contains Temperature Information

    const char *crdFilename; // Coordinate file name
    const char *outFilename; // Result file name
    const char *cfgFilename; // Config file name
    char *directory; // directory, where the crd file resides
    FILE *fp;

    coOutputPort *gridPort;
    coOutputPort *gridVelocityPort;
    coOutputPort *gridPressurePort;
    coOutputPort *gridTemperaturePort;
    coOutputPort *gridVisPort;
    coOutputPort *polygonPort;
    coOutputPort *polygonPressurePort;
    coOutputPort *polygonTemperaturePort;
    coOutputPort *polygonVisPort;

    coFileBrowserParam *crdFileParam; // Coordinate file
    coFileBrowserParam *outFileParam; // Result file
    coFileBrowserParam *cfgFileParam; // Config file

    coBooleanParam *doGrid; // generate Grid, or not
    coBooleanParam *doSurface; // generate Surface, or not
    coBooleanParam *doVel; // generate Velocity, or not
    coBooleanParam *doP; // generate Pessure, or not
    coBooleanParam *doVis; // generate Viscosity, or not
    coBooleanParam *doT; // generate Temperature, or not
    //coStringParam  *Surfaces; // Surfaces to read
    //coStringParam  *Grids; // Grids to read

public:
    ReadVisteon();
    virtual ~ReadVisteon();
};
#endif
