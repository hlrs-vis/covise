/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef INDEXMANIFOLDS_H
#define INDEXMANIFOLDS_H
/**************************************************************************\ 
 **                                                 (C)2000 VirCinity GmbH **
 **                                                                        **
 ** Description: Extract 0-, 1- & 2-dimensional index manifolds from a     **
 **              structured grid                                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Dirk Straka                                **
 **                    VirCinity IT-Consulting GmbH                        **
 **                             Nobelstr. 30                               **
 **                            70569 Stuttgart                             **
 **                                                                        **
 ** Date:  28.10.00  V0.1                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <do/coDoTriangleStrips.h>
#include <do/coDoPolygons.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoData.h>
#include <util/coviseCompat.h>

typedef enum
{
    missingX,
    missingY,
    missingZ
} missType;
class IndexManifolds : public coSimpleModule
{

private:
    // member functions
    virtual int compute(const char *port);
    virtual void quit();
    virtual void postInst();
    void computeSurfaceSizes(int &, int &, int &);
    // all our parameter ports
    coIntSliderParam *Index[3];
    coChoiceParam *DirChoice;
    coBooleanParam *GenStrips;

    const coDistributedObject *GridObj;
    const coDoAbstractData *DataObj;

    // the data in- and output ports
    coInputPort *inPortGrid;
    coInputPort *inPortData;
    coOutputPort *outPortSurf;
    coOutputPort *outPortSurfData;
    coOutputPort *outPortLine;
    coOutputPort *outPortLineData;
    coOutputPort *outPortPoint;
    coOutputPort *outPortPointData;
    const char *outSurfName;
    const char *outSurfDataName;
    const char *outLineName;
    const char *outLineDataName;
    const char *outPointName;
    const char *outPointDataName;

    // coordinates
    float *x_coords, *y_coords, *z_coords;

    // the data
    coDoAbstractData *out_data;

    // output geometry
    int gen_strips;
    coDoPolygons *polygons;
    coDoTriangleStrips *strips_out;

    int current_strip_length;
    int current_corner;
    int num_strips;
    int *tl;

    // variables
    int size[3]; // grid size of current set element
    int maxSize[3]; // max grid size of all set elements

    // Routines
    void setSliderBounds();
    bool insert_possible(int *corners, int newcorner);

public:
    void computeSurface(const coDoUniformGrid *grid);
    void computeSurface(const coDoRectilinearGrid *grid);
    void computeSurface(const coDoStructuredGrid *grid);
    void addCorners(
        int i,
        int j,
        int sizeA,
        int sizeB,
        int *corners);
    void computeCoords(float surfaceCoord,
                       float minA,
                       float deltaA,
                       int sizeA,
                       float minB,
                       float deltaB,
                       int sizeB,
                       float *surface,
                       float *A,
                       float *B,
                       int *polygonlist,
                       int *corners);
    void computeCoords(
        float surfaceCoord,
        int sizeA,
        float *gridA,
        int sizeB,
        float *gridB,
        float *surface,
        float *polyStartA,
        float *polyStartB,
        int *polygonlist,
        int *corners);
    void computeCoords(
        const coDoStructuredGrid *grid,
        int surfaceIndex,
        int sizeA,
        int sizeB,
        float *xPolyStart,
        float *yPolyStart,
        float *zPolyStart,
        int *polygonlist,
        int *corners,
        missType missing);

    coDoPolygons *computeSurface();
    void computePoint();
    void computeLine();
    IndexManifolds(int argc, char *argv[]);
};
#endif
