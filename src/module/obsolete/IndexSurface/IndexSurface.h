/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef INDEXSURFACE_H
#define INDEXSURFACE_H
/**************************************************************************\ 
 **                                                 (C)2000 VirCinity GmbH **
 **                                                                        **
 ** Description: Extract an Index Surface from a structured grid           **
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
#include <util/coviseCompat.h>

typedef enum
{
    missingX,
    missingY,
    missingZ
} missType;
class IndexSurface : public coSimpleModule
{

private:
    // member functions
    virtual int compute(const char *port);
    virtual void quit();
    virtual void param(const char *name);
    virtual void postInst();
    void computeSizes(int &, int &, int &);
    // all our parameter ports
    coIntSliderParam *Index;
    coChoiceParam *DirChoice;
    coBooleanParam *GenStrips;

    coDistributedObject *GridObj;
    coDistributedObject *DataObj;

    // the data in- and output ports
    coInputPort *inPortGrid;
    coInputPort *inPortData;
    coOutputPort *outPortSurf;
    coOutputPort *outPortSurfData;
    const char *outSurfName;
    const char *outSurfDataName;

    // coordinates
    float *x_coords, *y_coords, *z_coords;

    // the data
    coDoFloat *obj_sdata_out;
    float *sdata;
    coDoVec3 *obj_vdata_out;
    float *xdata, *ydata, *zdata;

    // output geometry
    int gen_strips;
    coDoPolygons *polygons;
    coDoTriangleStrips *strips_out;

    int current_strip_length;
    int current_corner;
    int num_strips;
    int *tl;

    // variables
    int size, sizei, sizek, sizej;
    int lasti, lastj, lastk;

    // Routines
    virtual int set_index_Value(const bool oldval);
    bool insert_possible(int *corners, int newcorner);

public:
    void computeSurface(coDoUniformGrid *grid);
    void computeSurface(coDoRectilinearGrid *grid);
    void computeSurface(coDoStructuredGrid *grid);
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
        coDoStructuredGrid *grid,
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
    IndexSurface();
};
#endif // INDEXSURFACE_H
