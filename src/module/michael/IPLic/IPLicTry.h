/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / file IsoParaLic.h

#ifndef _IP_LIC_TRY_H
#define _IP LIC_TRY_H

/***************************************************************************\ 
 **                                                           (C)2001 RUS **
 **                                                                       **
 ** Description:   COVISE LineIntegralConvolution application module      **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 ** Author: M. Muench                                                     **
 **                                                                       **
 ** History:                                                              **
 ** xx. ???? 01         v1                                                **
 ** xxxxxxxx         new covise api                                       **
\***************************************************************************/

/***********************************\ 
 *                                 *
 *  place the #include files here  *
 *                                 *
\***********************************/

//#include "Carbo.h"

#include "IPLicUtilTry.h"
//#include "RungeKutta.h"
//#include "IPGeometry.h"

//textures
const int TEXTURE_LEVEL = 0;

//neighbour relationship of triangle
const int COMMON_EDGE = -1; //neighbour by common edge
const int COMMON_NODE = 1; //neighbour by one common node only

class IPLic : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();

    int doPolygons(trivec &triangles, i2ten &triNeighbours,
                   coDoPolygons **polygon, int *num_triangles);
    void doVdata(trivec &triangles, coDoVec3 **vdata,
                 int num_values);
    void doTexPoly(coDoPolygons **texPoly, trivec &triangles,
                   int width, int height, int res);
    void doTexVec(trivec &triangles);
    void printTexVec(trivec &triangles);
    void printVdata(trivec &triangles);
    void doPixelImage(int width, int height, int res, int pix,
                      coDoPixelImage **pixels);
    void doConvolution(trivec &triangles, coDoPixelImage **pixels,
                       const i2ten &triNeighbours,
                       int width, int height, int res, int pix);
    void doTexture(trivec &triangles, int pix, coDoPolygons **texPoly,
                   coDoPixelImage **pixels, coDoTexture **texture);

    //  member data

    coInputPort *polygonInPort;
    coInputPort *vectorInPort;

    coOutputPort *packageOutPort;
    coOutputPort *textureOutPort;

    //  parameters

    coIntScalarParam *resolution;
    //coIntScalarParam* filterLength;
    coIntScalarParam *pixelSize;

public:
    IPLic();
    virtual ~IPLic();
};
#endif
