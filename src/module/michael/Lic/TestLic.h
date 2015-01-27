/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / file Lic.h

#ifndef _TEST_LIC_H
#define _TEST LIC_H

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

#include "LicUtil.h"
#include "RungeKutta.h"
#include "Geometry.h"

//textures
const int TEXTURE_LEVEL = 0;

//neighbour relationship of triangle
const int COMMON_EDGE = -1; //neighbour by common edge
const int COMMON_NODE = 1; //neighbour by one common node only

class Lic : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();

    void doPolygons(Patch **triPatch, trivec &triangles, coDoPolygons **polygon,
                    int *num_triangles, float sq);
    void doPixelImage(int width, int height, int pix, coDoPixelImage **pixels);
    void doConvolution(Patch **triPatch, trivec &triangles, coDoPolygons **polygon,
                       coDoVec3 *vdata, coDoPixelImage **pixels);
    void doTexture(int dimension, int pix, coDoPolygons **polygons,
                   coDoPixelImage **pixels, coDoTexture **texture);

    //  member data

    coInputPort *polygonInPort;
    //coInputPort* vectorInPort;

    coOutputPort *polygonOutPort;
    coOutputPort *packageOutPort;
    coOutputPort *textureOutPort;

    //  parameters

    //coIntScalarParam* resolution;
    coIntScalarParam *pixImgWidth;
    coIntScalarParam *pixImgHeight;
    coIntScalarParam *pixelSize;
    //coFloatParam* domainSize;
    coFloatParam *scaleQuad;

public:
    Lic();
    virtual ~Lic();
};
#endif
