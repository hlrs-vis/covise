/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_IDEA_H
#define _READ_IDEA_H
/**************************************************************************\
**                                                   	      (C)2008 HLRS  **
**                                                                        **
** Description: READ Idea (MPA)                                           **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <api/coSimpleModule.h>
using namespace covise;

#define XY 0
#define YZ 1
#define ZX 2

#define TWOD 0
#define THREED 1

class Triangulate : public coSimpleModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    //  ports
    coInputPort *p_inpoints;
    coInputPort *p_boundaries1;
    coInputPort *p_boundaries2;
    coInputPort *p_boundaries3;
    coInputPort *p_boundaries4;

    coOutputPort *p_outmesh;
    coOutputPort *p_outdata;
    coOutputPort *p_combinedPolys;

    // parameters
    coChoiceParam *p_case;
    coFloatParam *p_minAngle; // minimum angle of outer triangles for not being removed
    coChoiceParam *p_plane;

    int *get_triangles(int n, int *n_trias);
    void removeOuterEdges(int n);

    virtual void postInst();
    virtual void param(const char *paramname, bool inMapLoading);

    coDoPolygons *combine_polygons(int nCoord, float *xCoord, float *yCoord, float *zCoord);

public:
    Triangulate(int argc, char *argv[]);
    virtual ~Triangulate();
};

#endif
