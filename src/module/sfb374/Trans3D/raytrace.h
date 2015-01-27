/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          raytrace.h  -  ray tracing
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __RAYTRACE_H_

#define __RAYTRACE_H_

#include "arrays.h"
#include "classext.h"

#define MAXPREV 20 // maximum number of tracked previous rays

extern Matrix<point> AdjPatches;
extern imatrix FaceEdges;
extern Vector<point> ptCorners;
extern Vector<point> ptPrev;

// ***************************************************************************
// ray tracing class
// ***************************************************************************

class RayTrace
{
public:
    RayTrace(void)
    {
        edge.ReSize(8);
        bCheckPrev = true;
        iBadRays = 0;
    }

    void TraceSolid(int, int); // raytracing for solid
    void TraceGas(int, int); // multiple reflections tracing

    void CalcPatchVect(int, int, // calc sub-patch position
                       const TPoint &);
    void CalcPatchArea(int, int, // calc sub-patch area
                       const TPoint &);
    void CalcFaceVect(int, TPoint, // calc vector on face of cell
                      TPoint3D &);
    void CalcPatchIntersection(int, int, // calc intersection
                               TPoint &, int &); // with patch
    void CalcFaceIntersection(int, TPoint &, // calc intersection
                              int &); // with face
    void CalcLineIntersection(int, int, int &, // calc intersection with
                              int &, int &, int); // line
    prec CalcUFromV(prec); // calc u of intersection for given v
    void CalcEdges(int, int, int); // calc edges of volume element
    // distance pt to RayIn
    void CalcDistance(int, int, TPoint, prec &);
    void FindNextIntersection(int &, int &, // find next intersection
                              TPoint &, int &);

    int iRay; // ray number
    int iBadRays; // number of bad rays
    int iRefNum; // number of reflections

    bool bCheckPrev;

    prec ndArea; // area of sub-patch
    prec ndFlux; // incident flux for solid
    prec ndRefFlux; // reflected flux for gas

    prec dirxx, dirxy, dirxz, // coefficents of eqn.71b
        diryy, diryz;
    prec inposdir; // x*dir
    prec incos; // norm*dir = cos angle of incidence
    prec insin2; // sin(theta)^2 = 1-(norm*dir)^2
    prec a11, a12, a21, a22; // aij of eqn(73)
    prec b11, b12, b21, b22; // bij of eqn(73)
    prec delta_uv16th; // 16th of sub-patch interval Du x Dv

    TPoint3D RayIn; // incident position
    TPoint3D RayOut; // outgoing position
    TPoint3D RayOld; // old ray position for debug uses
    TPoint3D InDir; // incident direction
    TPoint3D OutDir; // outgoing direction
    TPoint3D Normal; // surface normal
    TPoint3D InNormal; // normal to plane of incidence
    TCPoint3D EField; // electric field vector
    Vector<TPoint3D> edge; // edge coordatines of element
};
#endif
