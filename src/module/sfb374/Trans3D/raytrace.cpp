/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          raytrace.cpp  -  ray tracing
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "raytrace.h"
#include "solid.h"
#include "grid.h"
#include "material.h"
#include "simul.h"
#include "laser.h"
#include "fortran.h"
#include "main.h"

#define QBNDLFAC 1.0e-3 // additional factor for neglectable power

namespace Trace
{
enum Edges
{
    ipjmkp = 0,
    ipjpkp = 1,
    imjpkp = 2,
    imjmkp = 3,
    ipjmkm = 4,
    ipjpkm = 5,
    imjpkm = 6,
    imjmkm = 7 // volume element edge definition
};
enum Corners // edges of surface patch
{
    ipj = 0,
    ipjp = 1,
    ijp = 2,
    ij = 3
};
enum Lines // boundary lines of surface patch
{
    ipconst = 0,
    jpconst = 1,
    iconst = 2,
    jconst = 3
};
enum Faces // faces of volume element
{
    upperk = 0,
    loweri = 1,
    upperi = 2,
    lowerj = 3,
    upperj = 4,
    lowerk = 5
};
enum Hit
{
    no_intersection = 0,
    intersection = 1,
    next_element = 2,
    passes_edge = 3,
    leaves_groove = 4 // intersection status
};
enum LineFlag
{
    loop_over = 0,
    neighbor_line = 1,
    opposite_line = 2,
    last_line = 3 // flag for checked line
};
enum NextElem
{
    diagonal_left = 0,
    straight = 1,
    diagonal_right = 2 // next element marker
};
}

Matrix<point> AdjPatches(4, 3); // 3 adjacent patches of boundary line
imatrix FaceEdges(6, 4); // edge number of 4 face corners
Vector<point> ptCorners(4); // indices of 4 corner points of patch
Vector<point> ptPrev(MAXPREV); // hit points of previous rays

// ***************************************************************************
// raytracing for absorbed rays
// ray with strength ndFlux entering from RayIn on element (i,j) with
// direction InDir (local surface normal: Normal)
//
// input:   element (i,j), RayIn, Normal, ndFlux, iRay
// output:  Solid.ndHeat, iRay, ndFlux, Normal, inposdir, RayIn, InDir
//
// ***************************************************************************

void RayTrace::TraceSolid(int i, int j)
{
    int ie, je, ke, ihit, iInFace, iOutFace;
    prec abscoeff, transfrac, costheta, tmp, outposdir,
        dist, reffrac;
    cmplx index;
    TPoint pt;

    abscoeff = Material.GetAbsIndex(Solid(i, j, 1).ndTemp, index);
    // estimate of transmitted fraction after after cell
    transfrac = ndFlux * exp(-0.5 * abscoeff * (Solid(i, j, 2).ndNode - Solid(i, j, 1).ndNode).Abs());
    if (transfrac < Simulation.ndMinPower * QBNDLFAC || (!Simulation.bHeatConduct && Solid(i, j, 1).ndTemp >= 1))
    {
        Solid(i, j, 1).ndHeat += ndFlux; // all deposited in first element
        return;
    }
    if (RayIn.z > Grid.ndWpThickness) // beyond workpiece thickness
    {
        SolSurf.ndTotalTrans += ndFlux; // all transmitted
        return;
    }
    RayOld = RayIn - InDir * (Grid.ndDepth / 10); // artificial origin of ray

    ie = i; // indices for current element
    je = j;
    ke = 1;
    if (Simulation.iReflection != TSimulation::constant)
    { // calc entering diection for Fresnel case
        costheta = InDir * Normal; // cosine of incident angle
        tmp = sqrt(sqr(index.real()) + sqr(costheta) - 1.0) - costheta;
        InDir = (InDir + tmp * Normal) / index.real(); // new direction
    }
    dirxx = 1.0 - sqr(InDir.x); // coefficients eqn. (71b/c);
    dirxy = InDir.x * InDir.y;
    dirxz = InDir.x * InDir.z;
    diryy = 1.0 - sqr(InDir.y);
    diryz = InDir.y * InDir.z;
    *pdebug << "ray no. " << iRay << " is entering surface element ("
            << ie << ',' << je << ')' << endl
            << "point of penetration:  " << pPT3D(RayIn) << endl
            << "penetration direction: " << pPT3D(InDir) << endl
            << "strength: " << ndFlux << endl;

    CalcEdges(ie, je, ke); // generate 8 edges of volume element
    inposdir = RayIn * InDir; // scalar product position*direction

    CalcFaceIntersection(Trace::lowerk, pt, ihit); // intersection with top face
    if (ihit != Trace::intersection)
    {
        pt.x = min(max(pt.x, -0.99999), 0.99999); // limit to element
        pt.y = min(max(pt.y, -0.99999), 0.99999);
    }
    CalcFaceVect(Trace::lowerk, pt, RayIn); // corrected incident position
    inposdir = RayIn * InDir;
    *pdebug << "penetration point: " << pPT3D(RayIn) << endl;

    iInFace = Trace::lowerk; // entered face
    do // search intersection rayout with outgoing ray
    {
        if (RayIn.z >= Grid.ndWpThickness)
        {
            SolSurf.ndTotalTrans += ndFlux; // rest transmitted
            return;
        }
        for (i = 0; i < 6; i++) // check for intersection with faces
        {
            iOutFace = i;
            if (iOutFace >= iInFace)
                iOutFace = i + 1;
            if (i == 5) // test incoming face last
                iOutFace = iInFace;

            *pdebug << "checking face " << iOutFace << endl;
            CalcFaceIntersection(iOutFace, pt, ihit);
            if (ihit != Trace::no_intersection) // intersection found
                break;
        }
        if (ihit == Trace::no_intersection) // no intersection found
        {
            *pdebug << "no intersection found in element ("
                    << ie << ',' << je << ',' << ke << ')' << endl;
            unitwriteln(9, "This point should not be reached; discarding ray");
            return;
        }

        CalcFaceVect(iOutFace, pt, RayOut); // outgoing position vector

        if (ihit == Trace::passes_edge) // correct direction if ray goes
        { // through edge
            InDir = RayOut - RayIn;
            InDir.Normalize();
            inposdir = RayIn * InDir;
            dirxx = 1.0 - sqr(InDir.x); // coefficients eqn. (71b/c);
            dirxy = InDir.x * InDir.y;
            dirxz = InDir.x * InDir.z;
            diryy = 1.0 - sqr(InDir.y);
            diryz = InDir.y * InDir.z;
            *pdebug << "corrected direction: " << pPT3D(InDir) << endl;
        }

        outposdir = RayOut * InDir;
        dist = outposdir - inposdir; // traveled distance
        inposdir = outposdir;
        if (dist < -1.0e-6)
        {
            *pdebug << "negative distance " << dist << " in element ("
                    << ie << ',' << je << ',' << ke << ')' << endl;
            return;
        }

        abscoeff = Material.GetAbsIndex(Solid(ie, je, ke).ndTemp, index);
        transfrac = exp(-abscoeff * dist); // transmitted fraction
        // absorbed power
        Solid(ie, je, ke).ndHeat += ndFlux * (1.0 - transfrac);
        *pdebug << "leaving element across face " << iOutFace
                << " at location " << pPT3D(RayOut) << endl
                << "distance in element: " << dist << " absorbed: "
                << ndFlux * (1.0 - transfrac) << endl;
        ndFlux *= transfrac; // transmitted power

        switch (iOutFace) // move on to adjacent element
        {
        default:
            *pdebug << "lost ray within element (" << ie << ',' << je << ','
                    << ke << ')' << endl;
            return;

        case Trace::upperk: // upper k=const surface
            ke++;
            *pdebug << "ray enters volume element (" << ie << ',' << je << ','
                    << ke << ')' << endl;
            if (ke == Grid.kVolNodes) // last element reached
            {
                SolSurf.ndTotalTrans += ndFlux; // rest transmitted
                WarningFunction("WARNING: depth chosen too small;"
                                " (beam penetrating into substrate)");
                *pdebug << "beam penetrating into substrate; discarding ray" << endl;
                return;
            }
            iInFace = Trace::lowerk; // lower surface of next element
            CalcEdges(ie, je, ke);
            break;

        case Trace::loweri: // lower i=const surface
            if (ie == 1) // boundary reached
            {
                *pdebug << "ray reflected at x=wplength" << endl;
                if (Grid.iFrontBound == TGrid::finite) // partially reflected
                { // if finite volume
                    // to do: check if normal Ok
                    TPoint3D xdir(1, 0, 0);
                    reffrac = Material.GetReflection(InDir, OutDir, xdir,
                                                     EField, Solid(ie, je, ke).ndTemp);
                    *pdebug << "exiting flux: " << ndFlux *(1 - reffrac) << endl;
                    ndFlux *= reffrac;
                }
                // reflecting ray at
                RayIn.x = 2.0 * Grid.ndWpLength - RayIn.x;
                InDir.x = -InDir.x; // x=ndWpLength
                dirxy = -dirxy;
                dirxz = -dirxz;
                *pdebug << "corrected direction: " << pPT3D(InDir) << endl;
                iInFace = Trace::loweri; // entering through same surface
                break;
            }
            ie--; // entering previous element
            *pdebug << "ray enters volume element (" << ie << ',' << je << ','
                    << ke << ')' << endl;
            iInFace = Trace::upperi;
            CalcEdges(ie, je, ke);
            break;

        case Trace::upperi: // upper i=const surface
            if (ie == Grid.iVolNodes) // boundary reached
            {
                *pdebug << "ray reflected at x=0" << endl;
                if (Grid.iBackBound == TGrid::finite) // partially reflected
                { // if finite volume
                    TPoint3D negxdir(-1, 0, 0);
                    reffrac = Material.GetReflection(InDir, OutDir, negxdir,
                                                     EField, Solid(ie, je, ke).ndTemp);
                    *pdebug << "exiting flux: " << ndFlux *(1 - reffrac) << endl;
                    ndFlux *= reffrac;
                }
                RayIn.x = -RayIn.x; // reflecting ray at x=0
                InDir.x = -InDir.x;
                dirxy = -dirxy;
                dirxz = -dirxz;
                *pdebug << "corrected direction: " << pPT3D(InDir) << endl;
                iInFace = Trace::upperi; // entering through same surface
                break;
            }
            ie++; // entering next element
            *pdebug << "ray enters volume element (" << ie << ',' << je << ','
                    << ke << ')' << endl;
            iInFace = Trace::loweri;
            CalcEdges(ie, je, ke);
            break;

        case Trace::lowerj: // lower j=const surface
            if (je == 1) // boundary reached
            {
                *pdebug << "ray reflected at y=0" << endl;
                // partially reflected
                if (Grid.iRightBound == TGrid::symmetric)
                { // if finite volume
                    TPoint3D negydir(0, -1, 0);
                    reffrac = Material.GetReflection(InDir, OutDir, negydir,
                                                     EField, Solid(ie, je, ke).ndTemp);
                    *pdebug << "exiting flux: " << ndFlux *(1 - reffrac) << endl;
                    ndFlux *= reffrac;
                }
                RayIn.y = -RayIn.y; // reflecting ray at y=0
                InDir.y = -InDir.y;
                dirxy = -dirxy;
                diryz = -diryz;
                *pdebug << "corrected direction: " << pPT3D(InDir) << endl;
                iInFace = Trace::lowerj; // entering through same surface
                break;
            }
            je--; // entering previous element
            *pdebug << "ray enters volume element (" << ie << ',' << je << ','
                    << ke << ')' << endl;
            iInFace = Trace::upperj;
            CalcEdges(ie, je, ke);
            break;

        case Trace::upperj: // upper j=const surface
            if (je == Grid.jVolNodes) // boundary reached
            {
                *pdebug << "ray reflected at y=wpwidth" << endl;
                // partially reflected
                if (Grid.iLeftBound == TGrid::symmetric)
                { // if finite volume
                    TPoint3D ydir(0, 1, 0);
                    reffrac = Material.GetReflection(InDir, OutDir, ydir,
                                                     EField, Solid(ie, je, ke).ndTemp);
                    *pdebug << "exiting flux: " << ndFlux *(1 - reffrac) << endl;
                    ndFlux *= reffrac;
                }
                RayIn.y = 2.0 * Grid.ndWpWidth - RayIn.y;
                InDir.y = -InDir.y; // reflecting ray at y=ndWpWidth
                dirxy = -dirxy;
                diryz = -diryz;
                *pdebug << "corrected direction: " << pPT3D(InDir) << endl;
                iInFace = Trace::upperj; // entering through same surface
                break;
            }
            je++; // entering next element
            *pdebug << "ray enters volume element (" << ie << ',' << je << ','
                    << ke << ')' << endl;
            iInFace = Trace::lowerj;
            CalcEdges(ie, je, ke);
            break;

        case Trace::lowerk: // lower k=const surface
            if (ke == 1) // top boundary reached
            {
                *pdebug << "ray reflected at z=0" << endl;
                TPoint3D negnormal = -SolSurf(ie, je).Normal;
                reffrac = Material.GetReflection(InDir, OutDir,
                                                 negnormal, EField,
                                                 Solid(ie, je, ke).ndTemp);
                *pdebug << "exiting flux: " << ndFlux *(1 - reffrac) << endl;
                if (reffrac > 1.0) // surface from rear
                {
                    *pdebug << "hitting surface from rear; discarding ray" << endl;
                    return;
                }
                ndFlux *= reffrac;
                InDir = OutDir;
                *pdebug << "corrected direction: " << pPT3D(InDir) << endl;
                dirxx = 1.0 - sqr(InDir.x); // coefficients eqn. (71b/c);
                dirxy = InDir.x * InDir.y;
                dirxz = InDir.x * InDir.z;
                diryy = 1.0 - sqr(InDir.y);
                diryz = InDir.y * InDir.z;
                iInFace = Trace::lowerk; // entering through same surface
                break;
            }
            ke--; // entering previous element
            *pdebug << "ray enters volume element (" << ie << ',' << je << ','
                    << ke << ')' << endl;
            iInFace = Trace::upperk;
            CalcEdges(ie, je, ke);
            break;
        }
    } while (ndFlux >= Simulation.ndMinPower * QBNDLFAC); // energy not depleted

    *pdebug << "ray energy depleted (flux=" << ndFlux << "<minflux="
            << Simulation.ndMinPower *QBNDLFAC << ')' << endl;
    Solid(ie, je, ke).ndHeat += ndFlux;
}

// ***************************************************************************
// calculate coordinates r(pt) on face nf of cell
// ***************************************************************************

void RayTrace::CalcFaceVect(int nf, TPoint pt, TPoint3D &r)
{
    prec an12, an22, an23, an33, an11, an21, an31, an41;

    an12 = 0.25 * (1.0 - pt.y); // interpolation between corner vectors
    an22 = 0.25 * (1.0 + pt.y);
    an23 = 1.0 + pt.x;
    an33 = 1.0 - pt.x;
    an11 = an23 * an12;
    an21 = an23 * an22;
    an31 = an33 * an22;
    an41 = an33 * an12;
    r = an11 * edge(FaceEdges(nf, 0)) + an21 * edge(FaceEdges(nf, 1)) + an31 * edge(FaceEdges(nf, 2)) + an41 * edge(FaceEdges(nf, 3));
    return;
}

// ***************************************************************************
// calculate intersection (u,v) of ray with surface mf
// ihit = hit flag
// ***************************************************************************

void RayTrace::CalcFaceIntersection(int mf, TPoint &pt, int &ihit)
{
    const prec eps = 1e-8;
    int m1, m2, m3, m4, i = 0, j = 0;
    prec ae1, ae2, ae3, ae4, be1, be2, be3, be4,
        ga1, ga2, ga3, rt, ft, srt, v1m, v2m, u1m, u2m, d1, d2;
    TPoint pt1, pt2;

    ihit = Trace::no_intersection;
    pt.x = 1e10;
    m1 = FaceEdges(mf, 0); // four edges of face
    m2 = FaceEdges(mf, 1);
    m3 = FaceEdges(mf, 2);
    m4 = FaceEdges(mf, 3);
    // a eqn(71b)
    ae1 = dirxx * edge(m1).x - dirxy * edge(m1).y - dirxz * edge(m1).z;
    ae2 = dirxx * edge(m2).x - dirxy * edge(m2).y - dirxz * edge(m2).z;
    ae3 = dirxx * edge(m3).x - dirxy * edge(m3).y - dirxz * edge(m3).z;
    ae4 = dirxx * edge(m4).x - dirxy * edge(m4).y - dirxz * edge(m4).z;
    // b eqn(71c)
    be1 = -dirxy * edge(m1).x + diryy * edge(m1).y - diryz * edge(m1).z;
    be2 = -dirxy * edge(m2).x + diryy * edge(m2).y - diryz * edge(m2).z;
    be3 = -dirxy * edge(m3).x + diryy * edge(m3).y - diryz * edge(m3).z;
    be4 = -dirxy * edge(m4).x + diryy * edge(m4).y - diryz * edge(m4).z;
    a11 = ae4 - ae3 + ae2 - ae1; // aij of eqn(73)
    a12 = -ae4 - ae3 + ae2 + ae1;
    a21 = be4 - be3 + be2 - be1;
    a22 = -be4 - be3 + be2 + be1;
    b11 = -ae4 + ae3 + ae2 - ae1; // bij of eqn(73)
    b12 = ae4 + ae3 + ae2 + ae1 - 4.0 * (RayIn.x - inposdir * InDir.x);
    b21 = -be4 + be3 + be2 - be1;
    b22 = be4 + be3 + be2 + be1 - 4.0 * (RayIn.y - inposdir * InDir.y);
    ga1 = a11 * b21 - b11 * a21; // gammai in eqn(75)
    ga2 = 0.5 * (a12 * b21 + a11 * b22 - b12 * a21 - b11 * a22);
    ga3 = a12 * b22 - b12 * a22;

    if ((a11 == 0 && a12 == 0) || (a21 == 0 && a22 == 0) || (fabs(ga1) <= 1e-15))
    {
        if (a11 == 0 && a12 == 0) // special treatment for rays parallel to side wall
        {
            if (b11 == 0)
                return;
            pt.y = -b12 / b11;
            pt.x = CalcUFromV(pt.y); // single root
        }
        else if (a21 == 0 && a22 == 0)
        {
            if (b21 == 0)
                return;
            pt.y = -b22 / b21;
            pt.x = CalcUFromV(pt.y); // single root
        }
        else
        {
            pt.y = -ga3 / (2.0 * ga2);
            pt.x = CalcUFromV(pt.y);
        }
        if (fabs(pt.x) <= 1 - eps && fabs(pt.y) <= 1 - eps)
            ihit = Trace::intersection;
    }
    else // two roots
    {
        rt = ga1 * ga3 / (ga2 * ga2);
        if (rt > 1.0) // discard if imaginary
        {
            *pdebug << "only complex roots found" << endl;
            return;
        }
        ft = -ga2 / ga1;
        if (fabs(rt) > 1e-10)
        {
            srt = ft * sqrt(1.0 - rt);
            pt1.y = ft + srt;
            pt2.y = ft - srt;
        }
        else
        {
            pt1.y = 0.5 * ft * rt;
            pt2.y = 2.0 * ft - pt1.y;
        }
        pt1.x = CalcUFromV(pt1.y);
        pt2.x = CalcUFromV(pt2.y);
        *pdebug << "two possible intersections: " << pPT(pt1) << " and "
                << pPT(pt2) << endl;
        v1m = fabs(pt1.y) - 1.0; // check which root is on surface, if any
        v2m = fabs(pt2.y) - 1.0;
        if (v1m < eps && v2m < eps) // both v values inside patch?
        {
            pt1.x = CalcUFromV(pt1.y);
            u1m = fabs(pt1.x) - 1.0;
            pt2.x = CalcUFromV(pt2.y);
            u2m = fabs(pt2.x) - 1.0;
            if (u1m < eps)
            {
                if (u2m < eps) // both on surface
                {
                    CalcDistance(i, j, pt1, d1); // determine distance of ray
                    CalcDistance(i, j, pt1, d2); //  to intersections
                    if (d1 < d2 && d1 > Grid.ndSpacing) // take closer one
                        pt = pt1;
                    else
                        pt = pt2;
                }
                else // only pt1 on surface
                    pt = pt1;
                ihit = Trace::intersection;
            }
            else if (u2m < eps) // only pt2 on surface
            {
                pt = pt2;
                ihit = Trace::intersection;
            }
            else
                return; // no root on surface
        }
        else if (v1m < eps) // only v1 inside
        {
            pt.y = pt1.y;
            pt.x = CalcUFromV(pt.y);
            u1m = fabs(pt.x) - 1.0;
            if (u1m < eps) // also u1 inside?
                ihit = Trace::intersection;
        }
        else if (v2m < eps) // only v2 inside
        {
            pt.y = pt2.y;
            pt.x = CalcUFromV(pt.y);
            u2m = fabs(pt.x) - 1.0;
            if (u2m < eps) // also u1 inside?
                ihit = Trace::intersection;
        }
        else if (v1m < v2m) // no root on surface; take closest
            pt.y = pt1.y;
        else
            pt.y = pt2.y;
        pt.x = CalcUFromV(pt.y);
        return;
    }
    if (ihit == Trace::no_intersection)
        return;

    if (pt.x > 1.0 - eps) // u close to corner
    {
        ihit = Trace::passes_edge;
        pt.x = 1.0 - eps;
    }
    else if (pt.x < -1.0 + eps)
    {
        ihit = Trace::passes_edge;
        pt.x = -1.0 + eps;
    }
    else if (pt.y > 1.0 - eps) // v close to corner
    {
        ihit = Trace::passes_edge;
        pt.y = 1.0 - eps;
    }
    else if (pt.y < -1.0 + eps)
    {
        ihit = Trace::passes_edge;
        pt.y = -1.0 + eps;
    }
    return;
}

// ***************************************************************************
// calculate intersection pt(u,v) of ray with patch (i,j)
// ihit = hit flag
// ***************************************************************************

void RayTrace::CalcPatchIntersection(int i, int j, TPoint &pt, int &ihit)
{
    inposdir = RayIn * InDir;
    dirxx = 1.0 - sqr(InDir.x); // coefficients eqn. (71b/c);
    dirxy = InDir.x * InDir.y;
    dirxz = InDir.x * InDir.z;
    diryy = 1.0 - sqr(InDir.y);
    diryz = InDir.y * InDir.z;
    edge(7) = Solid(i + 1, j, 1).ndNode;
    edge(6) = Solid(i + 1, j + 1, 1).ndNode;
    edge(5) = Solid(i, j + 1, 1).ndNode;
    edge(4) = Solid(i, j, 1).ndNode;
    CalcFaceIntersection(Trace::lowerk, pt, ihit);
}

// ***************************************************************************
// calculate coordinates edge() of 8 corners in element (i,j,k)
// ***************************************************************************

void RayTrace::CalcEdges(int i, int j, int k)
{
    int m, i2, j2, k2;
    TPoint3D pt;

    for (m = 0; m < 8; m++) // loop over 8 edges
    {
        k2 = max(k + 1 - ((m & 4) >> 1), 1); // k of opposite element
        // j of opposite element
        j2 = min(max(j - 1 + ((m + 1) & 2), 1), Grid.jVolNodes);
        i2 = min(max(i + 1 - (m & 2), 1), Grid.iVolNodes); // i of opposite element

        // average
        edge(m) = 0.125 * (Solid(i, j, k).ndNode + Solid(i, j, k2).ndNode + Solid(i2, j, k).ndNode + Solid(i2, j, k2).ndNode + Solid(i, j2, k).ndNode + Solid(i, j2, k2).ndNode + Solid(i2, j2, k).ndNode + Solid(i2, j2, k2).ndNode);
    }
}

// ***************************************************************************
// calculate u component of intersection for given v
// ***************************************************************************

prec RayTrace::CalcUFromV(prec v)
{
    prec ap1, ap2;

    ap1 = a11 * v + a12;
    ap2 = a21 * v + a22;
    if (fabs(ap1) <= fabs(ap2))
        return -(b21 * v + b22) / ap2;
    return -(b11 * v + b12) / ap1;
}

// ***************************************************************************
// calculate distance d between ray at RayIn with direction InDir
// and intersection pt on element (i,j)
// ***************************************************************************

void RayTrace::CalcDistance(int i, int j, TPoint pt, prec &d)
{
    prec an11, an21, an31, an41;
    TPoint3D ptPatch;

    an11 = (1.0 + pt.x) * (1.0 - pt.y); // calc vector ptPatch(u,v) on surface
    an21 = (1.0 + pt.x) * (1.0 + pt.y);
    an31 = (1.0 - pt.x) * (1.0 + pt.y);
    an41 = (1.0 - pt.x) * (1.0 - pt.y);

    ptPatch = an11 * Solid(i + 1, j, 1).ndNode + an21 * Solid(i + 1, j + 1, 1).ndNode + an31 * Solid(i, j + 1, 1).ndNode + an41 * Solid(i, j, 1).ndNode;
    d = fabs(ptPatch * InDir - inposdir);
    return;
}

// ***************************************************************************
// find next intersection of ray leaving element(ie,je) with surface
//
// input:   element (ie,je), ptPrev, RayIn, InDir, InNormal
// output:  element (ie,je), intersection position pt, hit flag ihit,
//          ptPrev, RayIn, InDir, RayOut, OutDir, InNormal
//
// ***************************************************************************

void RayTrace::FindNextIntersection(int &ie, int &je, TPoint &pt, int &ihit)
{
    int i, j, l, iTracedPatches, nextelem, iNumLines;
    prec tmp;

    i = ie;
    j = je;
    ihit = Trace::no_intersection;
    if (bCheckPrev) // try reflection of previous ray
    {
        iRefNum++; // next reflection
        if (iRefNum >= MAXPREV - 1) // check for hit with neighboring
        { // element of last rays
            iBadRays++; // too many reflections
            unitwriteln(9, "%i %i TOO MANY REFLECTIONS", iRay, iBadRays);
            return;
        }
        if (Simulation.bSpecReflec && ptPrev(iRefNum).x != 0 && ptPrev(iRefNum).x != ie && ptPrev(iRefNum).y != je)
        {
            unitwriteln(9, "checking for intersection with element (%2i,%2i)",
                        ptPrev(iRefNum).x, ptPrev(iRefNum).y);
            CalcPatchIntersection(ptPrev(iRefNum).x, ptPrev(iRefNum).y, pt, ihit);
            if (fabs(pt.y) <= 2.0 && fabs(pt.y) <= 2.0) // close to previous
            {
                if (ihit == Trace::no_intersection) // try neighbor
                {
                    ptPrev(iRefNum).x += int((pt.x + 1.0) / 2.0);
                    ptPrev(iRefNum).y += int((pt.y + 1.0) / 2.0);
                    // same patch?
                    if (ptPrev(iRefNum).x != ie && ptPrev(iRefNum).y != je)
                    {
                        *pdebug << "checking for intersection with element ("
                                << ptPrev(iRefNum).x << ',' << ptPrev(iRefNum).y
                                << ')' << endl;
                        CalcPatchIntersection(ptPrev(iRefNum).x, ptPrev(iRefNum).y,
                                              pt, ihit);
                        if (ihit == Trace::intersection)
                        {
                            ie = ptPrev(iRefNum).x; // intersection found
                            je = ptPrev(iRefNum).y;
                            return;
                        }
                    }
                }
                else
                {
                    ie = ptPrev(iRefNum).x; // intersection found
                    je = ptPrev(iRefNum).y;
                    return;
                }
            }
        }
        ptPrev(iRefNum) = 0;
        ptPrev(iRefNum + 1) = 0;
    }
    iTracedPatches = 0; // number of patches crossed by ray
    iNumLines = 0; // number of lines checked within element
    while (iNumLines <= 3) // loop over boundary lines
    {
        if (iTracedPatches == 0) // initially loop over 4 lines
            l = iNumLines;
        CalcLineIntersection(ie, je, l, ihit, nextelem, iNumLines);
        if (ihit == Trace::intersection) // hits surface
        {
            if (i == ie && j == je) // hitting emitting patch again
            {
                *pdebug << "hitting itself (probably bad normal); discarding ray"
                        << endl;
                ihit = Trace::no_intersection;
                return;
            }
            CalcPatchIntersection(ie, je, pt, ihit); // intersection w patch
            if (ihit != Trace::intersection)
            {
                iBadRays++;
                *pdebug << "couldn't determine intersection with patch" << endl;
                return;
            }
            pt.x = min(max(-0.99999, pt.x), 0.99999); // limit result to patch
            pt.y = min(max(-0.99999, pt.y), 0.99999);
            return;
        }
        if (ihit == Trace::leaves_groove) // leaves geometry
        {
            RayOut.z = -Grid.ndDepth / 10; // artificial ray end
            tmp = (RayOut.z - RayIn.z) / InDir.z;
            RayOut.x = RayIn.x + tmp * InDir.x;
            RayOut.y = RayIn.y + tmp * InDir.y;
            *pdebug << "line to: " << pPT3D(RayOut) << endl;
            return;
        }
        if (ihit != Trace::next_element)
        {
            iNumLines++; // try next line
            continue;
        }
        ie += AdjPatches(l, nextelem).x; // passed to next patch
        je += AdjPatches(l, nextelem).y;
        iTracedPatches++;
        if (iTracedPatches >= Grid.iVolNodes + Grid.jVolNodes)
        {
            iBadRays++;
            *pdebug << "too many traces for ray " << iRay << endl;
            return;
        }
        iNumLines = 1; // entered line can be omitted
        *pdebug << "ray enters element (" << ie << ',' << je << ')' << endl;
        if (ie < 1) // lower i boundary
        {
            tmp = (Grid.ndWpLength - RayIn.x) / InDir.x;
            RayOut.x = Grid.ndWpLength;
            RayOut.y = RayIn.y + tmp * InDir.y;
            RayOut.z = RayIn.z + tmp * InDir.z;
            *pdebug << "line to: " << pPT3D(RayOut) << endl;
            if (Grid.iFrontBound == TGrid::finite)
            {
                *pdebug << "ray leaves geometry" << endl;
                return;
            }
            *pdebug << "ray is reflected at x=wplength" << endl;
            RayOld = RayOut;
            RayIn.x = 2.0 * Grid.ndWpLength - RayIn.x;
            InDir.x = -InDir.x; // reflecting ray at x=ndWpLength
            Normal.x = -Normal.x;
            InNormal.y = -InNormal.y;
            InNormal.z = -InNormal.z;
            ie = 1; // remain on patch
            if (l == 0 || l == 2) // hitting i=const line
                l = (l + 2) & 3; // take opposite line
            iTracedPatches = 1; // start again with first line
        }
        else if (ie >= Grid.iVolNodes) // upper i boundary
        {
            tmp = -RayIn.x / InDir.x;
            RayOut.x = 0;
            RayOut.y = RayIn.y + tmp * InDir.y;
            RayOut.z = RayIn.z + tmp * InDir.z;
            *pdebug << "line to: " << pPT3D(RayOut) << endl;
            if (Grid.iBackBound == TGrid::finite)
            {
                *pdebug << "ray leaves geometry" << endl;
                return;
            }
            *pdebug << "ray is reflected at x=0" << endl;
            RayOld = RayOut;
            RayIn.x = -RayIn.x; // reflecting ray at x=0
            InDir.x = -InDir.x;
            Normal.x = -Normal.x;
            InNormal.y = -InNormal.y;
            InNormal.z = -InNormal.z;
            ie = Grid.iVolNodes; // remain on patch
            if (l == 0 || l == 2) // hitting i=const line
                l = (l + 2) & 3; // take opposite line
            iTracedPatches = 1; // start again with first line
        }
        else if (je < 1) // lower j boundary
        {
            tmp = -RayIn.y / InDir.y;
            RayOut.x = RayIn.x + tmp * InDir.x;
            RayOut.y = 0;
            RayOut.z = RayIn.z + tmp * InDir.z;
            *pdebug << "line to: " << pPT3D(RayOut) << endl;
            if (Grid.iRightBound == TGrid::finite)
            {
                *pdebug << "ray leaves geometry" << endl;
                return;
            }
            *pdebug << "ray is reflected at y=0" << endl;
            RayOld = RayOut;
            RayIn.y = -RayIn.y; // reflecting ray at y=0
            InDir.y = -InDir.y;
            Normal.y = -Normal.y;
            InNormal.x = -InNormal.x;
            InNormal.z = -InNormal.z;
            je = 1; // remain on patch
            if (l == 1 || l == 3) // hitting j=const line
                l = (l + 2) & 3; // take opposite line
            iTracedPatches = 1; // start again with first line
        }
        else if (je > Grid.jVolNodes) // upper j boundary
        {
            tmp = (Grid.ndWpWidth - RayIn.y) / InDir.y;
            RayOut.x = RayIn.x + tmp * InDir.x;
            RayOut.y = Grid.ndWpWidth;
            RayOut.z = RayIn.z + tmp * InDir.z;
            *pdebug << "line to: " << pPT3D(RayOut) << endl;
            if (Grid.iLeftBound == TGrid::finite)
            {
                *pdebug << "ray leaves geometry" << endl;
                return;
            }
            *pdebug << "ray is reflected at y=wpwidth" << endl;
            RayOld = RayOut;
            RayIn.y = 2 * Grid.ndWpWidth - RayIn.y;
            InDir.y = -InDir.y; // reflecting ray at y=ndWpWidth
            Normal.y = -Normal.y;
            InNormal.x = -InNormal.x;
            InNormal.z = -InNormal.z;
            je = Grid.jVolNodes; // remain on patch
            if (l == 1 || l == 3) // hitting j=const line
                l = (l + 2) & 3; // take opposite line
            iTracedPatches = 1; // start again with first line
        }
    }
    iBadRays++; // all lines checked
    *pdebug << "couldn't trace ray " << iRay << " beyond patch ("
            << ie << ',' << je << ')' << endl;
}

// ***************************************************************************
// raytracing for reflected rays
// ray with strength ndRefFlux leaving from RayIn on element (i,j) with
// direction OutDir (local surface normal: Normal)
//
// input:   element (i,j), RayIn, OutDir, Normal, ndRefFlux, iRay
// output:  Solid.ndHeat, SolSurf, iRay, InNormal, incos, insin2, inposdir,
//          ndRefFlux, ndFlux, Normal
//
// ***************************************************************************

void RayTrace::TraceGas(int i, int j)
{
    int ie, je, ia, ja, iarray, jarray, ihit;
    bool bx, by;
    prec tmp, raytrip, delta_omega, conerad2,
        fparref, fperref, fparin, fperin, ndTotalFlux,
        reffrac, parflux, perflux, ratio;
    TPoint3D ptRay, ptDir, ptNormal;
    TPoint pt;
    Matrix<bool> bPatchHitted(6, 6);
    rmatrix ndPatchFlux(6, 6);

    *pdebug << "reflected ray no. " << iRay << " i is emitted from element ("
            << i << ',' << j << endl
            << "point of emission:  " << pPT3D(RayIn) << endl
            << "surface normal:     " << pPT3D(Normal) << " area: " << ndArea
            << endl
            << "incident direction: " << pPT3D(InDir) << endl
            << "emission direction: " << pPT3D(OutDir) << " strength: "
            << ndRefFlux << endl;
    tmp = 180.0 / PI * acos(min(max(Normal * OutDir, -1.0), 1.0));
    *pdebug << "angle normal-emission:  " << tmp << endl;
    tmp = 180.0 / PI * acos(OutDir.z);
    *pdebug << "angle surface-emission: " << tmp << endl;
    RayOld.z = -Grid.ndDepth / 10; // artificial ray origin
    tmp = (RayOld.z - RayIn.z) / InDir.z;
    RayOld.x = RayIn.x + tmp * InDir.x;
    RayOld.y = RayIn.y + tmp * InDir.y;
    *pdebug << "line from: " << pPT3D(RayOld) << endl;
    RayOld = RayIn;
    *pdebug << "line to: " << pPT3D(RayOld) << endl;

    iRay++; // next ray
    raytrip = Simulation.ndRefRadius; // traveled distance (for diffuse refl.)
    delta_omega = ndArea / (PI * sqr(raytrip)); // opening angle of cone
    iRefNum = 0; // numer of reflections
    ie = i; // position of surface element under ray
    je = j;

    do // loop over reflections
    {
        InDir = OutDir;
        if (InDir.z * Normal.z < -0.99) // return if ray leaves vertically
        {
            *pdebug << "ray goes vertically up and out" << endl;
            return;
        }
        InNormal = CrossProduct(OutDir, Normal); // normal to plane of incidence
        incos = OutDir * Normal; // angle of incidence
        insin2 = 2.0 * (1.0 - sqr(incos)); // 2*sin(theta)^2

        FindNextIntersection(ie, je, pt, ihit); // find next surface point
        if (ihit != Trace::intersection)
            return;
        if (bCheckPrev) // store reflection for next ray
        {
            ptPrev(iRefNum).x = ie;
            ptPrev(iRefNum).y = je;
        }
        inposdir = RayIn * InDir;
        CalcPatchVect(ie, je, pt); // calc position and normal at pt(u,v)
        reffrac = Material.GetReflection(InDir, OutDir, Normal, EField,
                                         Solid(ie, je, 1).ndTemp,
                                         &fparin, &fperin, &fparref, &fperref);
        parflux = ndRefFlux * (1 - fparref) * fparin;
        perflux = ndRefFlux * (1 - fperref) * fperin;
        ndFlux = (1 - reffrac) * ndRefFlux;
        ndRefFlux -= ndFlux;
        *pdebug << "ray is reflected from element (" << ie << ',' << je << ')'
                << endl
                << "point of reflection:  " << pPT3D(RayIn) << endl
                << "surface normal:       " << pPT3D(Normal) << endl
                << "incoming direction:   " << pPT3D(InDir)
                << " reflected fraction: " << reffrac << endl
                << "reflection direction: " << pPT3D(OutDir)
                << "strength: " << ndRefFlux << endl;
        RayOld = RayIn;
        *pdebug << "line to: " << pPT3D(RayOld) << endl;

        tmp = RayIn * InDir - inposdir; // traveled distance
        if (Simulation.bSpecReflec)
            raytrip += tmp; // add to previous distance for diffuse reflection
        else
            raytrip = tmp;
        conerad2 = delta_omega * sqr(raytrip); // radius of cone on surface
        ndTotalFlux = 0;
        unitwriteln(9, "After a trip of %11.4lf %11.4lf absorbed by:",
                    raytrip, ndFlux);
        for (iarray = 0; iarray <= 5; iarray++) // check if neighbor nodes
        { // are within cone
            ia = ie + iarray - 2; // node index
            for (jarray = 0; jarray <= 5; jarray++)
            {
                ja = je + jarray - je - 2;
                bPatchHitted(iarray, jarray) = false;
                if (ia < 1) // front boundary
                {
                    // symmetric boundary x=wpl?
                    if (Grid.iFrontBound != TGrid::symmetric)
                        continue;
                    bx = true; // reflecting at x=wpl
                    ia = 2 - ia;
                    ptRay.x = 2 * Grid.ndWpLength;
                }
                else if (ia > Grid.iVolNodes) // back boundary
                {
                    // symmetric boundary x=0?
                    if (Grid.iBackBound != TGrid::symmetric)
                        continue;
                    bx = true; // reflecting at x=0
                    ia = 2 * Grid.iVolNodes - ia;
                    ptRay.x = 0;
                }
                else
                    bx = false; // no reflection of x
                if (ja < 1) // right boundary
                {
                    // symmetric boundary y=0?
                    if (Grid.iRightBound != TGrid::symmetric)
                        continue;
                    by = true; // reflecting at y=0
                    ja = 2 - ja;
                    ptRay.y = 0;
                }
                else if (ja > Grid.jVolNodes) // left boundary
                {
                    // symmetric boundary y=wpw?
                    if (Grid.iLeftBound != TGrid::symmetric)
                        continue;
                    by = true; // reflecting at y=wpw
                    ja = 2 * Grid.jVolNodes - ja;
                    ptRay.y = 2 * Grid.ndWpWidth;
                }
                else
                    by = false; // no reflection of y
                if (bx) // reflecting x
                {
                    // vector to node (ia,ja)
                    ptRay.x -= Solid(ia, ja, 1).ndNode.x;
                    // local normal at (ia,ja)
                    ptNormal.y = -SolSurf(ia, ja).Normal.x;
                }
                else
                {
                    ptRay.x = Solid(ia, ja, 1).ndNode.x;
                    ptNormal.x = SolSurf(ia, ja).Normal.x;
                }
                if (by) // reflecting y
                {
                    ptRay.y -= Solid(ia, ja, 1).ndNode.y;
                    ptNormal.y = -SolSurf(ia, ja).Normal.y;
                }
                else
                {
                    ptRay.y = Solid(ia, ja, 1).ndNode.y;
                    ptNormal.y = SolSurf(ia, ja).Normal.y;
                }
                ptRay.z = Solid(ia, ja, 1).ndNode.z;
                ptNormal.z = SolSurf(ia, ja).Normal.z;
                ptRay = RayIn - ptRay; // vector between nodes (ie,je),(ia,ja)
                if (ptRay.Norm() - sqr(ptRay * InDir) > conerad2)
                    continue; // if outside cone
                tmp = ptNormal * InDir; // cosine of incident angle
                if (tmp >= 0) // surface hitted from front?
                {
                    bPatchHitted(iarray, jarray) = true; // patch hitted
                    ndPatchFlux(iarray, jarray) = SolSurf(ia, ja).ndCoarseArea * tmp;
                    ndTotalFlux += ndPatchFlux(iarray, jarray);
                }
            }
        }

        ptNormal = Normal; // save current direction and normal
        ptRay = OutDir; // in temporary variables
        ptDir = InDir;
        if (ndTotalFlux <= 0.0) // no splitting
        {
            ia = ie + 1 + int(pt.x); // nodal index
            ja = je + 1 + int(pt.y);
            tmp = SolSurf(ia, ja).Normal * InDir; // cosine theta
            if (tmp < 0)
            {
                iBadRays++;
                *pdebug << "ray " << iRay << " hits surface from behind" << endl;
                return;
            }
            *pdebug << "bundle of strength " << ndFlux
                    << " is entering surface at (" << ia << ',' << ja << ')'
                    << endl;

            SolSurf(ia, ja).ndMultAbs += ndFlux; // absorbed in volume
            if (Simulation.iReflection == TSimulation::polarized)
            { // absorbed power of parallel and perpendicularly
                // polarized component
                SolSurf(ia, ja).ndParAbs += parflux;
                SolSurf(ia, ja).ndPerAbs += perflux;
            }
            SolSurf.ndTotalMultAbs += ndFlux;
            RayIn = Solid(ia, ja, 1).ndNode;
            Normal = SolSurf(ia, ja).Normal;
            TraceSolid(ia, ja); // trace ray into material
        }
        else // split onto patches
        {
            ratio = ndFlux / ndTotalFlux; // split factor
            for (iarray = 0; iarray <= 5; iarray++) // loop over neighbor nodes
            {
                ia = ie + iarray - 2; // node index
                if (ia < 1) // reflecting at x=wpl
                    ia = 2 - ia;
                else if (ia > Grid.iVolNodes) // reflecting at x=0
                    ia = 2 * Grid.iVolNodes - ia;
                for (jarray = 0; jarray <= 5; jarray++)
                {
                    ja = je + jarray - je - 2;
                    if (bPatchHitted(iarray, jarray))
                    {
                        if (ja < 1) // reflecting at y=0
                            ja = 2 - ja;
                        else if (ja > Grid.jVolNodes) // reflecting at y=wpw
                            ja = 2 * Grid.jVolNodes - ja;
                        ndFlux = ratio * ndPatchFlux(iarray, jarray);
                        tmp = ndPatchFlux(iarray, jarray) / ndTotalFlux;
                        *pdebug << "bundle of strength " << ndFlux
                                << " is entering surface at (" << ia << ',' << ja << ')'
                                << endl;
                        // absorbed in volume
                        SolSurf(ia, ja).ndMultAbs += ndFlux;
                        if (Simulation.iReflection == TSimulation::polarized)
                        { // absorbed power of parallel and perpendicularly
                            // polarized component
                            SolSurf(ia, ja).ndParAbs += parflux * tmp;
                            SolSurf(ia, ja).ndPerAbs += perflux * tmp;
                        }
                        SolSurf.ndTotalMultAbs += ndFlux;
                        RayIn = Solid(ia, ja, 1).ndNode;
                        Normal = SolSurf(ia, ja).Normal;
                        InDir = ptDir; // restore incident direction
                        TraceSolid(ia, ja); // trace ray into material
                    }
                }
            }
        }
        OutDir = ptRay; // restore old dir and normal
        Normal = ptNormal;
    } while (ndRefFlux >= Simulation.ndMinPower); // until energy depleted
    *pdebug << "bundle died due to lack of energy" << endl;
}

// ***************************************************************************
// calculate surface coordinates and local normal of element (i,j)
//
// input:       patch (i,j), position pt, Solid.ndNode, SolSurf.Normal
// output:      RayIn, Normal
//
// ***************************************************************************

void RayTrace::CalcPatchVect(int i, int j, const TPoint &pt)
{
    prec an12, an22, an23, an33, an11, an21, an31, an41;

    an12 = 1.0 - pt.y;
    an22 = 1.0 + pt.y;
    an23 = 1.0 + pt.x;
    an33 = 1.0 - pt.x;
    an11 = an23 * an12;
    an21 = an23 * an22;
    an31 = an33 * an22;
    an41 = an33 * an12;
    RayIn = an11 * Solid(i + 1, j, 1).ndNode + an21 * Solid(i + 1, j + 1, 1).ndNode + an31 * Solid(i, j + 1, 1).ndNode + an41 * Solid(i, j, 1).ndNode;
    Normal = an11 * SolSurf(i + 1, j).Normal + an21 * SolSurf(i + 1, j + 1).Normal + an31 * SolSurf(i, j + 1).Normal + an41 * SolSurf(i, j).Normal;
    RayIn *= 0.25;
    Normal.Normalize();
    return;
}

// ***************************************************************************
// calculate surface area of minipatch on element (i,j)
//
// input:       patch (i,j), position pt, DuDv/16 (sub area), Solid.ndNode
// output:      ndArea
//
// ***************************************************************************

void RayTrace::CalcPatchArea(int i, int j, const TPoint &pt)
{
    prec an12, an22, an23, an33;
    TPoint3D xu, xv;

    an12 = 1.0 - pt.y;
    an22 = 1.0 + pt.y;
    an23 = 1.0 + pt.x;
    an33 = 1.0 - pt.x;
    xu = an12 * (Solid(i + 1, j, 1).ndNode - Solid(i, j, 1).ndNode) + an22 * (Solid(i + 1, j + 1, 1).ndNode - Solid(i, j + 1, 1).ndNode);
    xv = an23 * (Solid(i + 1, j + 1, 1).ndNode - Solid(i + 1, j, 1).ndNode) + an33 * (Solid(i, j + 1, 1).ndNode - Solid(i, j, 1).ndNode);

    ndArea = sqrt(xu.Norm() * xv.Norm() - sqr(xu * xv)) * delta_uv16th;
    return;
}

// ***************************************************************************
// calculate intersection of ray with boundary (m,m+1) of element (i,j)
//
// parameters:  element (i,j), line l, hit flag, index of next element,
//              number of checked line (1=check opposite line,2 and 3
//              neighboring line; 0=check all lines)
// output:      next line, hit flag, index of next element, incos, insin2,
//              InNormal, InDir
// input:       ptCorners, Solid.ndNode, InNormal, RayIn, InDir
//
// ***************************************************************************

void RayTrace::CalcLineIntersection(int i, int j, int &l, int &ihit,
                                    int &nextelem, int ipatch)
{
    int iBegin, jBegin, iEnd, jEnd, m;
    prec fper, fpar, uv, dir2ReRi, norm2ReRi, d, h, tmp;
    TPoint3D ptNormal, ptDir, pt2ReRi;

    ihit = Trace::no_intersection;
    iBegin = i + ptCorners(l).x; // begin of boundary line
    jBegin = i + ptCorners(l).y;
    m = (l + 1) & 3; // next line/corner
    iEnd = i + ptCorners(m).x; // end of boundary line
    jEnd = i + ptCorners(m).y;

    ptNormal = 2.0 * RayIn - Solid(iBegin, jBegin, 1).ndNode
               - Solid(iEnd, jEnd, 1).ndNode;
    ptDir = Solid(iEnd, jEnd, 1).ndNode - Solid(iBegin, jBegin, 1).ndNode;

    fper = ptNormal * InNormal; // numerator eqn(66)
    fpar = ptDir * InNormal; // denominator eqn(66)
    if (fpar == 0) // no intersection
    {
        *pdebug << "Found no intersection; trying next side (|uv|>>0)";
        return;
    }
    uv = fper / fpar; // u*v of intersection;
    pt2ReRi = ptNormal - uv * ptDir; // 2(re-rip) in eqn(67)
    dir2ReRi = InDir * pt2ReRi; // s*2(re-rip)
    norm2ReRi = Normal * pt2ReRi; // n*2(re-rip)
    d = (dir2ReRi - incos * norm2ReRi) / insin2; // parameter d in eqn(67)

    *pdebug << "checking for intersection with line " << l << " of element ("
            << i << ',' << j << ')' << endl;

    if (fabs(d) <= 1e-8) // RayIn on boundary
    {
        *pdebug << "This shouldn\'t happen: found RayIn itself; "
                   "trying next side (d=" << d << ')' << endl;
        return;
    }

    h = (norm2ReRi - incos * dir2ReRi) / insin2; // parameter h in eqn(67)
    if (fabs(uv) > 1.0001) // check next side
    {
        switch (ipatch)
        {
        case Trace::loop_over: // just loop over all sides
            break;

        case Trace::neighbor_line: // line next to checked one
            if (uv > 0) // counter clockwise
                l = (l + 1) & 3;
            else // clockwise
                l = (l + 3) & 3;
            break;

        case Trace::opposite_line: // line opposite to checked one
            l = (l + 2) & 3;
            break;
        }
        *pdebug << "no intersection; trying next side (uv=" << uv
                << " d=" << d << " h=" << h << ')' << endl;
        return;
    }
    if (d < 0.0 && ipatch == 0) // intersection in opposite direction of ray
    {
        *pdebug << "no intersection; trying next side (uv=" << uv
                << " d=" << d << " h=" << h << ')' << endl;
        return;
    }

    tmp = RayIn.z + d * InDir.z;
    if (tmp <= Solid.ndMinSurZ) // out of groove
    {
        ihit = Trace::leaves_groove;
        *pdebug << "ray leaves groove (z=" << tmp << " < zmin="
                << Solid.ndMinSurZ << ')' << endl;
        return;
    }

    pt2ReRi /= 2;
    pt2ReRi = pt2ReRi / 2;
    pt2ReRi = pt2ReRi / 2;
    pt2ReRi = RayIn - pt2ReRi;

    pt2ReRi = RayIn - pt2ReRi / 2; // intersection point
    *pdebug << "crossing element at " << pPT3D(pt2ReRi) << "with uv=" << uv
            << ", d=" << d << ", h=" << h
            << Solid.ndMinSurZ << ')' << endl;

    if (h <= 0.0) // true intersection (passes under line)
    {
        ihit = Trace::intersection;
        if (ipatch > 0) // not initial element
            return;
        ihit = Trace::no_intersection; // hits same patch
        InDir = d * InDir - ((1e-6) - h) * Normal; // correct direction
        InDir.Normalize();
        InNormal = CrossProduct(InDir, Normal);
        incos = InDir * Normal;
        insin2 = 2.0 * (1.0 - sqr(incos));
        *pdebug << "hitting itself (probably bad normal); correcting InDir"
                << endl << "new direction: " << pPT3D(InDir) << endl;
        return;
    }

    if (uv >= 1.0) // pass on to next sides
        nextelem = Trace::diagonal_right; // right corner
    else if (uv <= -1.0)
        nextelem = Trace::diagonal_left; // left corner
    else
        nextelem = Trace::straight; // straight

    ihit = Trace::next_element;
    return;
}
