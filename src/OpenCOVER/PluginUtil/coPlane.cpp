/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coPlane.h"
#include <osg/Vec3>
#include <osg/BoundingBox>
#include <stdio.h>

using namespace opencover;

coPlane::coPlane(osg::Vec3 n, osg::Vec3 p)
{
    //fprintf(stderr,"\ncoPlane::coPlane\n");
    //fprintf(stderr,"\tn=[%f %f %f]\n", n[0], n[1], n[2]);
    //fprintf(stderr,"\tp=[%f %f %f]\n", p[0], p[1], p[2]);
    update(n, p);
}

void
coPlane::update(osg::Vec3 n, osg::Vec3 p)
{
    //fprintf(stderr,"coPlane::update\n");
    _normal = n;
    _normal.normalize();

    _point = p;

    _d = _normal * _point;

    //fprintf(stderr,"\t pos:[%f %f %f] normal=[%f %f %f]\n", _point[0], _point[1], _point[2], _normal[0], _normal[1], _normal[2]);
}
coPlane::~coPlane()
{
}

float
coPlane::getPointDistance(osg::Vec3 &p)
{
    float dist;

    dist = _normal * p - _d;

    return (dist);
}

bool
coPlane::getLineIntersectionPoint(osg::Vec3 &lp1, osg::Vec3 &lp2, osg::Vec3 &isectPoint)
{
    //fprintf(stderr,"coPlane::getLineIntersectionPoint\n");

    // von Juergen Schulze-Doebold
    float numer; // numerator (=Zahler)
    float denom; // denominator (=Nenner)
    osg::Vec3 diff1; // difference vector between v1 and v2
    osg::Vec3 diff2; // difference vector between pp and v1

    diff1 = lp1 - lp2;
    //fprintf(stderr,"diff1=[%f %f %f]\n", diff1[0], diff1[1], diff1[2]);
    denom = diff1 * _normal;
    if (denom == 0.0f) // does an intersection exist?
    {
        return false;
    }
    diff2 = _point - lp1;
    numer = diff2 * _normal;
    diff1 = diff1 * (numer / denom);
    isectPoint = lp1 + diff1;
    //fprintf(stderr,"\tlp1=[%f %f %f]\n", lp1[0], lp1[1], lp1[2]);
    //fprintf(stderr,"\tlp2=[%f %f %f]\n", lp2[0], lp2[1], lp2[2]);
    //fprintf(stderr,"\tisectPoint=[%f %f %f]\n", isectPoint[0], isectPoint[1], isectPoint[2]);
    return true;
}

bool
coPlane::getLineSegmentIntersectionPoint(osg::Vec3 &lp1, osg::Vec3 &lp2, osg::Vec3 &isectPoint)
{
    float xmin, xmax, ymin, ymax, zmin, zmax;
    if (lp1[0] < lp2[0])
    {
        xmin = lp1[0];
        xmax = lp2[0];
    }
    else
    {
        xmin = lp2[0];
        xmax = lp1[0];
    }
    if (lp1[1] < lp2[1])
    {
        ymin = lp1[1];
        ymax = lp2[1];
    }
    else
    {
        ymin = lp2[1];
        ymax = lp1[1];
    }

    if (lp1[2] < lp2[2])
    {
        zmin = lp1[2];
        zmax = lp2[2];
    }
    else
    {
        zmin = lp2[2];
        zmax = lp1[2];
    }

    if (getLineIntersectionPoint(lp1, lp2, isectPoint))
    {
        if ((isectPoint[0] >= xmin && isectPoint[0] <= xmax)
            && (isectPoint[1] >= ymin && isectPoint[1] <= ymax)
            && (isectPoint[2] >= zmin && isectPoint[2] <= zmax))

            return true;
    }

    return false;
}

osg::Vec3
coPlane::getProjectedPoint(osg::Vec3 &p)
{
    float d;
    osg::Vec3 point_proj;

    // distance point to plane
    d = getPointDistance(p);

    point_proj = p - _normal * d;

    return (point_proj);
}

int
coPlane::getBoxIntersectionPoints(osg::BoundingBox &box, osg::Vec3 *isectPoints)
{
    osg::Vec3 bpoints[8];
    osg::Vec3 tmpIsectPoints[12];

    osg::Vec3 iPoint;
    int numIsectPoints = 0;
    osg::Vec3 diff[5], tmp, n;
    int i, k;
    bool swapped;

    bpoints[0].set(box.xMin(), box.yMin(), box.zMin());
    bpoints[1].set(box.xMax(), box.yMin(), box.zMin());
    bpoints[2].set(box.xMax(), box.yMax(), box.zMin());
    bpoints[3].set(box.xMin(), box.yMax(), box.zMin());
    bpoints[4].set(box.xMin(), box.yMin(), box.zMax());
    bpoints[5].set(box.xMax(), box.yMin(), box.zMax());
    bpoints[6].set(box.xMax(), box.yMax(), box.zMax());
    bpoints[7].set(box.xMin(), box.yMax(), box.zMax());

    if (getLineSegmentIntersectionPoint(bpoints[0], bpoints[1], iPoint))
    {
        //fprintf(stderr,"found intersection in 0-1\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[0][0], bpoints[1][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[0][1], bpoints[1][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[0][2], bpoints[1][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[1], bpoints[2], iPoint))
    {
        //fprintf(stderr,"found intersection in 1-2\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[1][0], bpoints[2][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[1][1], bpoints[2][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[1][2], bpoints[2][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[2], bpoints[3], iPoint))
    {
        //fprintf(stderr,"found intersection in 2-3\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[2][0], bpoints[3][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[2][1], bpoints[3][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[2][2], bpoints[3][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[3], bpoints[0], iPoint))
    {
        //fprintf(stderr,"found intersection in 3-0\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[3][0], bpoints[0][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[3][1], bpoints[0][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[3][2], bpoints[0][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[4], bpoints[5], iPoint))
    {
        //fprintf(stderr,"found intersection in 4-5\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[4][0], bpoints[5][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[4][1], bpoints[5][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[4][2], bpoints[5][2], iPoint[2]);
        numIsectPoints++;
    }
    if (getLineSegmentIntersectionPoint(bpoints[5], bpoints[6], iPoint))
    {
        //fprintf(stderr,"found intersection in 5-6\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[5][0], bpoints[6][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[5][1], bpoints[6][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[5][2], bpoints[6][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[6], bpoints[7], iPoint))
    {
        //fprintf(stderr,"found intersection in 6-7\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[6][0], bpoints[7][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[6][1], bpoints[7][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[6][2], bpoints[7][2], iPoint[2]);
        numIsectPoints++;
    }
    if (getLineSegmentIntersectionPoint(bpoints[7], bpoints[4], iPoint))
    {
        //fprintf(stderr,"found intersection in 7-4\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[7][0], bpoints[4][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[7][1], bpoints[4][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[7][2], bpoints[4][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[0], bpoints[4], iPoint))
    {
        //fprintf(stderr,"found intersection in 0-4\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[0][0], bpoints[4][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[0][1], bpoints[4][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[0][2], bpoints[4][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[3], bpoints[7], iPoint))
    {
        //fprintf(stderr,"found intersection in 3-7\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[3][0], bpoints[7][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[3][1], bpoints[7][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[3][2], bpoints[7][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[1], bpoints[5], iPoint))
    {
        //fprintf(stderr,"found intersection in 1-5\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[1][0], bpoints[5][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[1][1], bpoints[5][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[1][2], bpoints[5][2], iPoint[2]);
        numIsectPoints++;
    }

    if (getLineSegmentIntersectionPoint(bpoints[2], bpoints[6], iPoint))
    {
        //fprintf(stderr,"found intersection in 2-6\n");
        tmpIsectPoints[numIsectPoints] = iPoint;
        //fprintf(stderr,"xmin xmax xisect = [%f %f %f]\n",  bpoints[2][0], bpoints[6][0], iPoint[0]);
        //fprintf(stderr,"ymin ymax yisect = [%f %f %f]\n",  bpoints[2][1], bpoints[6][1], iPoint[1]);
        //fprintf(stderr,"zmin zmax zisect = [%f %f %f]\n",  bpoints[2][2], bpoints[6][2], iPoint[2]);
        numIsectPoints++;
    }

    isectPoints[0] = tmpIsectPoints[0];
    bool found = false;
    int J = 1;
    for (int i = 1; i < numIsectPoints; i++)
    {
        found = false;
        for (int j = 0; j < J; j++)
        {
            if (tmpIsectPoints[i] == isectPoints[j])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            isectPoints[J] = tmpIsectPoints[i];
            //fprintf(stderr,"!found isectPoints[%d]=%f %f %f\n", J, isectPoints[J][0], isectPoints[J][1],isectPoints[J][2]);
            J++;
        }
    }

    numIsectPoints = J;
    //fprintf(stderr,"found %d intersection points\n", numIsectPoints);
    //for (int i=0; i<numIsectPoints; i++)
    //   fprintf(stderr,"tmpIsectPoints[i]=%f %f %f\n", tmpIsectPoints[i][0], tmpIsectPoints[i][1],tmpIsectPoints[i][2]);

    //fprintf(stderr,"found %d real intersection points\n", J);
    //for (int i=0; i<J; i++)
    //   fprintf(stderr,"isectPoints[i]=%f %f %f\n", isectPoints[i][0], isectPoints[i][1],isectPoints[i][2]);

    // now sort the points
    if (numIsectPoints >= 3 && numIsectPoints <= 7)
    {
        for (i = 0; i < numIsectPoints - 1; i++) // generate difference vectors
        {
            diff[i] = isectPoints[i + 1] - isectPoints[0];
        }
        swapped = true;
        while (swapped == true)
        {
            swapped = false;
            for (i = 0; i < numIsectPoints - 2 && swapped == false; i++)
            {
                for (k = i + 1; k < numIsectPoints - 1 && swapped == false; k++)
                {

                    n = diff[i] ^ diff[i + 1];
                    n.normalize();
                    if (n * _normal < 0.0f) // do normals point into opposite directions?
                    {
                        // swap points
                        tmp = isectPoints[k + 1];
                        isectPoints[k + 1] = isectPoints[i + 1];
                        isectPoints[i + 1] = tmp;
                        // swap difference vectors
                        tmp = diff[k];
                        diff[k] = diff[i];
                        diff[i] = tmp;
                        swapped = true;
                    }
                }
            }
        }
    }
    return numIsectPoints;
}
