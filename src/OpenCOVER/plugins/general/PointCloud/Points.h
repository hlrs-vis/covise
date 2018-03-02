/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POINTS_H_
#define _POINTS_H_

class fileInfo;

struct Point
{
    float x;
    float y;
    float z;
};

struct Color
{
    float r;
    float g;
    float b;
};

struct PointSet
{
    int size;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
    Point *points;
    Color *colors;
};

struct pointSelection
{
    const fileInfo *file;
    int pointSetIndex;
    int pointIndex;
    osg::MatrixTransform *transformationMatrix;
};

#endif
