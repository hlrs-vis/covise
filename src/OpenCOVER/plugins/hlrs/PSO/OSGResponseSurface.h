/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSGResponseSurface_h
#define OSGResponseSurface_h

#include <osg/Geode>
#include <osg/Geometry>

class OSGResponseSurface : public osg::Geode
{
public:
    OSGResponseSurface(double (*setresponse)(double *), double *setlowerbound, double *setupperbound, bool *setinteger, double h);

private:
    osg::Geometry *lineGeometry;
    osg::Vec3Array *lineVertices;
};

#endif
