/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POINTGEOMETRY_DRAWABLE_H_
#define _POINTGEOMETRY_DRAWABLE_H_

#include <osg/Version>
#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/BufferObject>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#include "Points.h"

class PointCloudGeometry : public osg::Geometry
{
public:
    PointCloudGeometry(::PointSet *);

    /** Copy constructor using CopyOp to manage deep vs shallow copy.*/
    PointCloudGeometry(const PointCloudGeometry &drawimage, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);
    virtual Object *cloneType() const
    {
        return new PointCloudGeometry();
    }
    virtual Object *clone(const osg::CopyOp &copyop) const
    {
        return new PointCloudGeometry(*this, copyop);
    }
    virtual bool isSameKindAs(const Object *obj) const
    {
        return dynamic_cast<const PointCloudGeometry *>(obj) != NULL;
    }
    virtual const char *libraryName() const
    {
        return "PointCloudPlugin";
    }
    virtual const char *className() const
    {
        return "PointModeGeometry";
    }
    void changeLod(float sampleNum); // adjust point density
    void setPointSize(float pointSize); // adjust point size

protected:
    PointCloudGeometry(); // hide default constructor
    PointCloudGeometry &operator=(const PointCloudGeometry &)
    {
        return *this;
    }
    virtual ~PointCloudGeometry();

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

private:
    PointSet *pointSet;
    float subsample = 1.;
    float pointSize = 1.;
    float maxPointSize;
    osg::Point *pointstate;
    osg::StateSet *stateset;
    osg::BoundingBox box;
    osg::Vec3Array *points;
    osg::Vec3Array *colors;
    osg::VertexBufferObject *vertexBufferArray;
    osg::ElementBufferObject *primitiveBufferArray;
};
#endif
