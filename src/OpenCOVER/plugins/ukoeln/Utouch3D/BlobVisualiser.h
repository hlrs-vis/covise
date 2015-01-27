/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BLOBVISUALISER_H
#define BLOBVISUALISER_H

/****************************************************************************\
 **                                              (C)2010 Anton Baumesberger  **
 **                                                                          **
 ** Description: Utouch3D Plugin                                             **
 **                                                                          **
 **                                                                          **
 ** Author: Anton Baumesberger	                                             **
 **                                                                          **
\****************************************************************************/

#include <osg/Vec2f>
#include <osg/Vec3>
#include <osg/ref_ptr>
#include <osg/Geometry>

class BlobVisualiser
{
public:
    BlobVisualiser(osg::Vec3f &center, float radius, int numOfSegments, osg::Vec4f &color);
    ~BlobVisualiser();

    void setPosition(const osg::Vec3f &newPos);
    void setRadius(const float r);
    void setColor(const osg::Vec4f &c);
    void setNumOfSegments(const int n);

    osg::ref_ptr<osg::Geometry> getGeometry() const;

protected:
    osg::Vec3f center;
    float radius;
    int numOfSegments;
    osg::Vec4f color;

    osg::ref_ptr<osg::Geometry> blobGeometry;

private:
    void createBlob();
    osg::ref_ptr<osg::Vec3Array> createCircle();
};

#endif // BLOBVISUALISER_H
