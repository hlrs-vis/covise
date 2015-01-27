/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BlobVisualiser.h"
#include <osg/Vec3>
#include <iostream>

using namespace osg;

BlobVisualiser::BlobVisualiser(Vec3f &center, float radius, int numOfSegments, Vec4f &color)
{
    this->setPosition(center);
    this->setRadius(radius);
    this->setNumOfSegments(numOfSegments);
    this->setColor(color);

    this->createBlob();
}

BlobVisualiser::~BlobVisualiser()
{
    this->blobGeometry->removePrimitiveSet(0);
    this->blobGeometry = NULL;
}

void BlobVisualiser::setPosition(const Vec3f &newPos)
{
    this->center = newPos;
}

void BlobVisualiser::setRadius(const float r)
{
    this->radius = r;
}

void BlobVisualiser::setNumOfSegments(const int n)
{
    this->numOfSegments = n;
}

void BlobVisualiser::setColor(const Vec4f &c)
{
    this->color = c;
}

ref_ptr<Geometry> BlobVisualiser::getGeometry() const
{
    if (blobGeometry.valid())
        return blobGeometry.get();
    else
    {
        std::cout << "BlobVisualiser::getGeometry() is NULL!" << std::endl;
        return NULL;
    }
}

void BlobVisualiser::createBlob()
{
    blobGeometry = new Geometry();

    ref_ptr<Vec3Array> vertices = this->createCircle();
    blobGeometry->setVertexArray(vertices.get());

    ref_ptr<Vec4Array> colors = new Vec4Array();
    colors->push_back(color);
    blobGeometry->setColorArray(colors.get());
    blobGeometry->setColorBinding(Geometry::BIND_OVERALL);

    ref_ptr<Vec3Array> normals = new Vec3Array;
    normals->push_back(Vec3f(0.0f, 0.0f, 1.0f));
    blobGeometry->setNormalArray(normals.get());
    blobGeometry->setNormalBinding(Geometry::BIND_OVERALL);

    blobGeometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLE_FAN /*POLYGON*/ /*LINE_LOOP*/, 0, vertices->size()));
}

ref_ptr<Vec3Array> BlobVisualiser::createCircle()
{

    ref_ptr<Vec3Array> vertices = new Vec3Array();

    /*
     fast circle calculation, see
     http://slabode.exofire.net/circle_draw.shtml
    */

    float theta = 2 * 3.1415926 / float(this->numOfSegments);
    float tangetialFactor = tanf(theta); // calculate the tangential factor

    float radialFactor = cosf(theta); // calculate the radial factor

    float x = this->radius; // we start at angle = 0
    float cx = this->center.x();
    float cy = this->center.y();
    float z = this->center.z();

    float y = 0;

    for (int ii = 0; ii < numOfSegments; ii++)
    {
        vertices->push_back(Vec3f(x + cx, y + cy, z));

        // calculate the tangential vector
        // remember, the radial vector is (x, y)
        // to get the tangential vector we flip those coordinates and negate one of them

        float tx = -y;
        float ty = x;

        // add the tangential vector

        x += tx * tangetialFactor;
        y += ty * tangetialFactor;

        // correct using the radial factor

        x *= radialFactor;
        y *= radialFactor;
    }

    return vertices;
}
