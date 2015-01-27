/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ShapeSegment.h"

ShapeSegment::ShapeSegment(osg::Group *parent, int nV)
{
    numVertices = nV;

    coordArray = new osg::Vec3Array(numVertices);
    normalArray = new osg::Vec3Array(numVertices);
    texcoordWallpos = new osg::Vec2Array(numVertices);

    geometry = new osg::Geometry();
    geometry->setVertexArray(coordArray);
    geometry->setNormalArray(normalArray);
    geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geometry->setUseDisplayList(false);
    geometry->setTexCoordArray(1, texcoordWallpos);
    geode = new osg::Geode();
    geode->addDrawable(geometry);
    parent->addChild(geode);

    wireframeGeometry = new osg::Geometry();
    wireframeGeometry->setVertexArray(coordArray);
    wireframeGeometry->setNormalArray(normalArray);
    wireframeGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    wireframeGeometry->setUseDisplayList(false);
    wireframeGeometry->setTexCoordArray(1, texcoordWallpos);
    wireframeGeode = new osg::Geode();
    wireframeGeode->addDrawable(wireframeGeometry);
    parent->addChild(wireframeGeode);

    osg::StateSet *s = wireframeGeometry->getOrCreateStateSet();
    s->setAttributeAndModes(new osg::Program(), osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
}

ShapeSegment::~ShapeSegment()
{
}

void ShapeSegment::setCoord(int index, osg::Vec3 pos)
{
    (*coordArray)[index] = pos;
    geometry->dirtyBound();
}

osg::Vec3 ShapeSegment::getCoord(int index)
{
    return (*coordArray)[index];
}

void ShapeSegment::setTexCoord(int index, osg::Vec2 tex)
{
    (*texcoordWallpos)[index] = tex;
}

void ShapeSegment::setNormal(int index, osg::Vec3 normal)
{
    (*normalArray)[index] = normal;
}

osg::Vec3 ShapeSegment::getNormal(int index)
{
    return (*normalArray)[index];
}

void ShapeSegment::setNormals(osg::Vec3 normal)
{
    for (int i = 0; i < numVertices; ++i)
    {
        (*normalArray)[i] = normal;
    }
}
