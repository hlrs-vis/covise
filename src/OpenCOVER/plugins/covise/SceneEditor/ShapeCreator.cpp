/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ShapeCreator.h"

#include <QDir>
#include <iostream>

#include <osgDB/ReadFile>
#include <osg/Shape>

#include <cover/coVRPluginSupport.h>

ShapeCreator::ShapeCreator()
{
}

ShapeCreator::~ShapeCreator()
{
}

SceneObject *ShapeCreator::createFromXML(QDomElement *root)
{
    Shape *shape = new Shape();
    if (!buildFromXML(shape, root))
    {
        delete shape;
        return NULL;
    }
    return shape;
}

bool ShapeCreator::buildFromXML(SceneObject *so, QDomElement *root)
{
    if (!buildGeometryFromXML((Shape *)so, root))
    {
        return false;
    }
    return SceneObjectCreator::buildFromXML(so, root);
}

bool ShapeCreator::buildGeometryFromXML(Shape *shape, QDomElement *root)
{
    float width(1.0f);
    float length(1.0f);
    float height(1.0f);
    Shape::GeometryType geometryType(Shape::GEOMETRY_UNSPECIFIED);

    QDomElement geoElem = root->firstChildElement("geometry");
    if (!geoElem.isNull())
    {
        QDomElement s = geoElem.firstChildElement("shape");
        if (!s.isNull())
        {
            if (s.attribute("value").toStdString() == "cuboid")
            {
                geometryType = Shape::GEOMETRY_CUBOID;
            }
            else if (s.attribute("value").toStdString() == "cylinder")
            {
                geometryType = Shape::GEOMETRY_CYLINDER;
            }
            else if (s.attribute("value").toStdString() == "prism")
            {
                geometryType = Shape::GEOMETRY_PRISM;
            }
        }
        QDomElement w = geoElem.firstChildElement("width");
        if (!w.isNull())
        {
            width = w.attribute("value").toFloat();
        }
        QDomElement h = geoElem.firstChildElement("height");
        if (!h.isNull())
        {
            if (h.attribute("value").toStdString() == "parent")
            {
                shape->setAutomaticHeight(true);
            }
            else
            {
                shape->setAutomaticHeight(false);
                height = h.attribute("value").toFloat();
            }
        }
        QDomElement l = geoElem.firstChildElement("length");
        if (!l.isNull())
        {
            length = l.attribute("value").toFloat();
        }
    }
    if (geometryType == Shape::GEOMETRY_UNSPECIFIED)
    {
        return false;
    }

    shape->setGeometryType(geometryType);
    shape->setSize(width, height, length);

    return true;
}
