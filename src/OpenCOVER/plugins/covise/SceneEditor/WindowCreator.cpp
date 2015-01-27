/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WindowCreator.h"

#include <QDir>
#include <iostream>

#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/ShapeDrawable>

#include <cover/coVRPluginSupport.h>

WindowCreator::WindowCreator()
{
}

WindowCreator::~WindowCreator()
{
}

SceneObject *WindowCreator::createFromXML(QDomElement *root)
{
    Window *win = new Window();
    if (!buildFromXML(win, root))
    {
        delete win;
        return NULL;
    }
    return win;
}

bool WindowCreator::buildFromXML(SceneObject *so, QDomElement *root)
{
    if (!buildGeometryFromXML((Window *)so, root))
    {
        return false;
    }
    return SceneObjectCreator::buildFromXML(so, root);
}

bool WindowCreator::buildGeometryFromXML(Window *Window, QDomElement *root)
{
    float width(1.0f);
    float height(1.0f);
    QDomElement geoElem = root->firstChildElement("geometry");
    if (!geoElem.isNull())
    {
        QDomElement w = geoElem.firstChildElement("width");
        if (!w.isNull())
        {
            width = w.attribute("value").toFloat();
        }
        QDomElement h = geoElem.firstChildElement("height");
        if (!h.isNull())
        {
            height = h.attribute("value").toFloat();
        }
    }

    // helper to mount window
    Window->setGeometryNode(new osg::Group());

    Window->setGeometry(width, height);

    return true;
}
