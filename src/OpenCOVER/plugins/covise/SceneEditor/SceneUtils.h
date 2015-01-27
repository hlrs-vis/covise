/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCENE_UTILS_H
#define SCENE_UTILS_H

#include "SceneObject.h"

#include <PluginUtil/coPlane.h>

#include <iostream>
#include <QDomElement>

class SceneUtils
{
private:
    SceneUtils();
    virtual ~SceneUtils();

    static osg::ref_ptr<osg::Node> createSingleGeometryFromXML(QDomElement *geometryElem);

public:
    static int insertNode(osg::Group *node, SceneObject *so);
    static int removeNode(osg::Group *node);

    static osg::ref_ptr<osg::Node> createGeometryFromXML(QDomElement *parentElem);

    static SceneObject *followFixedMountsToParent(SceneObject *so);

    static bool getPlaneIntersection(opencover::coPlane *plane, osg::Matrix pointerMat, osg::Vec3 &point);

    static float getPlaneVisibility(osg::Vec3 position, osg::Vec3 normal);
};

#endif
