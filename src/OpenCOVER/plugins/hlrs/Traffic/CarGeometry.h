/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_CARGEOMETRY_H
#define OPENCOVER_PLUGINS_TRAFFIC_CARGEOMETRY_H

#include <osg/BoundingBox>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <vector>

#include "Geometry.h"

class CarGeometry : public Geometry
{
public:
    CarGeometry(const std::string &name, const std::string &fileName, osg::Group *parentNode);
    ~CarGeometry();

    void setTransform(osg::Matrix transform);

protected:
    static osg::Node *loadFile(const std::string &file);

    void removeFromSceneGraph();

    osg::ref_ptr<osg::MatrixTransform> transformNode;
    osg::ref_ptr<osg::LOD> lodNode;
};

#endif
