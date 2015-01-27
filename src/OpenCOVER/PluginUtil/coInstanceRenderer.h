/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_INSTANCERENDERER_H
#define CO_INSTANCERENDERER_H

/*! \file
\brief  OpenCOVER use OpenGL Instance Rendering to render huge amounts of trees and similar static objects

\author Uwe Woessner <woessner@hlrs.de>
\author (C) 2006
High Performance Computing Center Stuttgart,
Allmandring 30,
D-70550 Stuttgart,
Germany

\date   2006
*/

#include <util/common.h>
#include <osg/Geode>
namespace opencover
{
class PLUGIN_UTILEXPORT coInstanceObject
{
public:
    coInstanceObject(std::string textureName, float width, float height);
    virtual ~coInstanceObject();
    void addInstances(osg::Vec3Array &positions);
    osg::Geode *getGeode()
    {
        return geode.get();
    };

protected:
    void createTwoQuadGeometry(float width, float height);
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Geometry> geom;
    int numInstances;
    std::string textureName;
    float width, height;
    osg::StateSet *stateSet;
};
class PLUGIN_UTILEXPORT coInstanceRenderer
{
public:
    coInstanceRenderer();
    virtual ~coInstanceRenderer();
    static coInstanceRenderer *instance();

    void addInstances(osg::Vec3Array &positions, int type);
    int addObject(std::string textureName, float width, float height);

protected:
    int numObjects;
    std::vector<coInstanceObject *> objects;
    osg::ref_ptr<osg::Group> instanceObjects;

private:
    static coInstanceRenderer *instance_;
};
}
#endif
