/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Wuerfel OpenCOVER Plugin (draws a cube)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Wuerfel.h"
#include <cover/coVRPluginSupport.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>
using namespace opencover;
Wuerfel::Wuerfel()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "Wuerfel World\n");

    osg::Box *unitCube = new osg::Box(osg::Vec3(0, 0, 0), 1000.0f);
    osg::ShapeDrawable *unitCubeDrawable = new osg::ShapeDrawable(unitCube);

    // Declare a instance of the geode class:
    basicShapesGeode = new osg::Geode();
    basicShapesGeode->setName("Wuerfel");

    osg::Vec4 _color;
    _color.set(0.0, 0.0, 1.0, 1.0);
    unitCubeDrawable->setColor(_color);
    unitCubeDrawable->setUseDisplayList(false);

    // Add the unit cube drawable to the geode:
    basicShapesGeode->addDrawable(unitCubeDrawable);

    cover->getObjectsRoot()->addChild(basicShapesGeode.get());
}

bool Wuerfel::destroy()
{
    cover->getObjectsRoot()->removeChild(basicShapesGeode.get());
    return true;
}

// this is called if the plugin is removed at runtime
Wuerfel::~Wuerfel()
{
    fprintf(stderr, "Goodbye\n");
}

COVERPLUGIN(Wuerfel)
