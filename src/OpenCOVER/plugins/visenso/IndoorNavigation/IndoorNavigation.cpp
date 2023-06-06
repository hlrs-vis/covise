/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2013 Visenso  **
 **                                                                        **
 ** Description: Indoor Navigation (HSG-IMIT)                              **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
\****************************************************************************/

#include "IndoorNavigation.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

#include <osg/ClipPlane>

IndoorNavigation *IndoorNavigation::plugin = NULL;

//
// Constructor
//
IndoorNavigation::IndoorNavigation()
: coVRPlugin(COVER_PLUGIN_NAME)
, path(NULL)
, avatar(NULL)
, pluginBaseNode(NULL)
, animationSeconds(0.0f)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nIndoorNavigation::IndoorNavigation\n");

    speed = coCoviseConfig::getFloat("COVER.Plugin.IndoorNavigation.Speed", 1.0f);
    clipOffset = coCoviseConfig::getFloat("COVER.Plugin.IndoorNavigation.ClipOffset", 0.0f);
    filename = coCoviseConfig::getEntry("COVER.Plugin.IndoorNavigation.Filename");
}

//
// Destructor
//
IndoorNavigation::~IndoorNavigation()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nIndoorNavigation::~IndoorNavigation\n");

    if (avatar)
        pluginBaseNode->removeChild(avatar.get());
    if (path)
        pluginBaseNode->removeChild(path.get());
    if (pluginBaseNode)
        cover->getObjectsRoot()->removeChild(pluginBaseNode.get());
}

//
// INIT
//
bool IndoorNavigation::init()
{
    if (plugin)
        return false;
    if (cover->debugLevel(3))
        fprintf(stderr, "\nIndoorNavigation::IndoorNavigation\n");

    // set plugin
    IndoorNavigation::plugin = this;

    pluginBaseNode = new osg::Group();
    cover->getObjectsRoot()->addChild(pluginBaseNode.get());

    // path
    path = new Path();
    pluginBaseNode->addChild(path);
    path->loadFromFile(filename);

    // avatar
    avatar = new Avatar();
    pluginBaseNode->addChild(avatar);

    return true;
}

void IndoorNavigation::preFrame()
{
    animationSeconds += cover->frameDuration() * speed;
    path->update(animationSeconds);
    avatar->update(path->getCurrentPosition(), path->getCurrentOrientation());

    if (cover->getObjectsRoot()->getNumClipPlanes() > 0)
    {
        osg::Plane plane = osg::Plane(osg::Vec3(0.0f, 0.0f, -1.0f), path->getCurrentPosition() + osg::Vec3(0.0f, 0.0f, clipOffset));
        cover->getObjectsRoot()->getClipPlane(0)->setClipPlane(plane);
    }
}

void IndoorNavigation::key(int, int keySym, int)
{
    if (keySym == 'x')
    {
        animationSeconds = 0.0f;
    }
    if (keySym == '4')
        path->changeTranslation(osg::Vec3(0.1f, 0.0f, 0.0f));
    if (keySym == '6')
        path->changeTranslation(osg::Vec3(-0.1f, 0.0f, 0.0f));
    if (keySym == '8')
        path->changeTranslation(osg::Vec3(0.0f, -0.1f, 0.0f));
    if (keySym == '2')
        path->changeTranslation(osg::Vec3(0.0f, 0.1f, 0.0f));
    if (keySym == '9')
        path->changeTranslation(osg::Vec3(0.0f, 0.0f, 0.05f));
    if (keySym == '3')
        path->changeTranslation(osg::Vec3(0.0f, 0.0f, -0.05f));
    if (keySym == '7')
        path->changeRotation(0.01f);
    if (keySym == '1')
        path->changeRotation(-0.01f);
    if (keySym == 'b')
        path->changeStartDrawing(-10);
    if (keySym == 'n')
        path->changeStartDrawing(10);
}

COVERPLUGIN(IndoorNavigation)
