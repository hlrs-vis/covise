/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneVis.h"

SceneVis::SceneVis(bool load)
    : Scene(load, true)
{
}

SceneVis::~SceneVis(void)
{
}

osg::Group *SceneVis::getSceneGroup(int num)
{
    osg::Group *sceneGroup = getProjector(num)->getVisScene()->getSceneGroup();
    return sceneGroup;
}

void SceneVis::updateScene(int num)
{
    getProjector(0)->getVisScene()->updateViewerPos();
}