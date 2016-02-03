/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include <osg/Group>
#include "Scene.h"

class SceneConf : public Scene
{
public:
    SceneConf(void);
    ~SceneConf(void);
    osg::Group *getSceneGroup(int = 0)
    {
        return confGroup.get();
    };
    void updateScene(int num = 0);
    osg::Group *makeProjectorsGroup();

private:
    osg::Group *makeSceneConf();
    osg::ref_ptr<osg::Group> confGroup;
};
