/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>
#include "Renderer.h"

using namespace opencover;

class ANARIPlugin : public coVRPlugin
{

public:
    static ANARIPlugin *plugin;
    static ANARIPlugin *instance();

    static int loadScene(const char *fileName, osg::Group *loadParent, const char *);
    static int unloadScene(const char *fileName, const char *);

    ANARIPlugin();
   ~ANARIPlugin();

    bool init();

    void preDraw(osg::RenderInfo &info);

    void expandBoundingSphere(osg::BoundingSphere &bs);

private:
    Renderer::SP renderer = nullptr;

};


