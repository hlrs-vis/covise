/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Lamure_NODE_PLUGIN_H
#define _Lamure_NODE_PLUGIN_H

#include <util/common.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
using namespace covise;
using namespace opencover;

#include <scm/core/math.h>

#include <config/CoviseConfig.h>
#include <util/coTypes.h>
#include "LamureDrawable.h"

class Lamure;
class management;


class LamurePlugin : public coVRPlugin
{
public:
    LamurePlugin();
    ~LamurePlugin();
    bool init();

    static LamurePlugin *instance() { return plugin; };

    static int SloadBVH(const char *filename, osg::Group *parent, const char *ck = "");
    int loadBVH(const char *filename, osg::Group *parent);
    static int SunloadBVH(const char *filename, const char *ck = "");
    int unloadBVH(const char *filename);

    // this will be called in PreFrame
    void preFrame();
    osg::ref_ptr<osg::Group> LamureGroup;
    osg::ref_ptr<LamureDrawable> drawable;


private:
    static LamurePlugin *plugin;
};
#endif
