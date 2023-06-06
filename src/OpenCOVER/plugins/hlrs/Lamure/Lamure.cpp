/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Lamure.h"
#include "LamureDrawable.h"
#include "management.h"
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRConfig.h>

#include <lamure/types.h>

#include <lamure/ren/config.h>
#include <lamure/ren/model_database.h>
#include <lamure/ren/cut_database.h>
#include <lamure/ren/dataset.h>
#include <lamure/ren/policy.h>

#include <lamure/pvs/pvs_database.h>


LamurePlugin *LamurePlugin::plugin = NULL;

static const int NUM_HANDLERS = 1;

static const FileHandler handlers[] = {
    { NULL,
      LamurePlugin::SloadBVH,
      LamurePlugin::SunloadBVH,
      "bvh" }
};



LamurePlugin::LamurePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "LamurePlugin::LamurePlugin\n");

    plugin = this;

    LamureGroup = new osg::Group();
    LamureGroup->setName("Lamure");
    osg::Geode *geode = new osg::Geode();
    geode->setName("LamureGeode");
    drawable = new LamureDrawable();
    geode->addDrawable(drawable.get());
    LamureGroup->addChild(geode);

    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->registerFileHandler(&handlers[index]);
}

// this is called if the plugin is removed at runtime
LamurePlugin::~LamurePlugin()
{
    fprintf(stderr, "LamurePlugin::~LamurePlugin\n");
    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->unregisterFileHandler(&handlers[index]);

    cover->getObjectsRoot()->removeChild(LamureGroup);
}


bool LamurePlugin::init()
{
    cover->getObjectsRoot()->addChild(LamureGroup);

    
    return true;
}

void
LamurePlugin::preFrame()
{
}

int LamurePlugin::SunloadBVH(const char *filename, const char *)
{
    return plugin->unloadBVH(filename);
}

int LamurePlugin::unloadBVH(const char *name)
{
    return 1;
}

int LamurePlugin::SloadBVH(const char *filename, osg::Group *parent, const char *)
{

    if (filename)
    {

        plugin->loadBVH(filename, parent);
    }

    return 0;
}

int LamurePlugin::loadBVH(const char *filename, osg::Group *)
{
    return drawable->loadBVH(filename);
}

COVERPLUGIN(LamurePlugin)
