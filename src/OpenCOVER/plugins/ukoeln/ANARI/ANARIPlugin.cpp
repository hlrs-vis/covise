/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "ANARIPlugin.h"

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <config/CoviseConfig.h>

ANARIPlugin *ANARIPlugin::plugin = nullptr;

static FileHandler handlers[] = {
    { NULL,
      ANARIPlugin::loadScene,
      ANARIPlugin::unloadScene,
      "obj" },
    { NULL,
      ANARIPlugin::loadScene,
      ANARIPlugin::unloadScene,
      "gltf" },
    { NULL,
      ANARIPlugin::loadScene,
      ANARIPlugin::unloadScene,
      "glb" },
    { NULL,
      ANARIPlugin::loadScene,
      ANARIPlugin::unloadScene,
      "ply" },
    { NULL,
      ANARIPlugin::loadVolumeRAW,
      ANARIPlugin::unloadVolumeRAW,
      "raw" }
};

ANARIPlugin *ANARIPlugin::instance()
{
    return plugin;
}

int ANARIPlugin::loadScene(const char *fileName, osg::Group *loadParent, const char *)
{
    if (plugin->renderer)
        plugin->renderer->loadScene(fileName);

    return 1;
}

int ANARIPlugin::unloadScene(const char *fileName, const char *)
{
    if (plugin->renderer)
        plugin->renderer->unloadScene(fileName);

    return 1;
}

int ANARIPlugin::loadVolumeRAW(const char *fileName, osg::Group *loadParent, const char *)
{
    if (plugin->renderer)
        plugin->renderer->loadVolumeRAW(fileName);

    return 1;
}

int ANARIPlugin::unloadVolumeRAW(const char *fileName, const char *)
{
    if (plugin->renderer)
        plugin->renderer->unloadVolumeRAW(fileName);

    return 1;
}

ANARIPlugin::ANARIPlugin()
{
}

ANARIPlugin::~ANARIPlugin()
{
    int numHandlers = sizeof(handlers) / sizeof(handlers[0]);
    for (int i=0; i<numHandlers; ++i) {
        coVRFileManager::instance()->unregisterFileHandler(&handlers[i]);
    }
}

bool ANARIPlugin::init()
{
    if (cover->debugLevel(1)) fprintf(stderr, "\n    ANARIPlugin::init\n");

    if (plugin) return false;

    plugin = this;

    renderer = std::make_shared<Renderer>();

    //register file handler
    int numHandlers = sizeof(handlers) / sizeof(handlers[0]);
    for (int i=0; i<numHandlers; ++i) {
        coVRFileManager::instance()->registerFileHandler(&handlers[i]);
    }

    return true;
}

void ANARIPlugin::preDraw(osg::RenderInfo &info)
{
    if (!renderer)
        return;

    renderer->renderFrame(info);
}

void ANARIPlugin::expandBoundingSphere(osg::BoundingSphere &bs)
{
    if (!renderer)
        return;

    renderer->expandBoundingSphere(bs);
}

void ANARIPlugin::addObject(const RenderObject *container, osg::Group *parent,
                            const RenderObject *geometry, const RenderObject *normals,
                            const RenderObject *colors, const RenderObject *texture)
{
    (void)container;
    (void)parent;
    (void)normals;
    (void)texture;

    if (geometry->isUniformGrid()) {
        int sizeX, sizeY, sizeZ;
        geometry->getSize(sizeX, sizeY, sizeZ);
        const uint8_t *byteData = colors->getByte(Field::Byte);

        if (sizeX && sizeY && sizeZ && byteData) {
            renderer->loadVolume(byteData, sizeX, sizeY, sizeZ, 1,
                                 colors->getMin(0), colors->getMax(0));
        }
    }
}

void ANARIPlugin::removeObject(const char *objName, bool replaceFlag)
{
    // NO!
}

COVERPLUGIN(ANARIPlugin)