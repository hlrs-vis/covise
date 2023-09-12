/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "ANARIPlugin.h"

#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <config/CoviseConfig.h>

ANARIPlugin *ANARIPlugin::plugin = nullptr;

static FileHandler handlers[] = {
    { NULL,
      ANARIPlugin::loadMesh,
      ANARIPlugin::unloadMesh,
      "obj" },
    { NULL,
      ANARIPlugin::loadMesh,
      ANARIPlugin::unloadMesh,
      "gltf" },
    { NULL,
      ANARIPlugin::loadMesh,
      ANARIPlugin::unloadMesh,
      "glb" },
    { NULL,
      ANARIPlugin::loadMesh,
      ANARIPlugin::unloadMesh,
      "ply" },
    { NULL,
      ANARIPlugin::loadVolumeRAW,
      ANARIPlugin::unloadVolumeRAW,
      "raw" },
    { NULL,
      ANARIPlugin::loadFLASH,
      ANARIPlugin::unloadFLASH,
      "hdf5" }
};

ANARIPlugin *ANARIPlugin::instance()
{
    return plugin;
}

int ANARIPlugin::loadMesh(const char *fileName, osg::Group *loadParent, const char *)
{
    if (plugin->renderer)
        plugin->renderer->loadMesh(fileName);

    return 1;
}

int ANARIPlugin::unloadMesh(const char *fileName, const char *)
{
    if (plugin->renderer)
        plugin->renderer->unloadMesh(fileName);

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

int ANARIPlugin::loadFLASH(const char *fileName, osg::Group *loadParent, const char *)
{
    if (plugin->renderer)
        plugin->renderer->loadFLASH(fileName);

    return 1;
}

int ANARIPlugin::unloadFLASH(const char *fileName, const char *)
{
    if (plugin->renderer)
        plugin->renderer->unloadFLASH(fileName);

    return 1;
}

ANARIPlugin::ANARIPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ANARI",cover->ui)
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

    // register file handlers
    int numHandlers = sizeof(handlers) / sizeof(handlers[0]);
    for (int i=0; i<numHandlers; ++i) {
        coVRFileManager::instance()->registerFileHandler(&handlers[i]);
    }

    // init menu
    anariMenu = new ui::Menu("ANARI", this);
    rendererMenu = new ui::Menu(anariMenu, "Renderer");
    rendererGroup = new ui::Group(rendererMenu, "Renderer");
    rendererButtonGroup = new ui::ButtonGroup(rendererGroup, "RendererGroup");

    std::vector<std::string> rendererTypes = renderer->getRendererTypes();
    rendererButtons.resize(rendererTypes.size());

    for (size_t i=0; i<rendererTypes.size(); ++i) {
        rendererButtons[i] = new ui::Button(rendererGroup,
                                            rendererTypes[i],
                                            rendererButtonGroup);
        rendererButtons[i]->setText(rendererTypes[i]);
        rendererButtons[i]->setCallback([=](bool state) {
            if (state) {
                renderer->setRendererType(rendererTypes[i]);
            }
        });
    }

    sppSlider = new ui::Slider(anariMenu, "SPP");
    sppSlider->setIntegral(true);
    sppSlider->setBounds(1, 16);
    sppSlider->setValue(1);
    sppSlider->setCallback([=](double value, bool /*released*/) {
        renderer->setPixelSamples((int)value);
    });

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
    } else if (geometry->isUnstructuredGrid() && colors) {
        int numCells, numIndices, numVerts;
        geometry->getSize(numCells, numIndices, numVerts);

        // Prefix into index array
        auto *cellIndex = geometry->getInt(Field::Elements);

        // Element indices
        auto *index = geometry->getInt(Field::Connections);

        // Element types (not needed)
        // auto *type = geometry->getInt(Field::Types);

        auto *X = geometry->getFloat(Field::X);
        auto *Y = geometry->getFloat(Field::Y);
        auto *Z = geometry->getFloat(Field::Z);

        auto *vertexData = colors->getFloat(Field::Red);

        if (cellIndex && index && X && Y && Z && vertexData) {
            // TODO: ANARI device(s) should support int32 indices!
            std::vector<uint64_t> cellIndex64(numCells);
            for (size_t i=0; i<numCells; ++i) {
                cellIndex64[i] = cellIndex[i];
            }

            // TODO: dito
            std::vector<uint64_t> index64(numIndices);
            for (size_t i=0; i<numIndices; ++i) {
                index64[i] = index[i];
            }

            float minValue = HUGE_VAL, maxValue = -HUGE_VAL;
            std::vector<float> vertexPosition(numVerts*3);
            for (size_t i=0; i<numVerts; ++i) {
                vertexPosition[i*3]   = X[i];
                vertexPosition[i*3+1] = Y[i];
                vertexPosition[i*3+2] = Z[i];

                minValue = std::min(minValue, vertexData[i]);
                maxValue = std::max(maxValue, vertexData[i]);
            }

            renderer->loadUMesh(vertexPosition.data(), cellIndex64.data(), index64.data(),
                                vertexData, numCells, numIndices, numVerts,
                                minValue, maxValue);
        }
    }
}

void ANARIPlugin::removeObject(const char *objName, bool replaceFlag)
{
    // NO!
}

COVERPLUGIN(ANARIPlugin)
