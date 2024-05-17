/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <cover/ui/Owner.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>
#include "Renderer.h"

namespace opencover {
namespace ui {
class Element;
class Group;
class Menu;
class Button;
class ButtonGroup;
class Slider;
}
}

using namespace opencover;

class ANARIPlugin : public coVRPlugin, public ui::Owner
{

public:
    static ANARIPlugin *plugin;
    static ANARIPlugin *instance();

    static int loadMesh(const char *fileName, osg::Group *loadParent, const char *);
    static int unloadMesh(const char *fileName, const char *);

    static int loadVolumeRAW(const char *fileName, osg::Group *loadParent, const char *);
    static int unloadVolumeRAW(const char *fileName, const char *);

    static int loadFLASH(const char *fileName, osg::Group *loadParent, const char *);
    static int unloadFLASH(const char *fileName, const char *);

    static int loadUMeshVTK(const char *fileName, osg::Group *loadParent, const char *);
    static int unloadUMeshVTK(const char *fileName, const char *);

    static int loadPointCloud(const char *fileName, osg::Group *loadParent, const char *);
    static int unloadPointCloud(const char *fileName, const char *);


    ANARIPlugin();
   ~ANARIPlugin();

    bool init() override;

    void preFrame() override;

    void expandBoundingSphere(osg::BoundingSphere &bs) override;

    void addObject(const RenderObject *container, osg::Group *parent,
                   const RenderObject *geometry, const RenderObject *normals,
                   const RenderObject *colors, const RenderObject *texture) override;

    void removeObject(const char *objName, bool replaceFlag) override;

protected:
    ui::Menu *anariMenu = nullptr;
    ui::Menu *rendererMenu = nullptr;
    ui::Group *rendererGroup = nullptr;
    ui::ButtonGroup *rendererButtonGroup = nullptr;
    std::vector<ui::Button *> rendererButtons;

    std::vector<ui::Element *> rendererUI;
    int previousRendererType = -1;
    int rendererType = 0;

private:
    Renderer::SP renderer = nullptr;

    void buildUI();
    void tearDownUI();
};


