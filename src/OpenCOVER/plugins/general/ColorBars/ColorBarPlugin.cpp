/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: ColorBar Plugin                                             **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "ColorBarPlugin.h"
#include <OpenVRUI/osg/mathUtils.h>
#include <PluginUtil/ColorBar.h>
#include <config/CoviseConfig.h>
#include <cover/OpenCOVER.h>
#include <cover/RenderObject.h>
#include <cover/coVRConfig.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/View.h>
#include <cover/ui/VruiView.h>
#include <grmsg/coGRMsg.h>
#include <osg/io_utils>

using namespace opencover;

ColorBarPlugin::ColorBarPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ColorBarPlugin", cover->ui)
{
}

bool ColorBarPlugin::init()
{
    //fprintf(stderr,"ColorBarPlugin::ColorBarPlugin\n");
    colorSubmenu = NULL;
    colorsModuleMap.clear();

    if (cover->visMenu)
    {
        colorSubmenu = new ui::Menu("Colors", this);
        cover->visMenu->add(colorSubmenu, ui::Container::KeepFirst); // after Execute
    }

    hudScale = covise::coCoviseConfig::getFloat("COVER.Plugin.ColorBar.HudScale", 0.5); // half screen height

    return true;
}

// this is called if the plugin is removed at runtime
ColorBarPlugin::~ColorBarPlugin()
{
    //fprintf(stderr,"ColorBarPlugin::~ColorBarPlugin\n");

    colorsModuleMap.clear();
}

void
ColorBarPlugin::removeObject(const char *container, bool replace)
{
    if (replace)
    {
        if (interactorMap.find(container) != interactorMap.end())
            removeQueue.push_back(container);
    }
    else
    {
        removeInteractor(container);
    }
}

void ColorBarPlugin::preFrame()
{
    for (auto it = visibleHuds.begin(); it != visibleHuds.end();)
    {
        if ((*it)->hudVisible())
            ++it;
        else
            it = visibleHuds.erase(it);
    }

    for (const auto &cm: colorsModuleMap)
    {
        auto &mod = cm.second;
        if (mod.hudVisible() && std::find(visibleHuds.begin(), visibleHuds.end(), &mod) == visibleHuds.end())
            visibleHuds.emplace_back(&mod);
    }

    osg::Vec3 bottomLeft, hpr, offset;
    if (coVRMSController::instance()->isMaster() && coVRConfig::instance()->numScreens() > 0) {
        const auto &s0 = coVRConfig::instance()->screens[0];
        hpr = s0.hpr;
        auto sz = osg::Vec3(s0.hsize, 0., s0.vsize);
        osg::Matrix mat;
        MAKE_EULER_MAT_VEC(mat, hpr);
        bottomLeft = s0.xyz - sz * mat * 0.5;
        auto minsize = std::min(s0.hsize, s0.vsize);
        bottomLeft += osg::Vec3(minsize, 0., minsize) * mat * 0.02;
        offset = osg::Vec3(s0.vsize/2.5, 0 , 0) * mat * hudScale;
    }
    for (int i=0; i<3; ++i)
    {
        coVRMSController::instance()->syncData(&bottomLeft[i], sizeof(bottomLeft[i]));
        coVRMSController::instance()->syncData(&hpr[i], sizeof(hpr[i]));
        coVRMSController::instance()->syncData(&offset[i], sizeof(offset[i]));
    }

    for (size_t i=0; i<visibleHuds.size(); ++i)
    {
        auto mod = visibleHuds[i];
        mod->colorbar->setHudPosition(bottomLeft, hpr, offset[0]/480);
        bottomLeft += offset;
    }
}

void
ColorBarPlugin::postFrame()
{
    for (size_t i=0; i<removeQueue.size(); ++i)
        removeInteractor(removeQueue[i]);
    removeQueue.clear();
}

void
ColorBarPlugin::removeInteractor(const std::string &container)
{
    InteractorMap::iterator it = interactorMap.find(container);
    if (it != interactorMap.end())
    {
        coInteractor *inter = it->second;

        for (ColorsModuleMap::iterator it2 = colorsModuleMap.begin();
             it2 != colorsModuleMap.end();
             ++it2)
        {
            if (it2->first->isSame(inter))
            {
                ColorsModule &mod = it2->second;
                --mod.useCount;
                if (mod.useCount == 0)
                {
                    if (mod.hudVisible())
                    {
                        for (auto vit = visibleHuds.begin(); vit != visibleHuds.end(); ++vit)
                        {
                            if (*vit == &mod)
                            {
                                visibleHuds.erase(vit);
                                break;
                            }
                        }
                    }
                    colorsModuleMap.erase(it2);
                    interactorMap.erase(it);
                    break;
                }
                else
                {
                    it->second = it2->first;
                    it->second->incRefCount();
                }
            }
        }
        inter->decRefCount();
    }
}

void ColorBarPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg)
{
    if(msg.getType() == grmsg::coGRMsg::COLOR_BAR_PLUGIN)
    {
        for(auto &c : colorsModuleMap)
        {
            static bool state = true;
            c.second.colorbar->show(state);
            state = !state;
        }
    }
};


void
ColorBarPlugin::newInteractor(const RenderObject *container, coInteractor *inter)
{
    if (strcmp(inter->getPluginName(), "ColorBars") != 0)
        return;

    inter->incRefCount();
    const char *containerName = container->getName();
    auto iit = interactorMap.find(containerName);
    if (iit != interactorMap.end())
    {
        auto oldInter = iit->second;
        oldInter->decRefCount();
        iit->second = inter;
    }
    else
    {
        interactorMap[containerName] = inter;
    }

    const char *colormapString = inter->getString(0); // Colormap string
    if (!colormapString)
    {
        // for Vistle: get from data object
        colormapString = container->getAttribute("COLORMAP");
    }

    // get the module name
    std::string moduleName = inter->getModuleName();
    int instance = inter->getModuleInstance();
    moduleName += "_" + std::to_string(instance);
    std::string host = inter->getModuleHost();
    moduleName += "@" + host;

    std::string menuName = moduleName;
    if (inter->getObject() && inter->getObject()->getAttribute("OBJECTNAME"))
        menuName = inter->getObject()->getAttribute("OBJECTNAME");
    if (container && container->getAttribute("OBJECTNAME"))
        menuName = container->getAttribute("OBJECTNAME");

    bool found = false;
    ColorsModuleMap::iterator it = colorsModuleMap.begin();
    for (; it != colorsModuleMap.end(); ++it)
    {
        if (it->first->isSame(inter))
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        it = colorsModuleMap.emplace(inter, ColorsModule(std::string(inter->getModuleName())+"_"+std::to_string(inter->getModuleInstance()), this)).first;
        inter->incRefCount();
        ColorsModule &mod = it->second;
        mod.menu = new ui::Menu(menuName, &mod);
        colorSubmenu->add(mod.menu);

        mod.colorbar = new ColorBar(mod.menu);
    }
    ColorsModule &mod = it->second;
    ++mod.useCount;
    mod.menu->setText(menuName);
    if (mod.colorbar)
    {
        mod.colorbar->addInter(inter);
        mod.colorbar->setName(menuName.c_str());
        if (colormapString)
            mod.colorbar->parseAttrib(colormapString);
    }
}

COVERPLUGIN(ColorBarPlugin)
