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
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/OpenCOVER.h>
#include <PluginUtil/ColorBar.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>

using namespace opencover;

ColorBarPlugin::ColorBarPlugin()
: ui::Owner("ColorBarPlugin", cover->ui)
{
}

bool ColorBarPlugin::init()
{
    //fprintf(stderr,"ColorBarPlugin::ColorBarPlugin\n");
    colorSubmenu = NULL;
    colorsModuleMap.clear();
    tabID = 0;

    // create the TabletUI User-Interface
    createMenuEntry();

    if (cover->visMenu)
    {
        colorSubmenu = new ui::Menu("Colors", this);
        cover->visMenu->add(colorSubmenu);
    }

    return true;
}

// this is called if the plugin is removed at runtime
ColorBarPlugin::~ColorBarPlugin()
{
    //fprintf(stderr,"ColorBarPlugin::~ColorBarPlugin\n");

    colorsModuleMap.clear();

    removeMenuEntry();
}

void ColorBarPlugin::createMenuEntry()
{
    colorBarTab = new coTUITab("ColorBars", coVRTui::instance()->mainFolder->getID());
    colorBarTab->setPos(0, 0);
    colorBarTab->setEventListener(this);

    _tabFolder = new coTUITabFolder("Folder", colorBarTab->getID());
    _tabFolder->setPos(0, 0);

    tabID = _tabFolder->getID();
}

void ColorBarPlugin::removeMenuEntry()
{
    delete _tabFolder;
    delete colorBarTab;
}

void ColorBarPlugin::tabletPressEvent(coTUIElement *)
{
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
        interactorMap.erase(it);

        for (ColorsModuleMap::iterator it2 = colorsModuleMap.begin();
             it2 != colorsModuleMap.end();
             ++it2)
        {
            if (it2->first->isSame(inter))
            {
                --it2->second.useCount;
                if (it2->second.useCount == 0)
                {
                    it2->first->decRefCount();
                    colorsModuleMap.erase(it2);
                    break;
                }
            }
        }
        inter->decRefCount();
    }
}

void
ColorBarPlugin::newInteractor(const RenderObject *container, coInteractor *inter)
{
    if (strcmp(inter->getPluginName(), "ColorBars") == 0)
    {
        const char *containerName = container->getName();
        interactorMap[containerName] = inter;
        inter->incRefCount();

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
        }
        ColorsModule &mod = it->second;
        ++mod.useCount;

        const char *colormapString = NULL;
        colormapString = inter->getString(0); // Colormap string

        float min = 0.0;
        float max = 1.0;
        int numColors;
        float *r = NULL;
        float *g = NULL;
        float *b = NULL;
        float *a = NULL;
        char *species = NULL;
        if (colormapString)
        {
            ColorBar::parseAttrib(colormapString, species, min, max, numColors, r, g, b, a);
        }
        else
        {
            species = new char[16];
            strcpy(species, "NoColors");
            numColors = 2;
            min = 0.0;
            max = 1.0;
            r = new float[2];
            g = new float[2];
            b = new float[2];
            a = new float[2];
            r[0] = 0.0;
            g[0] = 0.0;
            b[0] = 0.0;
            a[0] = 1.0;
            r[1] = 1.0;
            g[1] = 1.0;
            b[1] = 1.0;
            a[1] = 1.0;
        }

        // get the module name
        std::string moduleName = inter->getModuleName();
        int instance = inter->getModuleInstance();
        std::string host = inter->getModuleHost();

        char buf[32];
        sprintf(buf, "_%d", instance);
        moduleName += buf;
        moduleName += "@" + host;

        std::string menuName = moduleName;
        if (inter->getObject() && inter->getObject()->getAttribute("OBJECTNAME"))
            menuName = inter->getObject()->getAttribute("OBJECTNAME");
        if (container && container->getAttribute("OBJECTNAME"))
            menuName = container->getAttribute("OBJECTNAME");

        if (found)
        {
            if (mod.colorbar)
            {
                mod.colorbar->update(species, min, max, numColors, r, g, b, a);
                mod.colorbar->setName(menuName.c_str());
            }
        }
        else
        {
            mod.menu = new ui::Menu(menuName, &mod);
            colorSubmenu->add(mod.menu);

            if (cover->vruiView)
            {
                auto menu = dynamic_cast<vrui::coRowMenu *>(cover->vruiView->getMenu(mod.menu));
                auto item = dynamic_cast<vrui::coSubMenuItem *>(cover->vruiView->getItem(mod.menu));
                if (menu && item)
                    mod.colorbar = new ColorBar(item, menu, menuName.c_str(), species, min, max, numColors, r, g, b, a, tabID);
            }
        }
        if (mod.colorbar)
            mod.colorbar->addInter(inter);

        delete[] species;
        delete[] r;
        delete[] g;
        delete[] b;
        delete[] a;
    }
}

COVERPLUGIN(ColorBarPlugin)
