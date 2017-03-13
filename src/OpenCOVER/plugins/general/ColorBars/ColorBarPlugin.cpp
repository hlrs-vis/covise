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
#include <PluginUtil/ColorBar.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>

using namespace vrui;
using namespace opencover;

ColorBarPlugin::ColorBarPlugin()
{
}

bool ColorBarPlugin::init()
{
    //fprintf(stderr,"ColorBarPlugin::ColorBarPlugin\n");
    colorSubmenu = NULL;
    colorButton = NULL;
    colorbars.clear();
    tabID = 0;

    // create the TabletUI User-Interface
    createMenuEntry();

    // how to attach pinboardButton later to the Coviseentry
    coMenu *coviseMenu = NULL;
    VRMenu *menu = VRPinboard::instance()->namedMenu("COVISE");
    if (menu)
    {
        coviseMenu = menu->getCoMenu();

        // create the button for the pinboard
        colorButton = new coSubMenuItem("Colors...");
        colorSubmenu = new coRowMenu("Colors");
        colorButton->setMenu(colorSubmenu);

        coviseMenu->add(colorButton);
    }

    return true;
}

// this is called if the plugin is removed at runtime
ColorBarPlugin::~ColorBarPlugin()
{
    //fprintf(stderr,"ColorBarPlugin::~ColorBarPlugin\n");

    delete colorButton;
    delete colorSubmenu;
    for (std::map<std::string, ColorBar *>::iterator it = colorbars.begin();
         it != colorbars.end();
         ++it)
    {
        delete it->second;
    }
    colorbars.clear();

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
    if (replace) // partially handled by newInteractor
        return;

    std::map<std::string, ColorBar *>::iterator it = containerMap.find(container);
    if (it != containerMap.end())
    {
        ColorBar *colorbar = it->second;
        for (std::map<std::string, ColorBar *>::iterator it2 = colorbars.begin();
             it2 != colorbars.end();
             ++it2)
        {
            if (colorbar == it2->second)
            {
                colorbars.erase(it2);
                break;
            }
        }
        coSubMenuItem *item = menuMap[colorbar];
        menuMap.erase(colorbar);
        containerMap.erase(it);
        delete item;
        delete colorbar;
    }
}

void
ColorBarPlugin::newInteractor(const RenderObject *container, coInteractor *inter)
{
    if (strcmp(inter->getPluginName(), "ColorBars") == 0)
    {
        const char *containerName = container->getName();

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

        map<std::string, ColorBar *>::iterator it = colorbars.find(moduleName);
        opencover::ColorBar *colorBar = NULL;
        if (it != colorbars.end())
        {
            colorBar = it->second;
            colorBar->update(species, min, max, numColors, r, g, b, a);
            colorBar->setName(menuName.c_str());
        }
        else
        {
            vrui::coSubMenuItem *menuItem = new coSubMenuItem(menuName.c_str());
            _menu = new coRowMenu(menuName.c_str());
            menuItem->setMenu(_menu);

            if (colorSubmenu)
                colorSubmenu->add(menuItem);

            colorBar = new ColorBar(menuItem, _menu, menuName.c_str(), species, min, max, numColors, r, g, b, a, tabID);
            colorbars.insert(pair<std::string, ColorBar *>(moduleName, colorBar));
            menuMap[colorBar] = menuItem;
        }
        colorBar->addInter(inter);

        containerMap[containerName] = colorBar;

        delete[] species;
        delete[] r;
        delete[] g;
        delete[] b;
        delete[] a;
    }
}

COVERPLUGIN(ColorBarPlugin)
