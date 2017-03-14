/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COLORBAR_PLUGIN_H
#define _COLORBAR_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: ColorBar Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coInteractor.h>
#include <cover/coVRPlugin.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <map>

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>
#include <cover/coVRTui.h>
#include <PluginUtil/ColorBar.h>

namespace vrui
{
class coRowMenu;
}

class ColorBarPlugin : public opencover::coVRPlugin, public opencover::coTUIListener
{
public:
    ColorBarPlugin();
    ~ColorBarPlugin();
    bool init();

    // this will be called if an object with feedback arrives
    void newInteractor(const opencover::RenderObject *, opencover::coInteractor *i);
    void removeObject(const char *container, bool replace);

private:
    void tabletPressEvent(opencover::coTUIElement *);
    void createMenuEntry();
    void removeMenuEntry();

    /// The TabletUI Interface
    opencover::coTUITab *colorBarTab;
    opencover::coTUITabFolder *_tabFolder;
    int tabID;

    // VR Menu
    vrui::coSubMenuItem *colorButton;
    vrui::coRowMenu *colorSubmenu;
    vrui::coRowMenu *_menu;
    vrui::coMenu *coviseMenu;
    typedef std::map<std::string, opencover::coInteractor *> InteractorMap;
    InteractorMap interactorMap; // from container to interactor
    struct ColorsModule
    {
        ColorsModule()
            : useCount(0)
            , colorbar(NULL)
            , menu(NULL)
        {}
        ~ColorsModule()
        {
            delete colorbar;
            delete menu;
        }

        int useCount;
        opencover::ColorBar *colorbar;
        vrui::coSubMenuItem *menu;
    };
    typedef std::map<opencover::coInteractor *, ColorsModule> ColorsModuleMap;
    ColorsModuleMap colorsModuleMap;
};
#endif
