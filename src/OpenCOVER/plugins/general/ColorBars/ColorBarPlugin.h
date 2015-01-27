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
#include <map>

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>
#include <cover/coVRTui.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
}

namespace opencover
{
class ColorBar;
}

class ColorBarPlugin : public opencover::coVRPlugin, public opencover::coTUIListener
{
public:
    ColorBarPlugin();
    ~ColorBarPlugin();
    bool init();

    // this will be called if an object with feedback arrives
    void newInteractor(opencover::RenderObject *, opencover::coInteractor *i);
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
    std::map<std::string, opencover::ColorBar *> colorbars; // from name to colorbar
    std::map<std::string, opencover::ColorBar *> containerMap; // from container to colorbar
    std::map<opencover::ColorBar *, vrui::coSubMenuItem *> menuMap; // from color to submenu item
};
#endif
