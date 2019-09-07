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
#include <map>

#include <PluginUtil/ColorBar.h>

#include <cover/ui/Owner.h>

namespace opencover
{
namespace ui
{
class Menu;
}
}

class ColorBarPlugin: public opencover::coVRPlugin, public opencover::ui::Owner, public opencover::coTUIListener
{
public:
    ColorBarPlugin();
    ~ColorBarPlugin();
    bool init() override;

    // this will be called if an object with feedback arrives
    void newInteractor(const opencover::RenderObject *, opencover::coInteractor *i) override;
    void removeObject(const char *container, bool replace) override;
    void preFrame() override;
    void postFrame() override;

private:
    void removeInteractor(const std::string &container);
    std::vector<std::string> removeQueue;

    // VR Menu
    opencover::ui::Menu *colorSubmenu = nullptr;
    opencover::ui::Menu *_menu = nullptr;
    typedef std::map<std::string, opencover::coInteractor *> InteractorMap;
    InteractorMap interactorMap; // from container to interactor
    struct ColorsModule: public opencover::ui::Owner
    {
        ColorsModule(const std::string &name, opencover::ui::Owner *owner)
            : opencover::ui::Owner(name, owner)
            , useCount(0)
            , colorbar(NULL)
            , menu(NULL)
        {}
        ~ColorsModule()
        {
            delete colorbar;
        }

        bool hudVisible() const
        {
            return colorbar && colorbar->hudVisible();
        }

        int useCount;
        opencover::ColorBar *colorbar;
        opencover::ui::Menu *menu;
    };
    typedef std::map<opencover::coInteractor *, ColorsModule> ColorsModuleMap;
    ColorsModuleMap colorsModuleMap;
    std::vector<const ColorsModule *> visibleHuds;
};
#endif
