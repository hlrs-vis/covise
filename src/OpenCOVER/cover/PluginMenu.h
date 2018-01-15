/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLUGINMENU_H
#define PLUGINMENU_H

#include <string>
#include <vector>
#include "ui/Owner.h"

namespace opencover
{
namespace ui
{
class Button;
class Menu;
}
}

namespace opencover
{

class coVRPlugin;

class PluginMenu: public ui::Owner
{
public:
    static PluginMenu *instance();

    void updateState();
    void addEntry(const std::string &name, coVRPlugin *plugin);
    void addEntry(const std::string &name);
    void init();

private:
    PluginMenu();
    ~PluginMenu();
    void scanPlugins();

    struct Plugin
    {
        std::string name;
        ui::Button *button = nullptr;
        coVRPlugin *plugin = nullptr;
        bool configured = false;

        Plugin(const std::string &name)
            : name(name)
        {
        }

        void add(ui::Menu *menu, bool onlyTui=false);
    };

    std::map<std::string, Plugin> items;
    ui::Menu *menu = nullptr;

    static PluginMenu *s_instance;
};
}
#endif
