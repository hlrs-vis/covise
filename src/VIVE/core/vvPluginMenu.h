/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLUGINMENU_H
#define PLUGINMENU_H

#include <string>
#include <vector>
#include "ui/Owner.h"

namespace vive
{
namespace ui
{
class Button;
class Menu;
}
}

namespace vive
{

class vvPlugin;

class vvPluginMenu: public ui::Owner
{
public:
    static vvPluginMenu *instance();
    ~vvPluginMenu();

    void updateState();
    void addEntry(const std::string &name, vvPlugin *plugin);
    void addEntry(const std::string &name);
    void init();

private:
    vvPluginMenu();
    void scanPlugins();

    struct Plugin
    {
        std::string name;
        ui::Button *button = nullptr;
        vvPlugin *plugin = nullptr;
        bool configured = false;

        Plugin(const std::string &name)
            : name(name)
        {
        }

        void add(ui::Menu *menu, bool onlyTui=false);
    };

    std::map<std::string, Plugin> items;
    ui::Menu *menu = nullptr;

    static vvPluginMenu *s_instance;
};
}
#endif
