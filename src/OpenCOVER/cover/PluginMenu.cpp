/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PluginMenu.h"
#include "coVRPluginList.h"
#include "coVRPluginSupport.h"
#include <cassert>

#include <ui/Button.h>
#include <ui/Menu.h>

namespace opencover
{
PluginMenu *PluginMenu::s_instance = NULL;

void PluginMenu::Plugin::add(ui::Menu *menu)
{
    button = new ui::Button(menu, name);
    button->setState(plugin != nullptr);
    button->setCallback([this](bool load){
        plugin = coVRPluginList::instance()->getPlugin(name.c_str());
        if (load)
        {
            if (!plugin)
                plugin = coVRPluginList::instance()->addPlugin(name.c_str());
        }
        else
        {
            if (plugin)
                coVRPluginList::instance()->unload(plugin);
            plugin = coVRPluginList::instance()->getPlugin(name.c_str());
        }
        button->setState(plugin != NULL);
    });
}

PluginMenu::PluginMenu()
: ui::Owner("PluginMenu", cover->ui)
{
    assert(!s_instance);
}

PluginMenu::~PluginMenu()
{
    if (s_instance)
        s_instance = NULL;
}

PluginMenu *PluginMenu::instance()
{
    if (!s_instance)
        s_instance = new PluginMenu();
    return s_instance;
}

void PluginMenu::init()
{
    menu = new ui::Menu("Plugins", this);
    menu->setText("Plug-Ins");

    for (size_t i = 0; i < items.size(); ++i)
    {
        if (!items[i].button)
        {
            Plugin &p = items[i];
            p.add(menu);
        }
    }
}

void PluginMenu::addEntry(const std::string &name, coVRPlugin *plugin)
{
    Plugin p(name);
    p.plugin = plugin;
    if (menu)
    {
        p.add(menu);
    }
    items.push_back(p);
}

void PluginMenu::updateState()
{
    for (size_t i = 0; i < items.size(); ++i)
    {
        items[i].plugin = coVRPluginList::instance()->getPlugin(items[i].name.c_str());
        if (items[i].button)
            items[i].button->setState(items[i].plugin != NULL);
    }
}

}
