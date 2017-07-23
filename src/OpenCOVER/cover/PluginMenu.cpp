/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include "PluginMenu.h"
#include "coVRPluginList.h"
#include "coVRPluginSupport.h"
#include "VRPinboard.h"

using namespace vrui;

namespace opencover
{
PluginMenu *PluginMenu::s_instance = NULL;

void PluginMenu::Plugin::add(coMenu *menu)
{
    this->menu.reset(new coCheckboxMenuItem(name.c_str(), plugin != NULL));
    menu->add(this->menu.get());
    this->menu->setMenuListener(PluginMenu::instance());
}

PluginMenu::PluginMenu()
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
    menu.reset(new coRowMenu("Plug-Ins", cover->getMenu()));

    pinboardEntry.reset(new coSubMenuItem("Plug-Ins..."));
    pinboardEntry->setMenu(menu.get());

    for (size_t i = 0; i < items.size(); ++i)
    {
        if (!items[i].menu)
        {
            Plugin &p = items[i];
            p.add(menu.get());
        }
    }

    if (!items.empty())
        cover->getMenu()->add(pinboardEntry.get());
}

void PluginMenu::addEntry(const std::string &name, coVRPlugin *plugin)
{
    Plugin p(name);
    p.plugin = plugin;
    if (menu)
    {
        p.add(menu.get());
    }
    items.push_back(p);

    if (!items.empty() && pinboardEntry)
        cover->getMenu()->add(pinboardEntry.get());
}

void PluginMenu::updateState()
{
    for (size_t i = 0; i < items.size(); ++i)
    {
        items[i].plugin = coVRPluginList::instance()->getPlugin(items[i].name.c_str());
        if (items[i].menu)
            items[i].menu->setState(items[i].plugin != NULL);
    }
}

void PluginMenu::menuEvent(coMenuItem *item)
{
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (items[i].menu.get() != item)
            continue;

        items[i].plugin = coVRPluginList::instance()->getPlugin(items[i].name.c_str());
        if (items[i].menu->getState())
        {
            if (!items[i].plugin)
                items[i].plugin = coVRPluginList::instance()->addPlugin(items[i].name.c_str());
        }
        else
        {
            if (items[i].plugin)
                coVRPluginList::instance()->unload(items[i].plugin);
            items[i].plugin = coVRPluginList::instance()->getPlugin(items[i].name.c_str());
        }
        items[i].menu->setState(items[i].plugin != NULL);
    }
}
}
