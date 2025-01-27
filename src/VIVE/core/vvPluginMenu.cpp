/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/string_util.h>
#include <util/coFileUtil.h>
#include <util/environment.h>
#include "vvPluginMenu.h"
#include "vvPluginList.h"
#include "vvPluginSupport.h"
#include <cassert>

#include <ui/Button.h>
#include <ui/Menu.h>
#include <vsg/io/Options.h>

using covise::coDirectory;

namespace vive
{
vvPluginMenu *vvPluginMenu::s_instance = NULL;

void vvPluginMenu::Plugin::add(ui::Menu *menu, bool onlyTui)
{
    button = new ui::Button(menu, name);
    if (!configured)
        button->setPriority(ui::Element::Low);
    button->setState(plugin != nullptr);
	button->setShared(false);
    button->setCallback([this](bool load){
        plugin = vvPluginList::instance()->getPlugin(name.c_str());
        if (load)
        {
            if (!plugin)
                plugin = vvPluginList::instance()->addPlugin(name.c_str());
        }
        else
        {
            if (plugin)
                vvPluginList::instance()->unload(plugin);
            plugin = vvPluginList::instance()->getPlugin(name.c_str());
        }
        button->setState(plugin != NULL);
    });
}

vvPluginMenu::vvPluginMenu()
: ui::Owner("vvPluginMenu", vv->ui)
{
    assert(!s_instance);
    s_instance = this;
}

vvPluginMenu::~vvPluginMenu()
{
    assert(s_instance);
    s_instance = NULL;
}

vvPluginMenu *vvPluginMenu::instance()
{
    if (!s_instance)
        s_instance = new vvPluginMenu();
    return s_instance;
}

void vvPluginMenu::init()
{
    menu = new ui::Menu("Plugins", this);
    menu->setText("Plug-Ins");
    menu->allowRelayout(true);

    scanPlugins();

    for (auto &item: items)
    {
        if (!item.second.button)
        {
            Plugin &p = item.second;
            p.add(menu);
        }
    }
}

void vvPluginMenu::addEntry(const std::string &name, vvPlugin *plugin)
{
    Plugin p(name);
    p.plugin = plugin;
    auto res = items.insert(std::make_pair(name, p));
    res.first->second.configured = true;
    if (res.second && menu)
    {
        res.first->second.add(menu);
    }
}

void vvPluginMenu::addEntry(const std::string &name)
{
    Plugin p(name);
    items.insert(std::make_pair(name, p));
}

void vvPluginMenu::updateState()
{
    for (auto &item: items)
    {
        item.second.plugin = vvPluginList::instance()->getPlugin(item.second.name.c_str());
        if (item.second.button)
            item.second.button->setState(item.second.plugin != NULL);
    }
}

void vvPluginMenu::scanPlugins()
{
    const char *coviseDir = getenv("COVISEDIR");
    const char *archsuffix = getenv("ARCHSUFFIX");
    const char *covisepath = getenv("COVISE_PATH");
    char buf[1024];
    int n = 0;
    std::vector<std::string> paths;
#ifdef __APPLE__
    std::string bundlepath = covise::getBundlePath();
    if (!bundlepath.empty())
    {
        sprintf(buf, "%s/Contents/PlugIns/", bundlepath.c_str());
        paths.push_back(buf);
    }
#endif
    if (covisepath && archsuffix)
    {
#ifdef WIN32
        std::vector<std::string> p = split(covisepath, ';');
#else
        std::vector<std::string> p = split(covisepath, ':');
#endif
        for (std::vector<std::string>::iterator it = p.begin(); it != p.end(); ++it)
        {
#ifdef WIN32
            sprintf(buf, "%s\\%s\\lib\\VIVE\\plugins", it->c_str(), archsuffix);
#else
            sprintf(buf, "%s/%s/lib/VIVE/plugins", it->c_str(), archsuffix);
#endif
            paths.push_back(buf);
        }
    }
    else if ((coviseDir != NULL) && (archsuffix != NULL))
    {
#ifdef WIN32
        sprintf(buf, "%s\\%s\\lib\\VIVE\\plugins", coviseDir, archsuffix);
#else
        sprintf(buf, "%s/%s/lib/VIVE/plugins", coviseDir, archsuffix);
#endif
        paths.push_back(buf);
    }

    for (std::vector<std::string>::iterator it = paths.begin(); it != paths.end(); ++it)
    {
        std::unique_ptr<coDirectory> dir(coDirectory::open(it->c_str()));
        for (int i = 0; dir && i < dir->count(); i++)
        {
            if (dir->match(dir->name(i),
#ifdef WIN32
                           "*.dll"
#elif defined(__APPLE__)
                           "*.so"
#else
                           "*.so"
#endif
                           ))
#ifdef __APPLE__
                if (!dir->match(dir->name(i), "*.?.?.so"))
#endif
                {
                    if (!strstr(dir->name(i), "input_"))
                    {
                        std::string name = dir->name(i);
                        if (name.substr(0, 3)=="lib")
                            name = name.substr(3);
                        auto idx = name.find(".");
                        if (idx != std::string::npos)
                            name = name.substr(0, idx);
                        addEntry(name);
                        n++;
                    }
                }
        }
    }
    updateState();
}

}
