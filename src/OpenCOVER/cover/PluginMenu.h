/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLUGINMENU_H
#define PLUGINMENU_H

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coSubMenuItem.h>

namespace vrui
{
class coCheckboxMenuItem;
}

namespace opencover
{

class coVRPlugin;

class PluginMenu : public vrui::coMenuListener
{
public:
    static PluginMenu *instance();

    void updateState();
    void addEntry(const std::string &name, coVRPlugin *plugin);
    void init();

    void menuEvent(vrui::coMenuItem *item);

private:
    PluginMenu();
    ~PluginMenu();

    struct Plugin
    {
        std::string name;
        boost::shared_ptr<vrui::coCheckboxMenuItem> menu;
        coVRPlugin *plugin;

        Plugin(const std::string &name)
            : name(name)
            , plugin(NULL)
        {
        }

        void add(vrui::coMenu *menu);
    };

    std::vector<Plugin> items;
    boost::shared_ptr<vrui::coSubMenuItem> pinboardEntry;
    boost::shared_ptr<vrui::coMenu> menu;

    static PluginMenu *s_instance;
};
}
#endif
