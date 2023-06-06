/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "InfoPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <cassert>

using namespace opencover;
using namespace vrui;

InfoPlugin::InfoPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, m_menuItem(NULL)
, m_menu(NULL)
{
    fprintf(stderr, "InfoPlugin::InfoPlugin\n");
}

// this is called if the plugin is removed at runtime
InfoPlugin::~InfoPlugin()
{
    fprintf(stderr, "InfoPlugin::~InfoPlugin\n");
    for (ItemMap::iterator it = m_itemMap.begin();
         it != m_itemMap.end();
         ++it)
    {
        delete it->second;
    }
    m_itemMap.clear();

    delete m_menu;
    delete m_menuItem;
}

void InfoPlugin::addObject(const RenderObject *container, osg::Group *root, const RenderObject *obj, const RenderObject *, const RenderObject *, const RenderObject *)
{
    (void)root;
    fprintf(stderr, "InfoPlugin::addObject\n");
    if (!container)
        return;
    const char *name = container->getName();
    if (!name)
        return;
    if (!obj)
        return;
    const char *text = obj->getAttribute("TEXT");
    if (!text)
        return;

    const char *title = obj->getAttribute("TEXT_TITLE");
    if (!title)
        title = "Info";

    if (!m_menuItem)
    {
        std::string t = title;
        t += "...";
        m_menuItem = new coSubMenuItem(t.c_str());
        cover->getMenu()->add(m_menuItem);
    }

    if (!m_menu)
    {
        m_menu = new coRowMenu(title);
        m_menuItem->setMenu(m_menu);
    }

    assert(m_itemMap.find(name) == m_itemMap.end());

    coLabelMenuItem *item = new coLabelMenuItem(text);
    m_itemMap[name] = item;
    m_menu->add(item);
}

void
InfoPlugin::removeObject(const char *objName, bool replace)
{
    fprintf(stderr, "InfoPlugin::removeObject(%s, %d)\n", objName, (int)replace);
    ItemMap::iterator it = m_itemMap.find(objName);
    if (it == m_itemMap.end())
        return;

    delete it->second;
    m_itemMap.erase(it);
}

COVERPLUGIN(InfoPlugin)
