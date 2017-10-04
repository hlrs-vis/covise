#include "Manager.h"
#include "View.h"
#include "Element.h"
#include "Group.h"

#include <cctype>
#include <cassert>
#include <iostream>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>

namespace opencover {
namespace ui {

Manager::Manager()
: Owner("Manager", this)
{
}

void Manager::add(Element *elem)
{
    m_newElements.push_back(elem);
}

bool Manager::update()
{
    while (!m_newElements.empty())
    {
        m_changed = true;
        auto elem = m_newElements.front();
        m_newElements.pop_front();
        m_elements.insert(elem);
        if (elem->parent())
        {
            auto p = elem->parent();
            p->add(elem);
            assert(elem->m_parent = p);
        }
        for (auto v: m_views)
        {
            v.second->elementFactory(elem);
        }
        elem->update();
    }

    bool ret = m_changed;
    m_changed = false;
    return ret;
}

void Manager::setChanged()
{
    m_changed = true;
}

bool Manager::addView(View *view)
{
    std::string name = view->name();
    auto it = m_views.find(name);
    if (it != m_views.end())
    {
        return false;
    }

    m_views.emplace(name, view);
    view->m_manager = this;

    for (auto e: m_elements)
    {
        view->elementFactory(e);
        e->update();
    }

    return false;
}

bool Manager::removeView(const View *view)
{
    std::string name = view->name();
    auto it = m_views.find(name);
    if (it != m_views.end())
    {
        if (it->second == view)
        {
            m_views.erase(it);
            return true;
        }
    }
    return false;
}

bool Manager::removeView(const std::string &name)
{
    auto it = m_views.find(name);
    if (it != m_views.end())
    {
        m_views.erase(it);
        return true;
    }
    return false;
}

void Manager::updateText(const Element *elem) const
{
    for (auto v: m_views)
    {
        v.second->updateText(elem);
    }
}

void Manager::updateEnabled(const Element *elem) const
{
    for (auto v: m_views)
    {
        v.second->updateEnabled(elem);
    }
}

void Manager::updateVisible(const Element *elem) const
{
    for (auto v: m_views)
    {
        v.second->updateVisible(elem);
    }
}

void Manager::updateState(const Button *button) const
{
    for (auto v: m_views)
    {
        v.second->updateState(button);
    }
}

void Manager::updateChildren(const Menu *menu) const
{
    for (auto v: m_views)
    {
        v.second->updateChildren(menu);
    }
}

void Manager::updateChildren(const SelectionList *sl) const
{
    for (auto v: m_views)
    {
        v.second->updateChildren(sl);
    }
}

void Manager::updateInteger(const Slider *slider) const
{
    for (auto v: m_views)
    {
        v.second->updateInteger(slider);
    }
}

void Manager::updateValue(const Slider *slider) const
{
    for (auto v: m_views)
    {
        v.second->updateValue(slider);
    }
}

void Manager::updateBounds(const Slider *slider) const
{
    for (auto v: m_views)
    {
        v.second->updateBounds(slider);
    }
}

bool Manager::keyEvent(int type, int keySym, int mod) const
{
    if (type != osgGA::GUIEventAdapter::KEYDOWN)
        return false;

    bool handled = false;

    bool alt = mod & osgGA::GUIEventAdapter::MODKEY_ALT;
    bool ctrl = mod & osgGA::GUIEventAdapter::MODKEY_CTRL;
    bool shift = mod & osgGA::GUIEventAdapter::MODKEY_SHIFT;
    bool meta = mod & osgGA::GUIEventAdapter::MODKEY_META;

    if (shift && std::isupper(keySym))
    {
        //std::cerr << "ui::Manager: mapping to lower" << std::endl;
        keySym = std::tolower(keySym);
    }
    std::cerr << "key: ";
    if (meta)
        std::cerr << "meta+";
    if (ctrl)
        std::cerr << "ctrl+";
    if (alt)
        std::cerr << "alt+";
    if (shift)
        std::cerr << "shift+";
    std::cerr << "'" << (char)keySym << "'" << std::endl;

    for (auto elem: m_elements)
    {
        if (!elem->hasShortcut())
            continue;

        auto m = elem->modifiers();
        if (bool(m & ModAlt) != alt)
            continue;
        if (bool(m & ModCtrl) != ctrl)
            continue;
        if (bool(m & ModShift) != shift)
            continue;
        if (bool(m & ModMeta) != meta)
            continue;

        if (elem->symbol() == keySym)
        {
            elem->shortcutTriggered();
            if (handled)
            {
                std::cerr << "ui::Manager: duplicate mapping for shortcut on " << elem->path() << std::endl;
            }
            handled = true;
        }
    }
    return handled;
}

}
}
