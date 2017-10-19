#undef NDEBUG
#include "Manager.h"
#include "View.h"
#include "Element.h"
#include "Group.h"

#include <cctype>
#include <cassert>
#include <iostream>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>

#include <cover/coVRMSController.h>
#include <net/tokenbuffer.h>
#include <net/message.h>

namespace opencover {
namespace ui {

Manager::Manager()
: Owner("Manager", this)
{
}

void Manager::add(Element *elem)
{
    m_newElements.push_back(elem);

    elem->m_id = m_numCreated;
    m_elementsById.emplace(elem->elementId(), elem);
    m_elementsByPath.emplace(elem->path(), elem);
    ++m_numCreated;
}

void Manager::remove(Owner *owner)
{
    owner->clearItems();
}

void Manager::remove(Element *elem)
{
    //std::cerr << "DESTROY: " << elem->path() << std::endl;

    elem->clearItems();

    for (auto v: m_views)
    {
        v.second->removeElement(elem);
    }

    {
        auto it = m_elementsById.find(elem->elementId());
        if (it != m_elementsById.end())
        {
            m_elementsById.erase(it);
        }
    }

    {
        auto it = m_elementsByPath.find(elem->path());
        if (it != m_elementsByPath.end())
        {
            m_elementsByPath.erase(it);
        }
    }
}

Element *Manager::getById(int id) const
{
    auto it = m_elementsById.find(id);
    if (it == m_elementsById.end())
        return nullptr;

    return it->second;
}

Element *Manager::getByPath(const std::string &path) const
{
    auto it = m_elementsByPath.find(path);
    if (it == m_elementsByPath.end())
        return nullptr;

    return it->second;
}


bool Manager::update()
{
    while (!m_newElements.empty())
    {
        m_changed = true;
        auto elem = m_newElements.front();
        m_newElements.pop_front();
        m_elements.emplace(elem);

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

    for (auto elem: m_elements)
    {
        view->elementFactory(elem);
        elem->update();
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

void Manager::updateParent(const Element *elem) const
{
    for (auto v: m_views)
    {
        v.second->updateParent(elem);
    }
}

void Manager::updateChildren(const SelectionList *sl) const
{
    for (auto v: m_views)
    {
        v.second->updateChildren(sl);
    }
}

void Manager::updateScale(const Slider *slider) const
{
    for (auto v: m_views)
    {
        v.second->updateScale(slider);
    }
}

void Manager::updateIntegral(const Slider *slider) const
{
    for (auto v: m_views)
    {
        v.second->updateIntegral(slider);
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
    int modifiers = 0;
    std::cerr << "key: ";
    if (meta)
    {
        modifiers |= ModMeta;
        std::cerr << "meta+";
    }
    if (ctrl)
    {
        modifiers |= ModCtrl;
        std::cerr << "ctrl+";
    }
    if (alt)
    {
        modifiers |= ModAlt;
        std::cerr << "alt+";
    }
    if (shift)
    {
        modifiers |= ModShift;
        std::cerr << "shift+";
    }
    std::cerr << "'" << (char)keySym << "'" << std::endl;

    for (auto elem: m_elements)
    {
        if (elem->matchShortcut(modifiers, keySym))
        {
            elem->shortcutTriggered();
            if (handled)
            {
                std::cerr << "ui::Manager: duplicate mapping for shortcut on " << elem->path() << std::endl;
            }
            handled = true;
            continue;
        }
    }

    return handled;
}

void Manager::flushUpdates()
{
    if (!m_updates)
    {
        assert(m_numUpdates == 0);
        m_updates.reset(new covise::TokenBuffer);
    }

    for (const auto &state: m_elemState)
    {
        auto id = state.first;
        auto tb = state.second;

        *m_updates << id;
        *m_updates << false; // trigger
        *m_updates << *tb;

        ++m_numUpdates;
    }
    m_elemState.clear();
}

void Manager::queueUpdate(const Element *elem, bool trigger)
{
    if (elem->elementId() < 0)
    {
        assert(!trigger);
        return;
    }

    assert(elem->elementId() >= 0);

    auto it = m_elemState.find(elem->elementId());
    if (trigger)
    {
        if (it != m_elemState.end())
            m_elemState.erase(it);
        flushUpdates();

        *m_updates << elem->elementId();
        *m_updates << trigger;
        elem->save(*m_updates);

        ++m_numUpdates;
    }
    else
    {
        if (it == m_elemState.end())
        {
            it = m_elemState.emplace(elem->elementId(), std::make_shared<covise::TokenBuffer>()).first;
        }
        else
        {
            it->second->reset();
        }
        elem->save(*it->second);
    }
}

void Manager::processUpdates(std::shared_ptr<covise::TokenBuffer> updates, int numUpdates, bool runTriggers)
{
    updates->rewind();

    for (int i=0; i<numUpdates; ++i)
    {
        //std::cerr << "processing " << i << std::flush;
        int id = -1;
        *updates >> id;
        bool trigger = false;
        *updates >> trigger;
        auto elem = getById(id);
        //std::cerr << ": id=" << id << ", trigger=" << trigger << std::endl;
        assert(elem);
        elem->load(*updates);
        elem->update();
        if (trigger && runTriggers)
            elem->triggerImplementation();
    }
}

void Manager::sync()
{
    flushUpdates();
    auto ms = coVRMSController::instance();
    if (ms->isCluster())
    {
        ms->syncData(&m_numUpdates, sizeof(m_numUpdates));
        if (m_numUpdates > 0)
        {
            //std::cerr << "ui::Manager: syncing " << m_numUpdates << " updates" << std::endl;
            if (ms->isMaster())
            {
                covise::Message msg(*m_updates);
                coVRMSController::instance()->sendSlaves(&msg);
            }
            else
            {
                covise::Message msg;
                coVRMSController::instance()->readMaster(&msg);
                m_updates.reset(new covise::TokenBuffer(&msg));
            }
        }
    }

    int round = 0;
    while (m_numUpdates > 0)
    {
        //std::cerr << "ui::Manager: processing " << m_numUpdates << " updates in round " << round << std::endl;
        std::shared_ptr<covise::TokenBuffer> updates = m_updates;
        m_updates.reset(new covise::TokenBuffer);
        int numUpdates = m_numUpdates;
        m_numUpdates = 0;

        if (round > 2)
            break;
        processUpdates(updates, numUpdates, round<1);
        ++round;
    }

    //assert(m_numUpdates == 0);
}

}
}
