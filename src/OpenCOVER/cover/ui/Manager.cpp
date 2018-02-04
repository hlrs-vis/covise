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

#include <OpenVRUI/coMouseButtonInteraction.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <net/tokenbuffer.h>
#include <net/message.h>

namespace opencover {
namespace ui {

Manager::Manager()
: Owner("Manager", this)
{
    m_wheelInteraction = new vrui::coMouseButtonInteraction(vrui::coInteraction::Wheel, "MouseWheel", vrui::coInteraction::Low);
    vrui::coInteractionManager::the()->registerInteraction(m_wheelInteraction);
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
    auto it = m_elements.find(elem->m_order);
    if (it != m_elements.end())
    {
        m_elements.erase(it);
    }
    elem->clearItems();

    for (auto v: m_views)
    {
        v.second->removeElement(elem);
    }

    {
        auto it = m_elementsByPath.find(elem->path());
        if (it != m_elementsByPath.end())
        {
            m_elementsByPath.erase(it);
        }
    }
    {
        auto it = m_elementsById.find(elem->elementId());
        if (it != m_elementsById.end())
        {
            m_elementsById.erase(it);
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
        elem->m_order = m_elemOrder;
        m_elements.emplace(elem->m_order, elem);
        ++m_elemOrder;

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

    if (m_updateAllElements)
    {
        m_changed = true;

        m_updateAllElements = false;
        for (auto elem: m_elements)
            elem.second->update();
    }

    if (m_wheelInteraction->wasStarted() || m_wheelInteraction->isRunning())
    {
        m_changed = true;
        int c = cover->getMouseButton()->getWheel();
        int pressed = c<0 ? vrui::vruiButtons::WHEEL_DOWN : vrui::vruiButtons::WHEEL_UP;
        int count = std::abs(c);
        for (int i=0; i<count; ++i)
            buttonEvent(pressed);
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
        std::cerr << "Creating by id: " << elem.first << " -> " << elem.second->path() << std::endl;
        view->elementFactory(elem.second);
    }

    m_updateAllElements = true;

    return true;
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

void Manager::updateValue(const EditField *input) const
{
    for (auto v: m_views)
    {
        v.second->updateValue(input);
    }
}

bool Manager::keyEvent(int type, int mod, int keySym)
{
    bool handled = false;

    if (type == osgGA::GUIEventAdapter::KEYDOWN
            || type == osgGA::GUIEventAdapter::KEYUP)
    {
        bool down = type==osgGA::GUIEventAdapter::KEYDOWN;
        if (keySym == osgGA::GUIEventAdapter::KEY_Shift_L
                || keySym == osgGA::GUIEventAdapter::KEY_Shift_R)
        {
            if (down)
                m_modifiers |= ModShift;
            else
                m_modifiers &= ~ModShift;
        }
        if (keySym == osgGA::GUIEventAdapter::KEY_Control_L
                || keySym == osgGA::GUIEventAdapter::KEY_Control_R)
        {
            if (down)
                m_modifiers |= ModCtrl;
            else
                m_modifiers &= ~ModCtrl;
        }
        if (keySym == osgGA::GUIEventAdapter::KEY_Alt_L
                || keySym == osgGA::GUIEventAdapter::KEY_Alt_R)
        {
            if (down)
                m_modifiers |= ModAlt;
            else
                m_modifiers &= ~ModAlt;
        }
        if (keySym == osgGA::GUIEventAdapter::KEY_Meta_L
                || keySym == osgGA::GUIEventAdapter::KEY_Meta_R)
        {
            if (down)
                m_modifiers |= ModMeta;
            else
                m_modifiers &= ~ModMeta;
        }

        bool show = false;
        if (down)
        {
            //show = true;
        }
        if (show)
            std::cerr << "key " << (down ? "down" : "up") << ": ";

        bool alt = mod & osgGA::GUIEventAdapter::MODKEY_ALT;
        bool ctrl = mod & osgGA::GUIEventAdapter::MODKEY_CTRL;
        bool shift = mod & osgGA::GUIEventAdapter::MODKEY_SHIFT;
        bool meta = mod & osgGA::GUIEventAdapter::MODKEY_META;

        int modifiers = 0;
        if (meta)
        {
            modifiers |= ModMeta;
            if (show)
                std::cerr << "meta+";
        }
        if (ctrl)
        {
            modifiers |= ModCtrl;
            if (show)
                std::cerr << "ctrl+";
        }
        if (alt)
        {
            modifiers |= ModAlt;
            if (show)
                std::cerr << "alt+";
        }
        if (shift)
        {
            modifiers |= ModShift;
            if (show)
                std::cerr << "shift+";
        }

        if (down)
        {
            if (shift && keySym < 255 && std::isupper(keySym))
            {
                //std::cerr << "ui::Manager: mapping to lower" << std::endl;
                keySym = std::tolower(keySym);
            }
            if (show)
                std::cerr << "'" << (char)keySym << "'" << std::endl;

            for (auto &elemPair: m_elements)
            {
                auto &elem = elemPair.second;
                if (elem->enabled() && elem->matchShortcut(modifiers, keySym))
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
        }

        //std::cerr << "modifiers=" << modifiers << ", m_modifiers=" << m_modifiers << std::endl;
        //m_modifiers = modifiers;
    }
    else if (type == osgGA::GUIEventAdapter::RELEASE
             || type == osgGA::GUIEventAdapter::SCROLL)
    {
        std::cerr << "mouse: ";

        int button = 0;
        int modifiers = 0;
        if (type == osgGA::GUIEventAdapter::RELEASE)
        {
            if (mod == osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON)
                button = Left;
            if (mod == osgGA::GUIEventAdapter::MIDDLE_MOUSE_BUTTON)
                button = Middle;
            if (mod == osgGA::GUIEventAdapter::RIGHT_MOUSE_BUTTON)
                button = Right;
        }
        else
        {
            if (mod == osgGA::GUIEventAdapter::SCROLL_UP)
                button = ScrollUp;
            if (mod == osgGA::GUIEventAdapter::SCROLL_DOWN)
                button = ScrollDown;
        }

        switch (button)
        {
        case Left:
            std::cerr << "Left" << std::endl;
            break;
        case Middle:
            std::cerr << "Middle" << std::endl;
            break;
        case Right:
            std::cerr << "Right" << std::endl;
            break;
        case ScrollUp:
            std::cerr << "ScrollUp" << std::endl;
            break;
        case ScrollDown:
            std::cerr << "ScrollDown" << std::endl;
            break;
        }

        if (button != 0)
        {
            for (auto &elemPair: m_elements)
            {
                auto &elem = elemPair.second;
                if (elem->enabled() && elem->matchButton(modifiers, button))
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
        }
    }

    return handled;
}

bool Manager::buttonEvent(int button) const
{
    bool handled = false;

    for (auto &elemPair: m_elements)
    {
        auto &elem = elemPair.second;
        if (elem->enabled() && elem->matchButton(m_modifiers, button))
        {
            elem->shortcutTriggered();
            if (handled)
            {
                std::cerr << "ui::Manager: duplicate mapping for button on " << elem->path() << std::endl;
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
        auto mask = state.second.first;
        auto tb = state.second.second;

        *m_updates << id;
        *m_updates << mask;
        *m_updates << false; // trigger
        *m_updates << tb->get_length();
        m_updates->addBinary(tb->get_data(), tb->get_length());

        ++m_numUpdates;
    }
    m_elemState.clear();
}

void Manager::queueUpdate(const Element *elem, Element::UpdateMaskType mask, bool trigger)
{
    if (mask & Element::UpdateParent)
    {
        auto it = m_elements.find(elem->m_order);
        if (it != m_elements.end())
        {
            Element *e = it->second;
            assert(e == elem);
            m_elements.erase(it);
            e->m_order = m_elemOrder;
            m_elements.emplace(e->m_order, e);
            ++m_elemOrder;
        }
    }

    if (elem->elementId() < 0)
    {
        assert(!trigger);
        return;
    }

    assert(elem->elementId() >= 0);

    auto it = m_elemState.find(elem->elementId());
    if (it != m_elemState.end())
        mask |= it->second.first;
    if (trigger)
    {
        if (it != m_elemState.end())
            m_elemState.erase(it);
        flushUpdates();

        covise::TokenBuffer tb;
        elem->save(tb);

        *m_updates << elem->elementId();
        *m_updates << mask;
        *m_updates << trigger;
        *m_updates << tb.get_length();
        m_updates->addBinary(tb.get_data(), tb.get_length());

        ++m_numUpdates;
    }
    else
    {
        if (it == m_elemState.end())
        {
            it = m_elemState.emplace(elem->elementId(), std::make_pair(Element::UpdateMaskType(0),std::make_shared<covise::TokenBuffer>())).first;
        }
        else
        {
            it->second.second->reset();
        }
        it->second.first = mask;
        elem->save(*it->second.second);
    }
}

void Manager::processUpdates(std::shared_ptr<covise::TokenBuffer> updates, int numUpdates, bool runTriggers)
{
    updates->rewind();

    std::vector<ButtonGroup *> delayed;
    for (int i=0; i<numUpdates; ++i)
    {
        //std::cerr << "processing " << i << std::flush;
        int id = -1;
        *updates >> id;
        Element::UpdateMaskType mask(0);
        *updates >> mask;
        bool trigger = false;
        *updates >> trigger;
        int len = 0;
        *updates >> len;
        auto data = updates->getBinary(len);
        auto elem = getById(id);
        if (!elem)
        {
            std::cerr << "ui::Manager::processUpdates NOT FOUND: id=" << id << ", trigger=" << trigger << std::endl;
            continue;
        }
        if (cover->debugLevel(5))
            std::cerr << "ui::Manager::processUpdates for id=" << id << ": " << elem->path() << std::endl;
        covise::TokenBuffer tb(data, len);
        //std::cerr << ": id=" << id << ", trigger=" << trigger << std::endl;
        assert(elem);
        elem->load(tb);
        elem->update(mask);
        if (trigger)
        {
            if (runTriggers)
            {
                setChanged();
                if (auto bg = dynamic_cast<ButtonGroup *>(elem))
                    delayed.push_back(bg);
                else
                    elem->triggerImplementation();
            }
            else
            {
                std::cerr << "ui::Manager::processUpdates: path=" << elem->path() << " still would trigger" << std::endl;
            }
        }
    }

    for (auto &bg: delayed)
        bg->triggerImplementation();
}

bool Manager::sync()
{
    bool changed = false;

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
                coVRMSController::instance()->syncMessage(&msg);
            }
            else
            {
                covise::Message msg;
                coVRMSController::instance()->syncMessage(&msg);
                m_updates.reset(new covise::TokenBuffer(&msg));
            }
        }
    }

    int round = 0;
    while (m_numUpdates > 0)
    {
        changed = true;

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
    return changed;
}

std::vector<const Element *> Manager::getAllElements() const
{
    std::vector<const Element *> result;
    for (auto e: m_elements)
    {
        result.push_back(e.second);
    }
    return result;
}

}
}
