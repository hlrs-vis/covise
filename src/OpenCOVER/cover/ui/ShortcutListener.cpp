#include "ShortcutListener.h"
#include <sstream>
#include <vector>
#include <iostream>
#include <cctype>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <osgGA/GUIEventAdapter>

namespace opencover {
namespace ui {

void ShortcutListener::clearShortcuts()
{
    m_shortcuts.clear();
}

void ShortcutListener::setShortcut(const std::string &shortcut)
{
    clearShortcuts();
    addShortcut(shortcut);
}

void ShortcutListener::addShortcut(const std::string &shortcut)
{
    Shortcut sh;
    sh.text = shortcut;

    std::stringstream stream(shortcut);
    bool havePlus = false;
    std::vector<std::string> components;
    std::string item, key;
    while (std::getline(stream, item, '+'))
    {
        if (item.empty())
        {
            if (havePlus)
            {
                std::cerr << "ui::ShortcutListener: invalid key sequence with multiple consecutive '+': " << shortcut << std::endl;
            }
            havePlus = true;
            item = "+";
        }
        for (auto &c: item)
            c = std::tolower(c);

        if (item == "alt")
        {
            sh.modifiers |= ModAlt;
        }
        else if (item == "ctrl")
        {
            sh.modifiers |= ModCtrl;
        }
        else if (item == "shift")
        {
            sh.modifiers |= ModShift;
        }
        else if (item == "meta")
        {
            sh.modifiers |= ModMeta;
        }
        else if (key.empty() && item.length()==1)
        {
            key = item;
        }
        else if (item.length()>5 && item.substr(0,5) == "mouse")
        {
            key = item;
        }
        else if (item.length()>7 && item.substr(0,7) == "button:")
        {
            key = item;
        }
        else if (item == "escape" || item == "esc")
        {
            key = item;
        }
        else
        {
            key = item;
            std::cerr << "ui::ShortcutListener: invalid key sequence: " << shortcut << std::endl;
        }
    }

    if (key.length()>5 && key.substr(0, 5) == "mouse")
    {
        std::string button = key.substr(5);
        if (button == "left")
            sh.button = Left;
        if (button == "middle")
            sh.button = Middle;
        if (button == "right")
            sh.button = Right;
        if (button == "scrollup")
            sh.button = ScrollUp;;
        if (button == "scrolldown")
            sh.button = ScrollDown;;
        if (button == "scrollleft")
            sh.button = ScrollLeft;;
        if (button == "scrollright")
            sh.button = ScrollRight;;
    }
    else if (key.length()>7 && key.substr(0, 7) == "button:")
    {
        std::string button = key.substr(7);
        //std::cerr << "ui::ShortcutListener: configuring button " << button << std::endl;
        if (button == "action")
            sh.button = vrui::vruiButtons::ACTION_BUTTON;
        if (button == "drive")
            sh.button = vrui::vruiButtons::DRIVE_BUTTON;
        if (button == "drive")
            sh.button = vrui::vruiButtons::DRIVE_BUTTON;
        if (button == "xform")
            sh.button = vrui::vruiButtons::XFORM_BUTTON;
        if (button == "menu")
            sh.button = vrui::vruiButtons::MENU_BUTTON;
        if (button == "zoom")
            sh.button = vrui::vruiButtons::ZOOM_BUTTON;
        if (button == "back" || button == "backward")
            sh.button = vrui::vruiButtons::BACKWARD_BUTTON;
        if (button == "forward")
            sh.button = vrui::vruiButtons::FORWARD_BUTTON;
        if (button == "wheelup" || button == "scrollup")
            sh.button = vrui::vruiButtons::WHEEL_UP;
        if (button == "wheeldown" || button == "scrolldown")
            sh.button = vrui::vruiButtons::WHEEL_DOWN;
        if (button == "wheelleft" || button == "scrollleft")
            sh.button = vrui::vruiButtons::WHEEL_LEFT;
        if (button == "wheelright" || button == "scrollright")
            sh.button = vrui::vruiButtons::WHEEL_RIGHT;
    }
    else if (key == "esc" || key == "escape")
    {
        sh.symbol = osgGA::GUIEventAdapter::KEY_Escape;
    }
    else if ((key.length()==2 || key.length()==3) && key[0] == 'f')
    {
        int fnum = atoi(key.substr(1).c_str());
        if (fnum >= 1 && fnum <= 20)
            sh.symbol = osgGA::GUIEventAdapter::KEY_F1 + fnum-1;
    }
    else
    {
        sh.symbol = key[0];
    }

    //std::cerr << "ShortcutListener::setShortcut: symbol=" << (char)m_symbol << std::endl;

    m_shortcuts.push_back(sh);
}

bool ShortcutListener::hasShortcut() const
{
    return !m_shortcuts.empty();
}

bool ShortcutListener::matchShortcut(int mod, int sym) const
{
    for (const auto &sh: m_shortcuts)
    {
        if (sh.symbol && sh.modifiers == mod && sh.symbol == sym)
            return true;
    }

    return false;
}

bool ShortcutListener::matchButton(int mod, int button) const
{
    for (const auto &sh: m_shortcuts)
    {
        if (sh.button && sh.modifiers == mod && sh.button == button)
            return true;
    }

    return false;
}

size_t ShortcutListener::shortcutCount() const
{
    return m_shortcuts.size();
}

std::string ShortcutListener::shortcutText(size_t idx) const
{
    std::string text;
    if (idx >= m_shortcuts.size())
        return text;

    const auto &sh = m_shortcuts[idx];
    if (sh.modifiers & ModAlt)
        text += "Alt+";
    if (sh.modifiers & ModCtrl)
        text += "Ctrl+";
    if (sh.modifiers & ModMeta)
        text += "Meta+";
    if (sh.modifiers & ModShift)
        text += "Shift+";

    return sh.text;
}

void ShortcutListener::shortcutTriggered()
{
}

}
}
