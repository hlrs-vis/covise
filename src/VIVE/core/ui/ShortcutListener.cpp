#include "ShortcutListener.h"
#include <sstream>
#include <vector>
#include <iostream>
#include <cctype>

#include <OpenVRUI/sginterface/vruiButtons.h>
//#include <osgGA/GUIEventAdapter>

namespace vive {
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
    std::string item, text, key;
    while (std::getline(stream, text, '+'))
    {
        if (text.empty())
        {
            if (havePlus)
            {
                std::cerr << "ui::ShortcutListener: invalid key sequence with multiple consecutive '+': " << shortcut << std::endl;
            }
            havePlus = true;
            text = "+";
        }
        item = text;
        for (auto &c: item)
            c = std::tolower(c);

        if (item == "alt" || item == "opt" || item == "option")
        {
            sh.modifiers |= ModAlt;
        }
        else if (item == "ctrl" || item == "control")
        {
            sh.modifiers |= ModCtrl;
        }
        else if (item == "shift")
        {
            sh.modifiers |= ModShift;
        }
        else if (item == "meta" || item == "cmd" || item == "command")
        {
            sh.modifiers |= ModMeta;
        }
        else if (item.length()>5 && item.substr(0,5) == "mouse")
        {
            key = text;
        }
        else if (item.length()>7 && item.substr(0,7) == "button:")
        {
            key = text;
        }
        else if (item == "escape" || item == "esc")
        {
            key = "Escape";
        }
        else if (item == "space" || item == " ")
        {
            key = "Space";
        }
        else if (item == "enter" || item == "return")
        {
            key = "Enter";
        }
        else if (item == "backspace")
        {
            key = "Backspace";
        }
        else if (item == "delete" || item == "del")
        {
            key = "Delete";
        }
        else if (key.empty() && item.length() == 1)
        {
            key = text;
        }
        else if ((item.length() == 2 || item.length() == 3) && item[0] == 'f')
        {
            int fnum = atoi(item.substr(1).c_str());
           // if (fnum >= 1 && fnum <= 20)
             //   sh.symbol = osgGA::GUIEventAdapter::KEY_F1 + fnum-1;
            key = "F" + std::to_string(fnum);
        }
        else
        {
            key = text;
            std::cerr << "ui::ShortcutListener: invalid key sequence: " << shortcut << std::endl;
        }
    }

    if (item.length() > 5 && item.substr(0, 5) == "mouse")
    {
        std::string button = item.substr(5);
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
    else if (item.length() > 7 && item.substr(0, 7) == "button:")
    {
        std::string button = item.substr(7);
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
  /*  else if (key == "Escape")
    {
        sh.symbol = osgGA::GUIEventAdapter::KEY_Escape;
    }
    else if (key == "Space")
    {
        sh.symbol = osgGA::GUIEventAdapter::KEY_Space;
    }
    else if (key == "Enter")
    {
        sh.symbol = osgGA::GUIEventAdapter::KEY_Return;
    }
    else if (key == "Backspace")
    {
        sh.symbol = osgGA::GUIEventAdapter::KEY_BackSpace;
    }
    else if (key == "Delete")
    {
        sh.symbol = osgGA::GUIEventAdapter::KEY_Delete;
    }
    else if ((item.length() == 2 || item.length() == 3) && item[0] == 'f')
    {
        int fnum = atoi(item.substr(1).c_str());
        if (fnum >= 1 && fnum <= 20)
            sh.symbol = osgGA::GUIEventAdapter::KEY_F1 + fnum-1;
        key = "F" + std::to_string(fnum);
    }*/
    else
    {
        if (key.length() > 1)
        {
            std::cerr << "ui::ShortcutListener: truncating key=" << key << " to " << key.substr(0, 1) << std::endl;
        }
        key = key.substr(0, 1);
        sh.symbol = tolower(key[0]);
    }
    if (sh.modifiers & ModShift && key.length() == 1)
    {
        key[0] = toupper(key[0]);
    }
    sh.key = key;

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
    std::string key = sh.key;

#ifdef __APPLE__
    if (sh.modifiers & ModAlt)
        text += "Opt+";
    if (sh.modifiers & ModCtrl)
        text += "Ctrl+";
    if (sh.modifiers & ModShift)
        text += "Shift+";
    if (sh.modifiers & ModMeta)
        text += "Cmd+";
#else
    if (sh.modifiers & ModShift)
        text += "Shift+";
    if (sh.modifiers & ModCtrl)
        text += "Ctrl+";
    if (sh.modifiers & ModAlt)
        text += "Alt+";
    if (sh.modifiers & ModMeta)
        text += "Meta+";
#endif

    return text + key;
}

void ShortcutListener::shortcutTriggered()
{
}

}
}
