#include "ShortcutListener.h"
#include <sstream>
#include <vector>
#include <iostream>
#include <cctype>

namespace opencover {
namespace ui {

void ShortcutListener::setShortcut(const std::string &shortcut)
{
    m_modifiers = ModNone;
    m_symbol = 0;

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
            m_modifiers |= ModAlt;
        else if (item == "ctrl")
            m_modifiers |= ModCtrl;
        else if (item == "shift")
            m_modifiers |= ModShift;
        else if (item == "meta")
            m_modifiers |= ModMeta;
        else if (key.empty() && item.length()==1)
            key = item;
        else
            std::cerr << "ui::ShortcutListener: invalid key sequence: " << shortcut << std::endl;
    }

    m_symbol = key[0];

    //std::cerr << "ShortcutListener::setShortcut: symbol=" << (char)m_symbol << std::endl;
}

bool ShortcutListener::hasShortcut() const
{
    return m_symbol != 0;
}

int ShortcutListener::modifiers() const
{
    return m_modifiers;
}

int ShortcutListener::symbol() const
{
    return m_symbol;
}

void ShortcutListener::shortcutTriggered()
{
}

}
}
