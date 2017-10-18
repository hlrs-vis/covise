#include "ShortcutListener.h"
#include <sstream>
#include <vector>
#include <iostream>
#include <cctype>

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
            sh.modifiers |= ModAlt;
        else if (item == "ctrl")
            sh.modifiers |= ModCtrl;
        else if (item == "shift")
            sh.modifiers |= ModShift;
        else if (item == "meta")
            sh.modifiers |= ModMeta;
        else if (key.empty() && item.length()==1)
            key = item;
        else
            std::cerr << "ui::ShortcutListener: invalid key sequence: " << shortcut << std::endl;
    }

    sh.symbol = key[0];

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
        if (sh.modifiers == mod && sh.symbol == sym)
            return true;
    }

    return false;
}

void ShortcutListener::shortcutTriggered()
{
}

}
}
