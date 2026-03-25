#include "CoverButtonGroup.h"
#include "CoverButton.h"
#include "CoverMenu.h"
#include <memory>
#include <stdexcept>

CoverButtonGroup::CoverButtonGroup(core::interface::ui::IComponent *parent, const std::string &name, std::vector<std::unique_ptr<core::interface::ui::IButton>> &&buttons)
    : core::ButtonGroupBase(parent, name)
    , m_childs(std::move(buttons))
{
    auto menu = dynamic_cast<CoverMenu*>(parent);
    if (menu)
        m_buttonGroup = new opencover::ui::ButtonGroup(name, menu->getMenu());
    else
        throw std::runtime_error("Invalid parent for CoverButtonGroup " + name);

    for (auto &button : buttons)
    {
        auto c_button = dynamic_cast<CoverButton *>(button.get());
        if (c_button)
            m_buttonGroup->add(c_button->getButton());
    }
}

void CoverButtonGroup::setCallback(const std::function<void(bool)> &func)
{
    m_buttonGroup->setCallback(func);
}

void CoverButtonGroup::add(std::unique_ptr<core::interface::ui::IButton> button)
{
    auto c_button = dynamic_cast<CoverButton *>(button.get());
    if (c_button) {
        m_buttonGroup->add(c_button->getButton());
        m_childs.push_back(std::move(button));
    }
}

core::interface::ui::IButton* CoverButtonGroup::getChild(int position) { 
    if (position >= 0 && position < m_childs.size()) 
        return m_childs[position].get();
    throw std::runtime_error("Incorrect position for ButtonGroup: " + getName());
}
