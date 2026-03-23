#include "CoverButton.h"
#include "CoverMenu.h"
#include <stdexcept>

CoverButton::CoverButton(core::interface::ui::IComponent *parent, const std::string &name)
    : core::ButtonBase(parent, name)
    , m_button(nullptr)
{
    auto *menu = dynamic_cast<CoverMenu *>(parent);
    if (menu)
        m_button = new opencover::ui::Button(menu->getMenu(), name);
    else 
        throw std::runtime_error("No valid parent for Button " + name);
}

void CoverButton::setCallback(const std::function<void(bool)> &func)
{
    m_button->setCallback(std::move(func));
}

void CoverButton::setState(bool state)
{
    m_button->setState(state);
}

void CoverButton::setText(const std::string &txt)
{
    m_button->setText(txt);
}
