#pragma once
#include <cover/ui/Menu.h>
#include <lib/core/MenuBase.h>

class CoverMenu final : public core::MenuBase
{
public:
    CoverMenu(core::interface::ui::IComponent *parent, const std::string &name);
    auto getMenu() const { return m_menu; }

private:
    opencover::ui::Menu *m_menu;
};
