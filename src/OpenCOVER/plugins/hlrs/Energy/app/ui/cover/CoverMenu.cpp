#include "CoverMenu.h"
#include "CoverOwner.h"

using namespace opencover;

CoverMenu::CoverMenu(core::interface::ui::IComponent *parent, const std::string &name)
    : core::MenuBase(parent, name)
{
    auto menu = dynamic_cast<CoverMenu *>(parent);
    if (menu)
    {
        m_menu = new ui::Menu(menu->getMenu(), name);
        return;
    }

    auto owner = dynamic_cast<CoverOwner *>(parent);
    if (owner)
        m_menu = new ui::Menu(name, owner);
}
