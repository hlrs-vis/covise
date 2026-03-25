#include "CoverEditField.h"
#include "CoverMenu.h"
#include <stdexcept>

CoverEditField::CoverEditField(core::interface::ui::IComponent *parent, const std::string &name)
    : core::EditFieldBase(parent, name)
{
    auto menu = dynamic_cast<CoverMenu *>(parent);
    if (menu)
        m_field = new opencover::ui::EditField(menu->getMenu(), name);
    else
        std::runtime_error("Parent invalid for EditField " + name);
}
