#pragma once

#include "interfaces/ui/IMenu.h"
#include "ComponentBase.h"

namespace core
{
class MenuBase : public ComponentBase, public interface::ui::IMenu
{
public:
    MenuBase(core::interface::ui::IComponent *parent, const std::string &name)
        : ComponentBase(parent, name)
    {
    }
    virtual ~MenuBase() = default;
};
}
