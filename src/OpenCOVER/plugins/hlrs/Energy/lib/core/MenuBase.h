#pragma once

#include "interfaces/ui/IMenu.h"
#include "ComponentBase.h"

namespace core
{
class MenuBase : public ComponentBase, public interface::ui::IMenu
{
public:
    using ComponentBase::ComponentBase;
    virtual ~MenuBase() = default;
};
}
