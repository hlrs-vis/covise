#pragma once

#include "interfaces/ui/IButtonGroup.h"
#include "ComponentBase.h"

namespace core
{
class ButtonGroupBase : public ComponentBase, public interface::ui::IButtonGroup
{
public:
    ButtonGroupBase(core::interface::ui::IComponent *parent, const std::string &name)
        : ComponentBase(parent, name)
    {
    }
    virtual ~ButtonGroupBase() = default;
};
}
