#pragma once

#include "interfaces/ui/IButton.h"
#include "ComponentBase.h"

namespace core
{
class ButtonBase : public ComponentBase, public interface::ui::IButton
{
public:
    ButtonBase(interface::ui::IComponent *parent, const std::string &name)
        : ComponentBase(parent, name)
    {
    }
    virtual ~ButtonBase() = default;
};
}
