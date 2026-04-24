#pragma once

#include "interfaces/ui/IButtonGroup.h"
#include "ComponentBase.h"

namespace core
{
class ButtonGroupBase : public ComponentBase, public interface::ui::IButtonGroup
{
public:
    using ComponentBase::ComponentBase;
    virtual ~ButtonGroupBase() = default;
};
}
