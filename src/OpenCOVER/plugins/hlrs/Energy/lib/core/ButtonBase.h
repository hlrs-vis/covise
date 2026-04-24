#pragma once

#include "interfaces/ui/IButton.h"
#include "ComponentBase.h"

namespace core
{
class ButtonBase : public ComponentBase, public interface::ui::IButton
{
public:
    using ComponentBase::ComponentBase;
    virtual ~ButtonBase() = default;
};
}
