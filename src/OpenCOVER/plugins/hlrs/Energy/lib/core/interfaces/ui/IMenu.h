#pragma once
#include "IComponent.h"

namespace core::interface::ui
{
class IMenu : virtual public IComponent
{
public:
    virtual ~IMenu() = default;
};
}
