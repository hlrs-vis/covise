#pragma once
#include <functional>
#include <memory>

#include "IComponent.h"
#include "IButton.h"

namespace core::interface::ui
{
class IButtonGroup: virtual public IComponent
{
public:
    virtual ~IButtonGroup() = default;
    virtual void setCallback(const std::function<void(int)> &func) = 0;
    virtual void add(std::unique_ptr<IButton> button) = 0;
    virtual IButton* getChild(int position) = 0;
};
}
