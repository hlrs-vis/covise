#pragma once
#include <functional>
#include <string>

#include "IComponent.h"

namespace core::interface::ui
{
class IButton: virtual public IComponent
{
public:
    virtual ~IButton() = default;
    virtual void setCallback(const std::function<void(bool)> &func) = 0;
    virtual void setState(bool state) = 0;
    virtual void setText(const std::string &txt) = 0;
    virtual bool state() const = 0;
};
}
