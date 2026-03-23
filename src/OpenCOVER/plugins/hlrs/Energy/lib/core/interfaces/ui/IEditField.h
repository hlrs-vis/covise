#pragma once
#include "IComponent.h"
#include <functional>

namespace core::interface::ui
{
class IEditDoubleField : virtual public IComponent
{
public:
    virtual ~IEditDoubleField() = default;
    virtual void setCallback(const std::function<void(std::string)> &func) = 0;
    virtual void setValue(double val) = 0;
    virtual double getValue() = 0;
};
}
