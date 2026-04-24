#pragma once
#include "IComponent.h"
#include <functional>

namespace core::interface::ui
{
class ISelectionList : virtual public IComponent
{
public:
    virtual ~ISelectionList() = default;
    virtual void setCallback(const std::function<void(int)> &func) = 0;
    virtual int selectedIndex() const = 0;
    virtual std::string selectedItem() const = 0;
    virtual void setList(const std::vector<std::string> &names) = 0;
    virtual void select(int i) = 0;
};
}
