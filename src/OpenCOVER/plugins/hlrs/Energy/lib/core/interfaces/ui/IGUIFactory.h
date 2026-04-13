#pragma once
#include "IButton.h"
#include "IButtonGroup.h"
#include "IEditField.h"
#include "IMenu.h"
#include "IComponent.h"
#include <memory>

namespace core::interface::ui
{
class IGUIFactory
{
public:
    virtual ~IGUIFactory() = default;
    virtual std::unique_ptr<IButton> createButton(IComponent *parent, const std::string &name) const = 0;
    virtual std::unique_ptr<IButtonGroup> createButtonGroup(IComponent *parent, const std::string &name, std::vector<std::unique_ptr<IButton>> &&buttons) const = 0;
    virtual std::unique_ptr<IMenu> createMenu(const std::string &name, IComponent *parent = nullptr) const = 0;
    virtual std::unique_ptr<IEditDoubleField> createEditField(IComponent *parent, const std::string &name) const = 0;
};
}
