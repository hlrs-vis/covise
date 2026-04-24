#pragma once

#include "lib/core/interfaces/ui/IComponent.h"
#include "lib/core/interfaces/ui/ISelectionList.h"
#include <lib/core/interfaces/ui/IGUIFactory.h>

using namespace core::interface::ui;

class CoverUIFactory final : public IGUIFactory
{
public:
    std::unique_ptr<IButton> createButton(IComponent *parent, const std::string &name) const override;
    std::unique_ptr<IButtonGroup> createButtonGroup(IComponent *parent, const std::string &name, std::vector<std::unique_ptr<IButton>> &&buttons) const override;
    std::unique_ptr<IMenu> createMenu(const std::string &name, IComponent *parent = nullptr) const override;
    std::unique_ptr<IEditDoubleField> createEditField(IComponent *parent, const std::string &name) const override;
    std::unique_ptr<ISelectionList> createSelectionList(IComponent *parent, const std::string &name) const override;
};
