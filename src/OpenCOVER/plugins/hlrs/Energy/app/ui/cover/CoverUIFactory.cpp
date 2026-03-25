#include "CoverUIFactory.h"
#include "CoverButton.h"
#include "CoverButtonGroup.h"
#include "CoverMenu.h"
#include "CoverEditField.h"

std::unique_ptr<IButton> CoverUIFactory::createButton(IComponent *parent, const std::string &name) const
{
    return std::make_unique<CoverButton>(parent, name);
}

std::unique_ptr<IButtonGroup> CoverUIFactory::createButtonGroup(IComponent *parent, const std::string &name, std::vector<std::unique_ptr<IButton>> &&buttons) const
{
    return std::make_unique<CoverButtonGroup>(parent, name, std::move(buttons));
}

std::unique_ptr<IMenu> CoverUIFactory::createMenu(const std::string &name, IComponent *parent) const
{
    return std::make_unique<CoverMenu>(parent, name);
}

std::unique_ptr<IEditDoubleField> CoverUIFactory::createEditField(IComponent *parent, const std::string &name) const
{
    return std::make_unique<CoverEditField>(parent, name);
}
