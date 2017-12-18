#include "Menu.h"
#include "Manager.h"

namespace opencover {
namespace ui {

Menu::Menu(const std::string &name, Owner *owner)
: Group(name, owner)
{
}

Menu::Menu(Group *parent, const std::string &name)
: Group(parent, name)
{
}

Menu::~Menu()
{
    manager()->remove(this);
    clearItems();
    clearChildren();
}

}
}
