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

bool Menu::add(Element *elem)
{
    if (Group::add(elem))
    {
        manager()->updateChildren(this);
        return true;
    }
    return false;
}

bool Menu::remove(Element *elem)
{
    if (Group::remove(elem))
    {
        manager()->updateChildren(this);
        return true;
    }
    return false;
}

}
}
