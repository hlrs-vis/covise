#include "Group.h"
#include "Manager.h"

#include <algorithm>
#include <cassert>

namespace opencover {
namespace ui {

Group::Group(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

Group::Group(Group *parent, const std::string &name)
: Element(parent, name)
{
}

Group::~Group()
{
    manager()->remove(this);

    clearItems();
    clearChildren();
}

bool Group::add(Element *elem, int where)
{
    if (Container::add(elem, where))
    {
        if (elem->parent() != this)
        {
            if (elem->parent())
                elem->parent()->remove(elem);
            assert(!elem->m_parent);
        }
        elem->m_parent = this;
        manager()->queueUpdate(elem, UpdateParent);
        return true;
    }
    return false;
}

bool Group::remove(Element *elem)
{
    if (Container::remove(elem))
    {
        elem->m_parent = nullptr;
        manager()->queueUpdate(elem, UpdateParent);
        return true;
    }
    return false;
}

}
}

