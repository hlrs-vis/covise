#include "Group.h"
#include "Manager.h"

#include <algorithm>
#include <cassert>

namespace vive {
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

void Group::update(UpdateMaskType mask) const
{
    Element::update(mask);
    if (mask & UpdateChildren)
        manager()->updateChildren(this);
    if (mask & UpdateRelayout)
        manager()->updateRelayout(this);
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
        elem->setParent(this);
        manager()->queueUpdate(this, UpdateChildren);
        return true;
    }
    return false;
}

bool Group::remove(Element *elem)
{
    if (Container::remove(elem))
    {
        elem->setParent(nullptr);
        manager()->queueUpdate(this, UpdateChildren);
        return true;
    }
    return false;
}

void Group::allowRelayout(bool rl)
{
    m_allowRelayout = rl;
    manager()->queueUpdate(this, UpdateRelayout);
}


void Group::save(covise::TokenBuffer& buf) const
{
    Element::save(buf);
    buf << m_allowRelayout;
}

void Group::load(covise::TokenBuffer& buf)
{
    Element::load(buf);
    buf >> m_allowRelayout;
}

}
}
