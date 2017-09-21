#include "Element.h"
#include "Group.h"
#include "Manager.h"

#include <cassert>

namespace opencover {
namespace ui {

Element::Element(const std::string &name, Owner *owner)
: Owner(name, owner)
, m_label(name)
{
    manager()->add(this);
}

Element::Element(Group *group, const std::string &name)
: Owner(name, group)
, m_parent(group)
, m_label(name)
{
    manager()->add(this);
    assert(m_parent == group);
}

Element::~Element()
{
}

Group *Element::parent() const
{
    return m_parent;
}

void Element::update() const
{
    manager()->updateText(this);
    manager()->updateVisible(this);
    manager()->updateEnabled(this);
}

std::set<Container *> Element::containers()
{
    return m_containers;
}

void Element::setText(const std::string &label)
{
    m_label = label;
    manager()->updateText(this);
}

const std::string &Element::text() const
{
    return m_label;
}

bool Element::visible() const
{
    return m_visible;
}

void Element::setVisible(bool flag)
{
    m_visible = flag;
    manager()->updateVisible(this);
}

bool Element::enabled() const
{
    return m_enabled;
}

void Element::setEnabled(bool flag)
{
    m_enabled = flag;
    manager()->updateEnabled(this);
}

void Element::trigger() const
{
    manager()->setChanged();
    triggerImplementation();
}

void Element::triggerImplementation() const
{
}

}
}
