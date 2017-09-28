#undef NDEBUG
#include "Element.h"
#include "Group.h"
#include "Manager.h"

#include <cassert>

#include <net/tokenbuffer.h>

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
    manager()->remove(this);
}

int Element::elementId() const
{
    return m_id;
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
    manager()->queueUpdate(this);
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
    manager()->queueUpdate(this);
}

bool Element::enabled() const
{
    return m_enabled;
}

void Element::setEnabled(bool flag)
{
    m_enabled = flag;
    manager()->queueUpdate(this);
}

void Element::trigger() const
{
    manager()->setChanged();
    manager()->queueUpdate(this, true);
}

void Element::triggerImplementation() const
{
}

void Element::save(covise::TokenBuffer &buf) const
{
    buf << elementId();
    buf << m_visible;
    buf << m_enabled;
    buf << m_label;

}

void Element::load(covise::TokenBuffer &buf)
{
    int id;
    buf >> id;
    assert(m_id == id);
    buf >> m_visible;
    buf >> m_enabled;
    buf >> m_label;
}

}
}
