#undef NDEBUG
#include "Element.h"
#include "Group.h"
#include "Manager.h"
#include "View.h"

#include <cassert>

#include <net/tokenbuffer.h>
#include <vrb/client/SharedState.h>

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
    assert(m_parent == group);
    manager()->add(this);
}

Element::~Element()
{
    manager()->remove(this);
    while (!m_containers.empty())
    {
        auto it = m_containers.begin();
        if (*it == m_parent)
            m_parent = nullptr;
        (*it)->remove(this);
    }
    assert(!m_parent);
}

int Element::elementId() const
{
    return m_id;
}

void Element::setPriority(Element::Priority prio)
{
    m_priority = prio;
}

Element::Priority Element::priority() const
{
    return m_priority;
}

void Element::setShared(bool state)
{
    assert(!state && "sharing of ui::Element state requested, but sharing not implemented for Element type");
}

bool Element::isShared() const
{
    return bool(m_sharedState);
}

void Element::setIcon(const std::string &iconName)
{
    m_iconName = iconName;
}

const std::string &Element::iconName() const
{
    return m_iconName;
}

Group *Element::parent() const
{
    return m_parent;
}

void Element::setParent(Group *parent)
{
    if (m_parent)
        manager()->queueUpdate(m_parent, UpdateChildren);
    m_parent = parent;
    if (m_parent)
        manager()->queueUpdate(m_parent, UpdateChildren);
}

void Element::update(UpdateMaskType mask) const
{
    if (mask & UpdateText)
        manager()->updateText(this);
    if (mask & UpdateVisible)
        manager()->updateVisible(this);
    if (mask & UpdateEnabled)
        manager()->updateEnabled(this);
}

std::set<Container *> Element::containers()
{
    return m_containers;
}

void Element::setText(const std::string &label)
{
    m_label = label;
    manager()->queueUpdate(this, UpdateText);
}

const std::string &Element::text() const
{
    return m_label;
}

bool Element::visible(const View *view) const
{
    int bit = ~0;
    if (view)
        bit = view->typeBit();
    return (bit & m_viewBits);
}

void Element::setVisible(bool flag, int viewBits)
{
    if (flag)
    {
        m_viewBits |= viewBits;
    }
    else
    {
        m_viewBits &= ~viewBits;
    }
    manager()->queueUpdate(this, UpdateVisible);
}

bool Element::enabled() const
{
    return m_enabled;
}

void Element::setEnabled(bool flag)
{
    m_enabled = flag;
    manager()->queueUpdate(this, UpdateEnabled);
}

void Element::trigger() const
{
    manager()->setChanged();
    manager()->queueUpdate(this, UpdateNothing, true);
}

void Element::triggerImplementation() const
{
}

void Element::save(covise::TokenBuffer &buf) const
{
    buf << elementId();
    buf << m_viewBits;
    buf << m_enabled;
    buf << m_label;

}

void Element::load(covise::TokenBuffer &buf)
{
    int id;
    buf >> id;
    assert(m_id == id);
    buf >> m_viewBits;
    buf >> m_enabled;
    buf >> m_label;
}

void Element::updateSharedState()
{
    assert(!m_sharedState && "updating shared state of ui::Element requested, but sharing not implemented for Element type");
}

}
}
