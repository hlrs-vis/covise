#include "Action.h"
#include "Manager.h"

namespace opencover {
namespace ui {

Action::Action(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

Action::Action(Group *parent, const std::string &name)
: Element(parent, name)
{
}

Action::~Action()
{
    manager()->remove(this);
}

void Action::setCallback(const std::function<void ()> &f)
{
    m_callback = f;
}

std::function<void ()> Action::callback() const
{
    return m_callback;
}

void Action::triggerImplementation() const
{
    if (m_callback)
        m_callback();
}

void Action::shortcutTriggered()
{
    trigger();
}

}
}
