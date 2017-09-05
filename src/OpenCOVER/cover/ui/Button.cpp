#include "Button.h"
#include "Group.h"
#include "ButtonGroup.h"
#include "Manager.h"

namespace opencover {
namespace ui {

Button::Button(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

Button::Button(Group *parent, const std::string &name)
: Element(parent, name)
{
}

Button::Button(ButtonGroup *parent, const std::string &name, int id)
: Element(parent, name)
, m_radioGroup(parent)
, m_id(id)
{
}

int Button::id() const
{
    return m_id;
}

ButtonGroup *Button::group() const
{
    return m_radioGroup;
}

void Button::setGroup(ButtonGroup *rg, int id)
{
    m_radioGroup = rg;
    m_id = id;
}

bool Button::state() const
{
    return m_state;
}

void Button::setState(bool flag)
{
    m_state = flag;
    manager()->updateState(this);
}

void Button::setCallback(const std::function<void(bool)> &f)
{
    m_callback = f;
}

std::function<void(bool)> Button::callback() const
{
    return m_callback;
}

void Button::triggerImplementation() const
{
    if (m_callback)
        m_callback(m_state);
    if (group())
        group()->toggle(this);
}

void Button::radioTrigger() const
{
    if (m_callback)
        m_callback(m_state);
}

void Button::update() const
{
    Element::update();
    manager()->updateState(this);
}

void Button::shortcutTriggered()
{
    setState(!state());
    trigger();
}

}
}
