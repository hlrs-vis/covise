#include "Button.h"
#include "Group.h"
#include "ButtonGroup.h"
#include "Manager.h"

namespace opencover {
namespace ui {

Button::Button(const std::string &name, Owner *owner, ButtonGroup *bg, int id)
: Element(name, owner)
{
    setGroup(bg, id);
}

Button::Button(Group *parent, const std::string &name, ButtonGroup *bg, int id)
: Element(parent, name)
{
    setGroup(bg, id);
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
    if (m_radioGroup)
        m_radioGroup->remove(this);
    m_radioGroup = rg;
    if (m_radioGroup)
        m_radioGroup->add(this);
    m_id = id;
}

bool Button::state() const
{
    return m_state;
}

void Button::setState(bool flag, bool updateGroup)
{
    m_state = flag;
    if (updateGroup && m_radioGroup)
        m_radioGroup->toggle(this);
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
    if (group())
        group()->toggle(this);
    if (m_callback)
        m_callback(m_state);
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
