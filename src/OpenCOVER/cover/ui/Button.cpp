#include "Button.h"
#include "Group.h"
#include "ButtonGroup.h"
#include "Manager.h"

#include <vrbclient/SharedState.h>

#include <net/tokenbuffer.h>


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

Button::~Button()
{
    manager()->remove(this);
    setGroup(nullptr, -1);
}

int Button::buttonId() const
{
    return m_buttonId;
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
    m_buttonId = id;
}

bool Button::state() const
{
    return m_state;
}

void Button::setState(bool flag, bool updateGroup)
{
    if (flag != m_state)
    {
        m_state = flag;
        updateSharedState();
        manager()->queueUpdate(this, UpdateState);
    }
    if (updateGroup && group())
        group()->toggle(this);
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
}

void Button::radioTrigger() const
{
    //std::cerr << "radioTrigger, state=" << state() << ": " << this->path() << std::endl;
    if (m_callback)
        m_callback(m_state);
}

void Button::update(Element::UpdateMaskType mask) const
{
    Element::update(mask);
    if (mask & UpdateState)
        manager()->updateState(this);
}

void Button::save(covise::TokenBuffer &buf) const
{
    Element::save(buf);
    buf << m_state;
}

void Button::load(covise::TokenBuffer &buf)
{
    Element::load(buf);
    buf >> m_state;
    updateSharedState();
    //std::cerr << "Button::load " << path() << ": state=" << m_state << std::endl;
}

void Button::setShared(bool shared)
{
    if (shared)
    {
        if (!m_sharedState)
        {
            m_sharedState.reset(new SharedValue("ui."+path(), m_state));
            m_sharedState->setUpdateFunction([this](){
                setState(*static_cast<SharedValue *>(m_sharedState.get()));
                triggerImplementation();
            });
        }
    }
    else
    {
        m_sharedState.reset();
    }
}

void Button::updateSharedState()
{
    if (auto st = static_cast<SharedValue *>(m_sharedState.get()))
    {
        *st = m_state;
    }
}

void Button::shortcutTriggered()
{
    setState(!state());
    trigger();
}

}
}
