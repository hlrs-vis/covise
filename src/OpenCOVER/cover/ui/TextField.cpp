#include "TextField.h"
#include "Manager.h"
#include <net/tokenbuffer.h>
#include <vrbclient/SharedState.h>

#include <sstream>
#include <cstdlib>

namespace opencover {
namespace ui {

TextField::TextField(Group *parent, const std::string &name)
: Element(parent, name)
{
}

TextField::TextField(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

TextField::~TextField()
{
    manager()->remove(this);
}

void opencover::ui::TextField::setValue(const std::string &text)
{
    if (m_value != text)
    {
        m_value = text;
        updateSharedState();
        manager()->queueUpdate(this, UpdateValue);
    }
}

std::string TextField::value() const
{
    return m_value;
}

void TextField::setCallback(const std::function<void (const std::string &)> &f)
{
    m_callback = f;
}

std::function<void (const std::string &)> TextField::callback() const
{
    return m_callback;
}

void TextField::triggerImplementation() const
{
    if (m_callback)
        m_callback(m_value);
}

void TextField::update(Element::UpdateMaskType mask) const
{
    Element::update(mask);
    if (mask & UpdateValue)
        manager()->updateValue(this);
}

void TextField::save(covise::TokenBuffer &buf) const
{
    Element::save(buf);
    buf << m_value;
}

void TextField::load(covise::TokenBuffer &buf)
{
    Element::load(buf);
    buf >> m_value;
    updateSharedState();
}

void TextField::setShared(bool shared)
{
    if (shared)
    {
        if (!m_sharedState)
        {
            m_sharedState.reset(new SharedValue("ui."+path(), m_value));
            m_sharedState->setUpdateFunction([this](){
                setValue(*static_cast<SharedValue *>(m_sharedState.get()));
                triggerImplementation();
            });
        }
    }
    else
    {
        m_sharedState.reset();
    }
}

void TextField::updateSharedState()
{
    if (auto st = static_cast<SharedValue *>(m_sharedState.get()))
    {
        *st = m_value;
    }
}

}
}
