#include "Input.h"
#include "Manager.h"
#include <net/tokenbuffer.h>

#include <sstream>
#include <cstdlib>

namespace opencover {
namespace ui {

Input::Input(Group *parent, const std::string &name)
: Element(parent, name)
{
}

Input::Input(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

Input::~Input()
{
    manager()->remove(this);
}

void opencover::ui::Input::setValue(const std::string &text)
{
    if (m_value != text)
    {
        m_value = text;
        manager()->queueUpdate(this, UpdateValue);
    }
}

void Input::setValue(double num)
{
    std::stringstream str;
    str << num;

    if (m_value != str.str())
    {
        m_value = str.str();
        manager()->queueUpdate(this, UpdateValue);
    }
}

double Input::number() const
{
    return atof(m_value.c_str());
}

std::string Input::value() const
{
    return m_value;
}

void Input::setCallback(const std::function<void (const std::string &)> &f)
{
    m_callback = f;
}

std::function<void (const std::string &)> Input::callback() const
{
    return m_callback;
}

void Input::triggerImplementation() const
{
    if (m_callback)
        m_callback(m_value);
}

void Input::update(Element::UpdateMaskType mask) const
{
    Element::update(mask);
    if (mask & UpdateValue)
        manager()->updateValue(this);
}

void Input::save(covise::TokenBuffer &buf) const
{
    Element::save(buf);
    buf << m_value;
}

void Input::load(covise::TokenBuffer &buf)
{
    Element::load(buf);
    buf >> m_value;
}

}
}
