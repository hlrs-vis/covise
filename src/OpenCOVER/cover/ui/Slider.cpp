#include "Slider.h"
#include "Group.h"
#include "Manager.h"
#include <cmath>

#include <net/tokenbuffer.h>

namespace opencover {
namespace ui {

Slider::Slider(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

Slider::Slider(Group *parent, const std::string &name)
: Element(parent, name)
{
}

Slider::~Slider()
{
    manager()->remove(this);
}

void Slider::setMoving(bool flag)
{
    m_moving = flag;
}

bool Slider::isMoving() const
{
    return m_moving;
}

Slider::Presentation Slider::presentation() const
{
    return m_presentation;
}

void Slider::setPresentation(Slider::Presentation pres)
{
    m_presentation = pres;
}

void Slider::setCallback(const std::function<void (Slider::ValueType, bool)> &f)
{
    m_callback = f;
}

std::function<void (Slider::ValueType, bool)> Slider::callback() const
{
    return m_callback;
}

void Slider::triggerImplementation() const
{
    if (m_callback)
        m_callback(value(), !isMoving());
}

void Slider::update() const
{
    Element::update();
    manager()->updateInteger(this);
    manager()->updateBounds(this);
    manager()->updateValue(this);
}

void Slider::save(covise::TokenBuffer &buf) const
{
    Element::save(buf);
    buf << m_min;
    buf << m_max;
    buf << m_value;
    buf << m_moving;
}

void Slider::load(covise::TokenBuffer &buf)
{
    Element::load(buf);
    buf >> m_min;
    buf >> m_max;
    buf >> m_value;
    buf >> m_moving;
}

Slider::ValueType Slider::value() const
{
    return m_value;
}

void Slider::setValue(Slider::ValueType val)
{
    m_value = val;
    if (m_integer)
    {
        m_value = std::round(m_value);
    }
    manager()->queueUpdate(this);
}

Slider::ValueType Slider::min() const
{
    return m_min;
}

Slider::ValueType Slider::max() const
{
    return m_max;
}

void Slider::setInteger(bool flag)
{
    m_integer = flag;
    manager()->updateInteger(this);
}

bool Slider::integer() const
{
    return m_integer;
}

void Slider::setBounds(Slider::ValueType min, Slider::ValueType max)
{
    m_min = min;
    m_max = max;
    if (m_integer)
    {
        m_min = std::floor(m_min);
        m_max = std::ceil(m_max);
    }
    manager()->queueUpdate(this);
}

}
}
