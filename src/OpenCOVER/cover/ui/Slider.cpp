#include "Slider.h"
#include "Group.h"
#include "Manager.h"
#include <cmath>
#include <algorithm>

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

void Slider::setScale(Slider::Scale scale)
{
    m_scale = scale;
    if (scale == Logarithmic)
    {
        if (m_min < std::numeric_limits<float>::min())
        {
            m_min = std::numeric_limits<float>::min();
            std::cerr << "ui::Slider: lower bound of " << path() << " changed for log scale" << std::endl;
        }
        m_max = std::max(m_min, m_max);
    }
    manager()->updateScale(this);
}

Slider::Scale Slider::scale() const
{
    return m_scale;
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
    manager()->updateScale(this);
    manager()->updateIntegral(this);
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
    if (m_integral)
    {
        m_value = std::round(m_value);
    }
    manager()->queueUpdate(this);
}

void Slider::setLinValue(Slider::ValueType val)
{
    if (scale() == Logarithmic)
        setValue(std::pow(10., val));
    else
        setValue(val);
}

Slider::ValueType Slider::min() const
{
    return m_min;
}

Slider::ValueType Slider::max() const
{
    return m_max;
}

Slider::ValueType Slider::linMin() const
{
    if (scale() == Logarithmic)
    {
        auto m = std::max(static_cast<ValueType>(m_min), std::numeric_limits<ValueType>::min());
        return std::log10(m);
    }
    return min();
}

Slider::ValueType Slider::linMax() const
{
    if (scale() == Logarithmic)
    {
        auto m = std::max(static_cast<ValueType>(m_min), std::numeric_limits<ValueType>::min());
        m = std::max(m , m_max);
        return std::log10(m);
    }
    return max();
}

Slider::ValueType Slider::linValue() const
{
    if (scale() == Logarithmic)
    {
        auto m = std::max(static_cast<ValueType>(m_min), std::numeric_limits<ValueType>::min());
        auto v = std::max(m , value());
        return std::log10(v);
    }
    return value();
}

void Slider::setIntegral(bool flag)
{
    m_integral = flag;
    manager()->updateIntegral(this);
}

bool Slider::integral() const
{
    return m_integral;
}

void Slider::setBounds(Slider::ValueType min, Slider::ValueType max)
{
    m_min = min;
    m_max = max;
    if (m_integral)
    {
        m_min = std::floor(m_min);
        m_max = std::ceil(m_max);
    }
    manager()->queueUpdate(this);
}

}
}
