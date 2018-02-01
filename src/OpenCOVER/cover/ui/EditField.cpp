#include "EditField.h"
#include "Manager.h"
#include <net/tokenbuffer.h>

#include <sstream>
#include <cstdlib>

namespace opencover {
namespace ui {

EditField::EditField(Group *parent, const std::string &name)
: Element(parent, name)
{
}

EditField::EditField(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

EditField::~EditField()
{
    manager()->remove(this);
}

void opencover::ui::EditField::setValue(const std::string &text)
{
    if (m_value != text)
    {
        m_value = text;
        manager()->queueUpdate(this, UpdateValue);
    }
}

void EditField::setValue(double num)
{
    std::stringstream str;
    str << num;

    if (m_value != str.str())
    {
        m_value = str.str();
        manager()->queueUpdate(this, UpdateValue);
    }
}

double EditField::number() const
{
    return atof(m_value.c_str());
}

std::string EditField::value() const
{
    return m_value;
}

void EditField::setCallback(const std::function<void (const std::string &)> &f)
{
    m_callback = f;
}

std::function<void (const std::string &)> EditField::callback() const
{
    return m_callback;
}

void EditField::triggerImplementation() const
{
    if (m_callback)
        m_callback(m_value);
}

void EditField::update(Element::UpdateMaskType mask) const
{
    Element::update(mask);
    if (mask & UpdateValue)
        manager()->updateValue(this);
}

void EditField::save(covise::TokenBuffer &buf) const
{
    Element::save(buf);
    buf << m_value;
}

void EditField::load(covise::TokenBuffer &buf)
{
    Element::load(buf);
    buf >> m_value;
}

}
}
