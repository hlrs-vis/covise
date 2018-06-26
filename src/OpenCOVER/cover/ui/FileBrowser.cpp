#include "FileBrowser.h"
#include "Manager.h"
#include <net/tokenbuffer.h>

#include <sstream>
#include <cstdlib>

namespace opencover {
namespace ui {

FileBrowser::FileBrowser(Group *parent, const std::string &name)
: Element(parent, name)
{
}

FileBrowser::FileBrowser(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

FileBrowser::~FileBrowser()
{
    manager()->remove(this);
}

void opencover::ui::FileBrowser::setValue(const std::string &text)
{
    if (m_value != text)
    {
        m_value = text;
        manager()->queueUpdate(this, UpdateValue);
    }
}

void FileBrowser::setValue(double num)
{
    std::stringstream str;
    str << num;

    if (m_value != str.str())
    {
        m_value = str.str();
        manager()->queueUpdate(this, UpdateValue);
    }
}

double FileBrowser::number() const
{
    return atof(m_value.c_str());
}

std::string FileBrowser::value() const
{
    return m_value;
}

void FileBrowser::setCallback(const std::function<void (const std::string &)> &f)
{
    m_callback = f;
}

std::function<void (const std::string &)> FileBrowser::callback() const
{
    return m_callback;
}

void FileBrowser::triggerImplementation() const
{
    if (m_callback)
        m_callback(m_value);
}

void FileBrowser::update(Element::UpdateMaskType mask) const
{
    Element::update(mask);
    if (mask & UpdateValue)
        manager()->updateValue(this);
}

void FileBrowser::save(covise::TokenBuffer &buf) const
{
    Element::save(buf);
    buf << m_value;
}

void FileBrowser::load(covise::TokenBuffer &buf)
{
    Element::load(buf);
    buf >> m_value;
}

}
}
