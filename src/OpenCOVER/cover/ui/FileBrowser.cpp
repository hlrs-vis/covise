#include "FileBrowser.h"
#include "Manager.h"
#include <net/tokenbuffer.h>

#include <sstream>
#include <cstdlib>

namespace opencover {
namespace ui {

FileBrowser::FileBrowser(Group *parent, const std::string &name, bool save)
: Element(parent, name)
, m_save(save)
{
}

FileBrowser::FileBrowser(const std::string &name, Owner *owner, bool save)
: Element(name, owner)
, m_save(save)
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

std::string FileBrowser::value() const
{
    return m_value;
}

void FileBrowser::setFilter(const std::string &filter)
{
    if (m_filter != filter)
    {
        m_filter = filter;
        manager()->queueUpdate(this, UpdateFilter);
    }
}

std::string FileBrowser::filter() const
{
    return m_filter;
}

bool FileBrowser::forSaving() const
{
    return m_save;
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
    if (mask & UpdateFilter)
        manager()->updateFilter(this);
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
