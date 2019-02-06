#include "FileBrowser.h"
#include "Manager.h"
#include <net/tokenbuffer.h>

#include <sstream>
#include <cstdlib>

namespace opencover {
namespace ui {

FileBrowser::FileBrowser(Group *parent, const std::string &name, bool save)
: TextField(parent, name)
, m_save(save)
{
}

FileBrowser::FileBrowser(const std::string &name, Owner *owner, bool save)
: TextField(name, owner)
, m_save(save)
{
}

FileBrowser::~FileBrowser()
{
    manager()->remove(this);
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

void FileBrowser::update(Element::UpdateMaskType mask) const
{
    TextField::update(mask);
    if (mask & UpdateFilter)
        manager()->updateFilter(this);
}

}
}
