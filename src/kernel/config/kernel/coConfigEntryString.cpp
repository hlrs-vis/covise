/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigLog.h>
#include <config/coConfigEntryString.h>

using std::ostream;
using namespace covise;

coConfigEntryString::coConfigEntryString(const std::string &entry, const std::string &configName, const std::string &configGroupName, coConfigConstants::ConfigScope scope, bool isListItem)
    : entry(entry), configName(configName), configGroupName(configGroupName), configScope(scope), islistItem(isListItem)
{
}

bool covise::operator<(const coConfigEntryString &first, const coConfigEntryString &second)
{
    return first.entry < second.entry || (first.entry == second.entry && first.configName < second.configName) || (first.configName == second.configName && first.configGroupName < second.configGroupName);
}

coConfigEntryStringList &coConfigEntryStringList::merge(const coConfigEntryStringList &list)
{
    if(!list.entries().empty())
    {
        m_listType = list.m_listType;
        m_entries.insert(list.m_entries.begin(), list.m_entries.end());
    }
    return *this;
}

coConfigEntryStringList coConfigEntryStringList::filter(const std::regex &filter) const
{

    COCONFIGDBG("coConfigEntryStringList::filter info: filtering");

    coConfigEntryStringList list;

    for (const auto &configEntry : m_entries)
    {
        if (std::regex_search(configEntry.entry, filter))
            list.m_entries.insert(configEntry);
    }

    list.setListType(m_listType);

    return list;
}

coConfigEntryStringList::ListType coConfigEntryStringList::getListType() const
{
    COCONFIGDBG("coConfigEntryStringList::getListType info: type is " << m_listType);
    return m_listType;
}

void coConfigEntryStringList::setListType(coConfigEntryStringList::ListType listType)
{
    m_listType = listType;
    COCONFIGDBG("coConfigEntryStringList::setListType info: new type is " << m_listType);
}

std::set<coConfigEntryString> &coConfigEntryStringList::entries()
{
    return m_entries;
}

const std::set<coConfigEntryString> &coConfigEntryStringList::entries() const
{
    return m_entries;
}

namespace covise
{

    std::stringstream &operator<<(std::stringstream &out, const coConfigEntryStringList list)
    {

        out << "[";

        auto iterator = list.entries().begin();
        while (iterator != list.entries().end())
        {

            out << iterator->entry;
            iterator++;
            if (iterator != list.entries().end())
                out << ",";
        }
        out << "]";
        return out;
    }
}
