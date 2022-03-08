/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGENTRYSTRING_H
#define COCONFIGENTRYSTRING_H

#include <sstream>
#include <list>

#include <regex>

#include "coConfigConstants.h"
#include <util/coTypes.h>

namespace covise
{

    struct CONFIGEXPORT coConfigEntryString
    {
        // apperently c++11 cant figger out how braced enclosed initialization works
        coConfigEntryString(const std::string &entry = "", const std::string &configName = "", const std::string &configGroupName = "", coConfigConstants::ConfigScope scope = coConfigConstants::ConfigScope::Default, bool isListItem = false);
        std::string entry;
        std::string configName;
        std::string configGroupName;
        coConfigConstants::ConfigScope configScope = coConfigConstants::Default;
        bool islistItem = false;
    };

    bool CONFIGEXPORT operator<(const coConfigEntryString &first, const coConfigEntryString &second);
    class CONFIGEXPORT coConfigEntryStringList
    {
    public:
        enum ListType
        {
            UNKNOWN = 0,
            VARIABLE,
            PLAIN_LIST
        };

        coConfigEntryStringList &merge(const coConfigEntryStringList &list);
        coConfigEntryStringList filter(const std::regex &filter) const;

        ListType getListType() const;
        void setListType(ListType listType);
        std::set<coConfigEntryString> &entries();
        const std::set<coConfigEntryString> &entries() const;

    private:
        std::set<coConfigEntryString> m_entries;
        ListType m_listType = ListType::UNKNOWN;
};

std::stringstream &operator<<(std::stringstream &out, const coConfigEntryStringList list);
}
#endif
