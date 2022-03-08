/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGTOOLS_H
#define COCONFIGTOOLS_H
#include <util/coExport.h>
#include <string>
#include <map>
#include <array>
class CONFIGEXPORT coConfigTools
{
 public:
    static const std::array<const char*, 4> attributeNames;
    static bool matchingAttributes(const std::map<std::string, std::string> &attributes);
    static bool matchingHost(const std::string &host);
    static bool matchingMaster(const std::string &master);
    static bool matchingArch(const std::string &arch);
    static bool matchingRank(const std::string &rank);
};
#endif
