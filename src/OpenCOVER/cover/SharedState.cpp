/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedState.h"

namespace opencover
{
template <>
void serialize<std::vector<std::string>>(covise::TokenBuffer &tb, const std::vector<std::string> &value)
{
    uint32_t size = value.size();
    tb << size;
    for (size_t i = 0; i < size; i++)
    {
        tb << value[i];
    }
}

template <>
void deserialize<std::vector<std::string>>(covise::TokenBuffer &tb, std::vector<std::string> &value)
{
    uint32_t size;
    tb >> size;
    value.clear();
    value.resize(size);
    for (size_t i = 0; i < size; i++)
    {
        std::string path;
        tb >> path;
        value[i] = path;
    }
}

}