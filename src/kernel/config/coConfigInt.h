/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGINT_H
#define COCONFIGINT_H

#define COCONFIGVALUE_USE_CACHE

#include "coConfigValue.h"
#include <util/coTypes.h>

namespace covise
{

class coConfigGroup;

class CONFIGEXPORT coConfigInt : public coConfigValue<int>
{

public:
    coConfigInt(const std::string &configGroupName, const std::string &variable, const std::string &section);
    coConfigInt(const std::string &variable, const std::string &section);
    coConfigInt(const std::string &simpleVariable);
    coConfigInt(coConfigGroup *group,
                const std::string &variable, const std::string &section);
    coConfigInt(coConfigGroup *group, const std::string &simpleVariable);
    coConfigInt(const coConfigInt &value);

    virtual ~coConfigInt() = default;

    coConfigInt &operator=(int);

protected:
    virtual int fromString(const std::string &value) const;
    virtual std::string toString(const int &value) const;
};
}
#undef COCONFIGVALUE_USE_CACHE
#endif
