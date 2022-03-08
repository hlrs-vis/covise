/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGLONG_H
#define COCONFIGLONG_H

#define COCONFIGVALUE_USE_CACHE

#include "coConfigValue.h"
#include <util/coTypes.h>

namespace covise
{

class coConfigGroup;

class CONFIGEXPORT coConfigLong : public coConfigValue<long>
{

public:
    coConfigLong(const std::string &configGroupName, const std::string &variable, const std::string &section);
    coConfigLong(const std::string &variable, const std::string &section);
    coConfigLong(const std::string &simpleVariable);
    coConfigLong(coConfigGroup *group,
                 const std::string &variable, const std::string &section);
    coConfigLong(coConfigGroup *group, const std::string &simpleVariable);
    coConfigLong(const coConfigLong &value);

    virtual ~coConfigLong();

    coConfigLong &operator=(long);

protected:
    virtual long fromString(const std::string &value) const;
    virtual std::string toString(const long &value) const;
};
}
#undef COCONFIGVALUE_USE_CACHE
#endif
