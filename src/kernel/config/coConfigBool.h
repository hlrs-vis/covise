/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGBOOL_H
#define COCONFIGBOOL_H

#define COCONFIGVALUE_USE_CACHE

#include "coConfigValue.h"
#include <util/coTypes.h>

namespace covise
{

class coConfigGroup;

class CONFIGEXPORT coConfigBool : public coConfigValue<bool>
{

public:
    coConfigBool(const std::string &configGroupName, const std::string &variable, const std::string &section);
    coConfigBool(const std::string &variable, const std::string &section);
    coConfigBool(const std::string &simpleVariabl);
    coConfigBool(coConfigGroup *group,
                 const std::string &variable, const std::string &section);
    coConfigBool(coConfigGroup *group, const std::string &simpleVariable);
    coConfigBool(const coConfigBool &value);

    virtual ~coConfigBool();

    coConfigBool &operator=(bool);

protected:
    virtual bool fromString(const std::string &value) const;
    virtual std::string toString(const bool &value) const;
};
}
#undef COCONFIGVALUE_USE_CACHE
#endif
