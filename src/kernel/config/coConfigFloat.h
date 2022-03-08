/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGFLOAT_H
#define COCONFIGFLOAT_H

#define COCONFIGVALUE_USE_CACHE

#include "coConfigValue.h"
#include <util/coTypes.h>

namespace covise
{

class coConfigGroup;

class CONFIGEXPORT coConfigFloat : public coConfigValue<float>
{

public:
    coConfigFloat(const std::string &configGroupName, const std::string &variable, const std::string &section);
    coConfigFloat(const std::string &variable, const std::string &section);
    coConfigFloat(const std::string &simpleVariable);
    coConfigFloat(coConfigGroup *group,
                  const std::string &variable, const std::string &section);
    coConfigFloat(coConfigGroup *group, const std::string &simpleVariable);
    coConfigFloat(const coConfigFloat &value);

    virtual ~coConfigFloat();

    coConfigFloat &operator=(float);

protected:
    virtual float fromString(const std::string &value) const;
    virtual std::string toString(const float &value) const;
};
}
#undef COCONFIGVALUE_USE_CACHE
#endif
