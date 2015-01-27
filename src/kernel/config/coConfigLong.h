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
    coConfigLong(const QString &configGroupName, const QString &variable, const QString &section);
    coConfigLong(const QString &variable, const QString &section);
    coConfigLong(const QString &simpleVariable);
    coConfigLong(coConfigGroup *group,
                 const QString &variable, const QString &section);
    coConfigLong(coConfigGroup *group, const QString &simpleVariable);
    coConfigLong(const coConfigLong &value);

    virtual ~coConfigLong();

    coConfigLong &operator=(long);

protected:
    virtual long fromString(const QString &value) const;
    virtual QString toString(const long &value) const;
};
}
#undef COCONFIGVALUE_USE_CACHE
#endif
