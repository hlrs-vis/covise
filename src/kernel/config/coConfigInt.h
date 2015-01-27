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
    coConfigInt(const QString &configGroupName, const QString &variable, const QString &section);
    coConfigInt(const QString &variable, const QString &section);
    coConfigInt(const QString &simpleVariable);
    coConfigInt(coConfigGroup *group,
                const QString &variable, const QString &section);
    coConfigInt(coConfigGroup *group, const QString &simpleVariable);
    coConfigInt(const coConfigInt &value);

    virtual ~coConfigInt();

    coConfigInt &operator=(int);

protected:
    virtual int fromString(const QString &value) const;
    virtual QString toString(const int &value) const;
};
}
#undef COCONFIGVALUE_USE_CACHE
#endif
