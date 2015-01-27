/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGSTRING_H
#define COCONFIGSTRING_H

#include "coConfigValue.h"

namespace covise
{

class coConfigGroup;

class CONFIGEXPORT coConfigString : public coConfigValue<QString>
{

public:
    coConfigString(const QString &configGroupName, const QString &variable, const QString &section);
    coConfigString(const QString &variable, const QString &section);
    coConfigString(const QString &simpleVariable);
    coConfigString(coConfigGroup *group,
                   const QString &variable, const QString &section);
    coConfigString(coConfigGroup *group, const QString &simpleVariable);
    coConfigString(const coConfigString &value);

    virtual ~coConfigString();

    coConfigString &operator=(QString);

protected:
    virtual QString fromString(const QString &value) const;
    virtual QString toString(const QString &value) const;
};
}
#endif
