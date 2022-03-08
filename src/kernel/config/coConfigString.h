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

class CONFIGEXPORT coConfigString : public coConfigValue<std::string>
{

public:
    coConfigString(const std::string &configGroupName, const std::string &variable, const std::string &section);
    coConfigString(const std::string &variable, const std::string &section);
    coConfigString(const std::string &simpleVariable);
    coConfigString(coConfigGroup *group,
                   const std::string &variable, const std::string &section);
    coConfigString(coConfigGroup *group, const std::string &simpleVariable);
    coConfigString(const coConfigString &value);

    virtual ~coConfigString();

    coConfigString &operator=(std::string);

protected:
    virtual std::string fromString(const std::string &value) const;
    virtual std::string toString(const std::string &value) const;
};
}
#endif
