/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGVALUE_H
#define COCONFIGVALUE_H

#include <string>
#include <util/coTypes.h>

namespace covise
{

class coConfigGroup;

template <class T>
class coConfigValue
{

public:
    coConfigValue(const std::string &configGroupName, const std::string &variable, const std::string &section);
    coConfigValue(const std::string &variable, const std::string &section);
    coConfigValue(const std::string &simpleVariable);
    coConfigValue(coConfigGroup *group,
                  const std::string &variable, const std::string &section);
    coConfigValue(coConfigGroup *group,
                  const std::string &simpleVariable);
    coConfigValue(const coConfigValue<T> &value);

    virtual ~coConfigValue() = default;

    virtual void update();
    virtual bool hasValidValue();
    virtual bool hasValidValue() const;

    coConfigValue<T> &operator=(const T &);
    virtual operator T();

    virtual bool operator==(const coConfigValue<T> &);
    virtual bool operator!=(const coConfigValue<T> &);

    virtual void setSaveToGroup(coConfigGroup *group);
    virtual coConfigGroup *getSaveToGroup() const;

    virtual void setAutoUpdate(bool update);
    virtual bool isAutoUpdate() const;

    virtual bool isModified() const;

protected:
    virtual T fromString(const std::string &value) const = 0;
    virtual std::string toString(const T &value) const = 0;

    std::string variable;
    std::string section;
    std::string value;
    std::string unmodifiedValue;
    std::string configGroupName;

    bool autoUpdate;
    bool modified;

    coConfigGroup *group = nullptr;

#ifdef COCONFIGVALUE_USE_CACHE
    T cache;
#endif
};
}
//#include "coConfigValue.inl"
#endif
