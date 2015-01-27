/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGVALUE_H
#define COCONFIGVALUE_H

#include <QString>
#include <util/coTypes.h>

namespace covise
{

class coConfigGroup;

template <class T>
class coConfigValue
{

public:
    coConfigValue(const QString &configGroupName, const QString &variable, const QString &section);
    coConfigValue(const QString &variable, const QString &section);
    coConfigValue(const QString &simpleVariable);
    coConfigValue(coConfigGroup *group,
                  const QString &variable, const QString &section);
    coConfigValue(coConfigGroup *group,
                  const QString &simpleVariable);
    coConfigValue(const coConfigValue<T> &value);

    virtual ~coConfigValue();

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
    virtual T fromString(const QString &value) const = 0;
    virtual QString toString(const T &value) const = 0;

    QString variable;
    QString section;
    QString value;
    QString unmodifiedValue;
    QString configGroupName;

    bool autoUpdate;
    bool modified;

    coConfigGroup *group;
    coConfigGroup *saveToGroup;

#ifdef COCONFIGVALUE_USE_CACHE
    T cache;
#endif
};
}
//#include "coConfigValue.inl"
#endif
