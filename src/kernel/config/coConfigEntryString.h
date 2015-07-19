/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGENTRYSTRING_H
#define COCONFIGENTRYSTRING_H

#include <QStringList>
#include <QTextStream>
#include <QLinkedList>

class QRegExp;

#include "coConfigConstants.h"
#include <util/coTypes.h>

namespace covise
{

class CONFIGEXPORT coConfigEntryString : public QString
{

public:
    coConfigEntryString(const QString &string = QString::null,
                        const coConfigConstants::ConfigScope scope = coConfigConstants::Default,
                        const QString &configName = "",
                        bool isListItem = false);

    virtual ~coConfigEntryString();

    coConfigConstants::ConfigScope getConfigScope() const;
    coConfigEntryString &setConfigScope(coConfigConstants::ConfigScope scope);

    const QString &getConfigName() const;
    coConfigEntryString &setConfigName(const QString &name);

    const QString &getConfigGroupName() const;
    coConfigEntryString &setConfigGroupName(const QString &name);

    bool isListItem() const;
    void setListItem(bool on);

private:
    QString configName;
    QString configGroupName;
    coConfigConstants::ConfigScope configScope;
    bool listItem;
};

class CONFIGEXPORT coConfigEntryStringList : public QLinkedList<coConfigEntryString>
{

public:
    coConfigEntryStringList();
    ~coConfigEntryStringList();

    enum ListType
    {
        UNKNOWN = 0,
        VARIABLE,
        PLAIN_LIST
    };

    coConfigEntryStringList merge(coConfigEntryStringList &list);
    coConfigEntryStringList filter(const QRegExp &filter) const;

    operator QStringList();

    ListType getListType() const;
    void setListType(ListType listType);

private:
    ListType listType;
};

QTextStream &operator<<(QTextStream &out, const coConfigEntryStringList list);
}
#endif
