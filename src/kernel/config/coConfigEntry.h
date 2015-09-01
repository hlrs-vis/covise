/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGENTRY_H
#define COCONFIGENTRY_H

#include "coConfigConstants.h"
#include "coConfigEntryString.h"
#include "coConfigEntryPtrList.h"
#include "coConfigSchemaInfos.h"

#include <QHash>
#include <QObject>
#include <QString>
#include <QRegExp>
#include <QList>

#include <util/coTypes.h>

#include "coConfigEditorController.h"

#ifndef CO_gcc3
EXPORT_TEMPLATE2(template class CONFIGEXPORT QHash<QString, QString *>)
#endif

namespace covise
{

class CONFIGEXPORT coConfigEntry : public Subject<coConfigEntry>
{
    friend class coConfigXercesEntry;

public:
    coConfigEntry();
    virtual ~coConfigEntry();

    coConfigEntryStringList getScopeList(QString scope);
    coConfigEntryStringList getVariableList(QString scope);

    coConfigEntryString getValue(const QString &variable, QString scope);
    //coConfigEntryStringList getValues(const QString & variable, QString scope);

    const char *getEntry(const char *variable);

    bool setValue(const QString &variable, const QString &value,
                  const QString &section);
    void addValue(const QString &variable, const QString &value,
                  const QString &section);

    bool deleteValue(const QString &variable, const QString &section);
    bool deleteSection(const QString &section);

    bool hasValues() const;

    const QString &getPath() const;
    QString getName() const;
    const char *getCName() const;
    const QString &getConfigName() const;

    bool isList() const;
    bool hasChildren() const;

    void setReadOnly(bool ro);
    bool isReadOnly() const;

    static QString &cleanName(QString &name);

    coConfigSchemaInfos *getSchemaInfos();
    void setSchemaInfos(coConfigSchemaInfos *infos);

    void entryChanged();

    virtual void merge(const coConfigEntry *with);
    virtual coConfigEntry *clone() const = 0;

protected:
    coConfigEntry(const coConfigEntry *entry);

    void setPath(const QString &path);
    void makeSection(const QString &section);

private:
    bool matchingAttributes() const;
    bool matchingArch() const;
    bool matchingRank() const;

    coConfigConstants::ConfigScope configScope;
    QString configName;
    QString path;

    bool isListNode;
    bool readOnly;

    coConfigEntryPtrList children;
    QHash<QString, QString *> attributes;
    QStringList textNodes;

    coConfigSchemaInfos *schemaInfos;
    QString elementGroup;

    //TODO How to get rid of this friend...
    friend class coConfigEntryToEditor;
    mutable char *cName;
    mutable QString name;
};
}
#endif
