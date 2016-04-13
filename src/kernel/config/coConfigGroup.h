/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGGROUP_H
#define COCONFIGGROUP_H

#include <QHash>

#include <config/coConfigEntryString.h>
#include <config/coConfigEntry.h>
#include <config/coConfigRoot.h>
#include <util/coTypes.h>

#ifndef CO_gcc3
EXPORT_TEMPLATE2(template class CONFIGEXPORT QHash<QString, covise::coConfigRoot *>)
#endif

namespace covise
{

class CONFIGEXPORT coConfigGroup
{

public:
    coConfigGroup(const QString &groupName);
    virtual ~coConfigGroup();

    virtual coConfigEntryStringList getScopeList(const QString &section = 0,
                                                 const QString &variableName = 0) const;
    virtual coConfigEntryStringList getVariableList(const QString &section = 0) const;

    virtual coConfigEntryString getValue(const QString &variable,
                                         const QString &section,
                                         const QString &defaultValue) const;
    virtual coConfigEntryString getValue(const QString &variable,
                                         const QString &section) const;
    virtual coConfigEntryString getValue(const QString &simpleVariable) const;

    virtual const char *getEntry(const char *simpleVariable) const;

    virtual bool isOn(const QString &variable, const QString &section, bool defaultValue) const;
    virtual bool isOn(const QString &variable, const QString &section) const;
    virtual bool isOn(const QString &simpleVariable, bool defaultValue) const;
    virtual bool isOn(const QString &simpleVariable) const;

    virtual void setValue(const QString &variable, const QString &value,
                          const QString &section,
                          const QString &configuration = QString::null,
                          const QString &targetHost = QString::null, bool move = false);

    virtual void setValue(const QString &simpleVariable, const QString &value);

    virtual bool deleteValue(const QString &variable, const QString &section,
                             const QString &configuration = QString::null,
                             const QString &targetHost = QString::null);

    virtual bool deleteSection(const QString &section,
                               const QString &configuration = QString::null,
                               const QString &targetHost = QString::null);

    virtual QStringList getHostnameList() /*const*/;
    virtual QString getActiveHost() const;
    virtual bool setActiveHost(const QString &host);

    virtual QStringList getClusterList() /*const*/;
    virtual QString getActiveCluster() const;
    virtual bool setActiveCluster(const QString &master);

    virtual const QString &getGroupName() const;

    virtual void reload();
    //virtual bool save(const QString & filename, ConfigScope scope = Global) const;

    virtual coConfigRoot *addConfig(const QString &filename, const QString &name, bool create = false);
    virtual void removeConfig(const QString &name);

    virtual bool save(const QString &filename = QString::null) const;

    void setReadOnly(const QString &config, bool ro);
    void setReadOnly(bool ro);

    bool isReadOnly(const QString &config) const;
    bool isReadOnly() const;

    virtual coConfigGroup *clone() const;
    virtual void merge(const coConfigGroup *with);
    void flatten();

private:
    coConfigGroup(const coConfigGroup *source);

    QString activeHostname;
    QStringList hostnames;
    QString activeCluster;
    QStringList masternames;
    QString groupName;

    bool readOnly;
    //friend  QHash<QString, coConfigEntry*> mainWindow::loadFile(const QString & fileName);
    QHash<QString, coConfigRoot *> configs;
};
}
#endif
