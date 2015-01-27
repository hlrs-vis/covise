/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGROOT_H
#define COCONFIGROOT_H

#include <QFile>
#include <QHash>
#include <QSet>

#include <config/coConfigEntry.h>
#include <config/coConfigEntryString.h>

#include <util/coTypes.h>

#ifndef CO_gcc3
EXPORT_TEMPLATE2(template class CONFIGEXPORT QHash<QString, covise::coConfigEntry *>)
#endif

namespace covise
{

class coConfigGroup;

class CONFIGEXPORT coConfigRoot
{
    friend class coConfigXercesRoot;

public:
    static coConfigRoot *createNew(const QString &name, const QString &filename, bool create = false, coConfigGroup *group = NULL);

    coConfigRoot(const QString &name, const QString &filename, bool create = false, coConfigGroup *group = NULL);
    virtual ~coConfigRoot();

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
                          const QString &targetHost = 0, bool move = false);

    virtual bool deleteValue(const QString &variable, const QString &section,
                             const QString &targetHost = 0);

    virtual bool deleteSection(const QString &section, const QString &targetHost = 0);

    virtual QStringList getHostnameList() const;
    virtual QString getActiveHost() const;
    virtual bool setActiveHost(const QString &host);

    virtual const QString &getConfigName() const;

    virtual void reload();
    virtual bool save(const QString &filename = QString::null) const;

    void setGroup(coConfigGroup *group);

    void setReadOnly(bool ro);
    bool isReadOnly() const;

    QStringList getHosts();
    coConfigEntry *getConfigForHost(const QString &hostname);
    coConfigEntry *getGlobalConfig()
    {
        return this->globalConfig;
    }

    //      QHash<QString, coConfigEntry*> getEntryList();

    virtual coConfigRoot *clone() const = 0;
    virtual void merge(const coConfigRoot *with) = 0;

protected:
    virtual void load(bool create = false) = 0;

    void init();

    QString findConfigFile(const QString &filename, bool preferGlobal = false);

    void findLocalConfig(const QString &filename, QFile &target);
    void findGlobalConfig(const QString &filename, QFile &target);

    void clear();

    coConfigGroup *group;

    QString filename;
    QString activeHostname;
    QStringList hostnames;

    QString configName;

    coConfigEntry *globalConfig;
    coConfigEntry *hostConfig;

    QHash<QString, coConfigEntry *> hostConfigs;

    virtual void createGlobalConfig() = 0;
    virtual void createHostConfig(const QString &hostname) = 0;

    bool create;
    bool readOnly;

    QSet<QString> included;
};
}
#endif
