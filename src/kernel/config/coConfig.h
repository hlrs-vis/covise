/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIG_H
#define COCONFIG_H

#include <QHash>

#include "coConfigBool.h"
#include "coConfigConstants.h"
#include "coConfigFloat.h"
#include "coConfigGroup.h"
#include "coConfigInt.h"
#include "coConfigLong.h"
#include "coConfigString.h"
#include <util/coTypes.h>

#ifndef CO_gcc3
EXPORT_TEMPLATE2(template class CONFIGEXPORT QHash<QString, covise::coConfigGroup *>)
#endif

namespace covise
{

class CONFIGEXPORT coConfig
{

public:
    coConfigEntryStringList getScopeList(const QString &section = 0,
                                         const QString &variableName = 0) const;
    coConfigEntryStringList getVariableList(const QString &section = 0) const;

    coConfigEntryString getValue(const QString &variable,
                                 const QString &section,
                                 const QString &defaultValue) const;
    coConfigEntryString getValue(const QString &variable,
                                 const QString &section) const;
    coConfigEntryString getValue(const QString &simpleVariable) const;

    const char *getEntry(const char *simpleVariable) const;

    coConfigFloat getFloat(const QString &variable,
                           const QString &section,
                           float defaultValue) const;
    coConfigFloat getFloat(const QString &simpleVariable,
                           float defaultValue) const;
    coConfigFloat getFloat(const QString &variable,
                           const QString &section) const;
    coConfigFloat getFloat(const QString &simpleVariable) const;

    coConfigInt getInt(const QString &variable,
                       const QString &section,
                       int defaultValue) const;
    coConfigInt getInt(const QString &simpleVariable,
                       int defaultValue) const;
    coConfigInt getInt(const QString &variable,
                       const QString &section) const;
    coConfigInt getInt(const QString &simpleVariable) const;

    coConfigLong getLong(const QString &variable,
                         const QString &section,
                         long defaultValue) const;
    coConfigLong getLong(const QString &simpleVariable,
                         long defaultValue) const;
    coConfigLong getLong(const QString &variable,
                         const QString &section) const;
    coConfigLong getLong(const QString &simpleVariable) const;

    coConfigBool getBool(const QString &variable,
                         const QString &section,
                         bool defaultValue) const;
    coConfigBool getBool(const QString &simpleVariable,
                         bool defaultValue) const;
    coConfigBool getBool(const QString &variable,
                         const QString &section) const;
    coConfigBool getBool(const QString &variable,
                         const char *section) const;
    coConfigBool getBool(const QString &simpleVariable) const;

    coConfigString getString(const QString &variable,
                             const QString &section,
                             const QString &defaultValue) const;
    coConfigString getString(const QString &simpleVariable) const;

    bool isOn(const QString &variable, const QString &section, bool defaultValue) const;
    bool isOn(const QString &variable, const QString &section) const;
    bool isOn(const QString &variable, const char *section) const;
    bool isOn(const QString &simpleVariable, bool defaultValue) const;
    bool isOn(const QString &simpleVariable) const;

    void setValueForHost(const QString &variable, const QString &value,
                         const QString &section,
                         const QString &targetHost, bool move = false,
                         const QString &config = 0, const QString &configGroup = 0);

    void setValueInConfig(const QString &variable, const QString &value,
                          const QString &section,
                          const QString &configGroup,
                          const QString &config = "config",
                          bool move = false);

    void setValue(const QString &variable, const QString &value,
                  const QString &section,
                  const QString &config = 0,
                  bool move = false);

    void setValue(const QString &simpleVariable, const QString &value);

    bool deleteValueForHost(const QString &variable, const QString &section,
                            const QString &targetHost,
                            const QString &config = 0, const QString &configGroup = 0);

    bool deleteValueInConfig(const QString &variable, const QString &section,
                             const QString &configGroup, const QString &config = "config");

    bool deleteValue(const QString &variable, const QString &section, const QString &config = 0);

    bool deleteValue(const QString &simpleVariable);

    bool deleteSectionForHost(const QString &section, const QString &targetHost,
                              const QString &config = 0, const QString &configGroup = 0);

    bool deleteSectionInConfig(const QString &section, const QString &configGroup, const QString &config = "config");

    bool deleteSection(const QString &section, const QString &config = 0);

    const QStringList &getHostnameList() const;
    const QString &getActiveHost() const;
    bool setActiveHost(const QString &host);
    const QString &getActiveCluster() const;
    bool setActiveCluster(const QString &master);

    virtual void addConfig(const QString &filename, const QString &name, bool create = false);
    virtual void addConfig(coConfigGroup *group);
    virtual void removeConfig(const QString &name);

    //void destroyOwnInstance();
    void reload();

    bool save() const;
    bool save(const QString &filename) const;

    void setAdminMode(bool mode);
    bool isAdminMode();

    enum DebugLevel
    {
        DebugOff = 0x00,
        DebugGetSets = 0x01,
        DebugAll = 0xFF
    };

    static bool isDebug();
    static DebugLevel getDebugLevel();
    static void setDebugLevel(DebugLevel level)
    {
        debugLevel = level;
    }

protected:
    coConfig();
    ~coConfig();

    void load();

public: /*static*/
    static coConfig *getInstance()
    {
        if (config)
        {
            return config;
        }
        config = new coConfig();
        config->isGlobalConfig = true;
        return config;
    }

    //static coConfig *getOwnInstance(const QString & filename = 0);

protected:
    bool isGlobalConfig;

private:
    static coConfig *config;
    QString activeHostname;
    QStringList hostnames;
    QString activeCluster;
    QStringList masternames;

    QHash<QString, coConfigGroup *> configGroups;

    bool adminMode;
    static DebugLevel debugLevel;
};
}
#include "coConfigValue.inl"
#endif
