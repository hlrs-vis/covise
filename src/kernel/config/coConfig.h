/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIG_H
#define COCONFIG_H

#include <set>
#include <map>

#include "coConfigBool.h"
#include "coConfigConstants.h"
#include "coConfigFloat.h"
#include "coConfigGroup.h"
#include "coConfigInt.h"
#include "coConfigLong.h"
#include "coConfigString.h"
#include <util/coTypes.h>

namespace covise
{

class CONFIGEXPORT coConfig
{

public:
    coConfigEntryStringList getScopeList(const std::string &section = "",
                                         const std::string &variableName = "") const;
    coConfigEntryStringList getVariableList(const std::string &section = "") const;

    coConfigEntryString getValue(const std::string &variable,
                                 const std::string &section,
                                 const std::string &defaultValue) const;
    coConfigEntryString getValue(const std::string &variable,
                                 const std::string &section) const;
    coConfigEntryString getValue(const std::string &simpleVariable) const;

    const char *getEntry(const char *simpleVariable) const;

    coConfigFloat getFloat(const std::string &variable,
                           const std::string &section,
                           float defaultValue) const;
    coConfigFloat getFloat(const std::string &simpleVariable,
                           float defaultValue) const;
    coConfigFloat getFloat(const std::string &variable,
                           const std::string &section) const;
    coConfigFloat getFloat(const std::string &simpleVariable) const;

    coConfigInt getInt(const std::string &variable,
                       const std::string &section,
                       int defaultValue) const;
    coConfigInt getInt(const std::string &simpleVariable,
                       int defaultValue) const;
    coConfigInt getInt(const std::string &variable,
                       const std::string &section) const;
    coConfigInt getInt(const std::string &simpleVariable) const;

    coConfigLong getLong(const std::string &variable,
                         const std::string &section,
                         long defaultValue) const;
    coConfigLong getLong(const std::string &simpleVariable,
                         long defaultValue) const;
    coConfigLong getLong(const std::string &variable,
                         const std::string &section) const;
    coConfigLong getLong(const std::string &simpleVariable) const;

    coConfigBool getBool(const std::string &variable,
                         const std::string &section,
                         bool defaultValue) const;
    coConfigBool getBool(const std::string &simpleVariable,
                         bool defaultValue) const;
    coConfigBool getBool(const std::string &variable,
                         const std::string &section) const;
    coConfigBool getBool(const std::string &variable,
                         const char *section) const;
    coConfigBool getBool(const std::string &simpleVariable) const;

    coConfigString getString(const std::string &variable,
                             const std::string &section,
                             const std::string &defaultValue) const;
    coConfigString getString(const std::string &simpleVariable) const;

    bool isOn(const std::string &variable, const std::string &section, bool defaultValue = false) const;
    bool isOn(const std::string &simpleVariable, bool defaultValue) const;
    bool isOn(const std::string &simpleVariable) const;

    void setValueForHost(const std::string &variable, const std::string &value,
                         const std::string &section,
                         const std::string &targetHost, bool move = false,
                         const std::string &config = "", const std::string &configGroup = "");

    void setValueInConfig(const std::string &variable, const std::string &value,
                          const std::string &section,
                          const std::string &configGroup,
                          const std::string &config = "config",
                          bool move = false);

    void setValue(const std::string &variable, const std::string &value,
                  const std::string &section,
                  const std::string &config = "",
                  bool move = false);

    void setValue(const std::string &simpleVariable, const std::string &value);

    bool deleteValueForHost(const std::string &variable, const std::string &section,
                            const std::string &targetHost,
                            const std::string &config = "", const std::string &configGroup = "");

    bool deleteValueInConfig(const std::string &variable, const std::string &section,
                             const std::string &configGroup, const std::string &config = "config");

    bool deleteValue(const std::string &variable, const std::string &section, const std::string &config = "");

    bool deleteValue(const std::string &simpleVariable);

    bool deleteSectionForHost(const std::string &section, const std::string &targetHost,
                              const std::string &config = "", const std::string &configGroup = "");

    bool deleteSectionInConfig(const std::string &section, const std::string &configGroup, const std::string &config = "config");

    bool deleteSection(const std::string &section, const std::string &config = "");

    const std::set<std::string> &getHostnameList() const;
    const std::string &getActiveHost() const;
    bool setActiveHost(const std::string &host);
    const std::string &getActiveCluster() const;
    bool setActiveCluster(const std::string &master);

    virtual void addConfig(const std::string &filename, const std::string &name, bool create = false);
    virtual void addConfig(coConfigGroup *group);
    virtual void removeConfig(const std::string &name);

    //void destroyOwnInstance();
    void reload();

    bool save() const;
    bool save(const std::string &filename) const;

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

    // static coConfig *getOwnInstance(const std::string & filename = "");

protected:
    bool isGlobalConfig;

private:
    static coConfig *config;
    std::string activeHostname;
    std::set<std::string> hostnames;
    std::string activeCluster;
    std::set<std::string> masternames;

    std::map<std::string, coConfigGroup *> configGroups;

    bool adminMode;
    static DebugLevel debugLevel;
};
}
#include "coConfigValue.inl"
#endif
