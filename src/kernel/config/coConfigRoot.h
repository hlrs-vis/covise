/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGROOT_H
#define COCONFIGROOT_H

#include <string>
#include <map>
#include <set>

#include <config/coConfigEntry.h>
#include <config/coConfigEntryString.h>
#include "coConfigFile.h"
#include <util/coTypes.h>
#include <boost/filesystem/path.hpp>
#ifndef CO_gcc3
EXPORT_TEMPLATE2(template class CONFIGEXPORT std::map<std::string, covise::coConfigEntry *>)
#endif

namespace covise
{

class coConfigGroup;

class CONFIGEXPORT coConfigRoot
{
    friend class coConfigXercesRoot;

public:
    static coConfigRoot *createNew(const std::string &name, const std::string &filename, bool create = false, coConfigGroup *group = NULL);

    coConfigRoot(const std::string &name, const std::string &filename, bool create = false, coConfigGroup *group = NULL);
    virtual ~coConfigRoot();

    virtual coConfigEntryStringList getScopeList(const std::string &section = "",
                                                 const std::string &variableName = "") const;
    virtual coConfigEntryStringList getVariableList(const std::string &section = "") const;

    virtual coConfigEntryString getValue(const std::string &variable,
                                         const std::string &section,
                                         const std::string &defaultValue) const;
    virtual coConfigEntryString getValue(const std::string &variable,
                                         const std::string &section) const;
    virtual coConfigEntryString getValue(const std::string &simpleVariable) const;

    virtual const char *getEntry(const char *simpleVariable) const;

    virtual bool isOn(const std::string &variable, const std::string &section, bool defaultValue = false) const;
    virtual bool isOn(const std::string &simpleVariable, bool defaultValue) const;
    virtual bool isOn(const std::string &simpleVariable) const;

    virtual void setValue(const std::string &variable, const std::string &value,
                          const std::string &section,
                          const std::string &targetHost = "", bool move = false);

    virtual bool deleteValue(const std::string &variable, const std::string &section,
                             const std::string &targetHost = "");

    virtual bool deleteSection(const std::string &section, const std::string &targetHost = "");

    virtual const std::set<std::string> &getHostnameList() const;
    virtual const std::string &getActiveHost() const;
    virtual bool setActiveHost(const std::string &host);
    virtual const std::set<std::string> &getClusterList() const;
    virtual const std::string &getActiveCluster() const;
    virtual bool setActiveCluster(const std::string &master);

    virtual const std::string &getConfigName() const;

    virtual void reload();
    virtual bool save(const std::string &filename = std::string()) const;

    void setGroup(coConfigGroup *group);

    void setReadOnly(bool ro);
    bool isReadOnly() const;

    std::set<std::string> getHosts();
    coConfigEntry *getConfigForHost(const std::string &hostname);
    coConfigEntry *getConfigForCluster(const std::string &masterhost);
    coConfigEntry *getGlobalConfig()
    {
        return this->globalConfig;
    }

    virtual coConfigRoot *clone() const = 0;
    virtual void merge(const coConfigRoot *with) = 0;

protected:
    virtual void load(bool create = false) = 0;

    void init();

    std::string findConfigFile(const std::string &filename, bool preferGlobal = false);

    boost::filesystem::path findLocalConfig(const std::string &filename);
    boost::filesystem::path findGlobalConfig(const std::string &filename);

    void clear();

    coConfigGroup *group;

    std::string filename;
    std::string activeHostname;
    std::set<std::string> hostnames;
    std::string activeCluster;
    std::set<std::string> masternames;

    std::string configName;

    coConfigEntry *globalConfig;
    coConfigEntry *clusterConfig;
    coConfigEntry *hostConfig;

    std::map<std::string, coConfigEntry *> hostConfigs;
    std::map<std::string, coConfigEntry *> clusterConfigs;

    virtual void createGlobalConfig() = 0;
    virtual void createHostConfig(const std::string &hostname) = 0;
    virtual void createClusterConfig(const std::string &mastername) = 0;

    bool create;
    bool readOnly;

    std::set<std::string> included;
};
}
#endif
