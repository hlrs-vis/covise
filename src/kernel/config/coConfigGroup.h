/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGGROUP_H
#define COCONFIGGROUP_H

#include <config/coConfigEntryString.h>
#include <config/coConfigEntry.h>
#include <config/coConfigRoot.h>
#include <util/coTypes.h>

namespace covise
{

class CONFIGEXPORT coConfigGroup
{

public:
    coConfigGroup(const std::string &groupName);
    virtual ~coConfigGroup();

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
    virtual bool isOn(const std::string &simpleVariable, bool defaultValue = false) const;

    virtual void setValue(const std::string &variable, const std::string &value,
                          const std::string &section,
                          const std::string &configuration = std::string(),
                          const std::string &targetHost = std::string(), bool move = false);

    virtual void setValue(const std::string &simpleVariable, const std::string &value);

    virtual bool deleteValue(const std::string &variable, const std::string &section,
                             const std::string &configuration = std::string(),
                             const std::string &targetHost = std::string());

    virtual bool deleteSection(const std::string &section,
                               const std::string &configuration = std::string(),
                               const std::string &targetHost = std::string());

    virtual std::set<std::string> getHostnameList() /*const*/;
    virtual std::string getActiveHost() const;
    virtual bool setActiveHost(const std::string &host);

    virtual std::set<std::string> getClusterList() /*const*/;
    virtual std::string getActiveCluster() const;
    virtual bool setActiveCluster(const std::string &master);

    virtual const std::string &getGroupName() const;

    virtual void reload();
    // virtual bool save(const std::string & filename, ConfigScope scope = Global) const;

    virtual coConfigRoot *addConfig(const std::string &filename, const std::string &name, bool create = false);
    virtual void removeConfig(const std::string &name);

    virtual bool save(const std::string &filename = std::string()) const;

    void setReadOnly(const std::string &config, bool ro);
    void setReadOnly(bool ro);

    bool isReadOnly(const std::string &config) const;
    bool isReadOnly() const;

    virtual coConfigGroup *clone() const;
    virtual void merge(const coConfigGroup *with);
    void flatten();

private:
    coConfigGroup(const coConfigGroup *source);

    std::string activeHostname;
    std::set<std::string> hostnames;
    std::string activeCluster;
    std::set<std::string> masternames;
    std::string groupName;

    bool readOnly;
    // friend  QHash<std::string, coConfigEntry*> mainWindow::loadFile(const std::string & fileName);
    std::map<std::string, coConfigRoot *> configs;
};
}
#endif
