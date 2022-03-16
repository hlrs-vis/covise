/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigGroup.h>
#include <config/coConfigLog.h>
#include <config/coConfigConstants.h>
#include "coConfigXercesRoot.h"
#include <util/string_util.h>
#include <regex>
#include <iostream>

using namespace std;
using namespace covise;

coConfigGroup::coConfigGroup(const std::string &groupName)
{

    this->groupName = groupName;
    activeHostname = coConfigConstants::getHostname();
    readOnly = false;
}

coConfigGroup::~coConfigGroup()
{
    for (auto &c : configs)
        delete c.second;
}

coConfigGroup::coConfigGroup(const coConfigGroup *source)
    : activeHostname(source->activeHostname), hostnames(source->hostnames), masternames(source->masternames), activeCluster(source->activeCluster), groupName(source->groupName), readOnly(source->readOnly)
{
    for (auto &entry : source->configs)
    {
        coConfigRoot *root = entry.second->clone();
        root->setGroup(this);
        this->configs.insert({entry.first, root});
    }
}

coConfigRoot *coConfigGroup::addConfig(const std::string &filename, const std::string &name, bool create)
{

    if (configs.find(name) != configs.end())
    {
        COCONFIGLOG("coConfigGroup::addConfig err: tried to add config " << filename << " twice");
        return configs[name];
    }

    coConfigRoot *configRoot = new coConfigXercesRoot(name, filename, create, this);
    configs.insert({name, configRoot});
    return configRoot;
}

void coConfigGroup::removeConfig(const std::string &name)
{
    auto config = configs.find(name);
    if (config != configs.end())
    {
        delete config->second;
        configs.erase(config);
    }
}

std::set<std::string> coConfigGroup::getHostnameList() /*const*/
{
    hostnames.clear();
    for (const auto &config : configs)
    {
        hostnames.insert(config.second->getHostnameList().begin(), config.second->getHostnameList().end());
    }
    return hostnames;
}

std::string coConfigGroup::getActiveHost() const
{
    return activeHostname;
}

bool coConfigGroup::setActiveHost(const std::string &host)
{

    activeHostname = host;
    for (const auto &config : configs)
        config.second->setActiveHost(host);

    return true;
}

std::set<std::string> coConfigGroup::getClusterList() /*const*/
{
    masternames.clear();
    for (const auto &config : configs)
    {
        masternames.insert(config.second->getClusterList().begin(), config.second->getClusterList().end());
    }
    return masternames;
}

std::string coConfigGroup::getActiveCluster() const
{
    return activeCluster;
}

bool coConfigGroup::setActiveCluster(const std::string &master)
{
    activeCluster = master;
    for (const auto &config : configs)
        config.second->setActiveCluster(master);
    return true;
}

void coConfigGroup::reload()
{
    COCONFIGDBG("coConfigGroup::reload info: reloading config");
    for (const auto &config : configs)
        config.second->reload();
}

coConfigEntryStringList coConfigGroup::getScopeList(const std::string &section,
                                                    const std::string &variableName) const
{

    coConfigEntryStringList merged;

    for (const auto &config : configs)
    {
        coConfigEntryStringList list = config.second->getScopeList(section, variableName);
        merged.merge(list);
    }
    if (variableName.empty())
    {
        return merged;
    }
    else
    {

        return merged.filter(std::regex("^" + variableName + ":.*"));
    }
}

coConfigEntryStringList coConfigGroup::getVariableList(const std::string &section) const
{

    coConfigEntryStringList merged;

    for (const auto config : configs)
    {
        coConfigEntryStringList list = config.second->getVariableList(section);
        merged.merge(list);
    }
    return merged;
}

coConfigEntryString coConfigGroup::getValue(const std::string &variable,
                                            const std::string &section,
                                            const std::string &defaultValue) const
{

    coConfigEntryString value = getValue(variable, section);

    if (value == coConfigEntryString{})
    {
        return coConfigEntryString{defaultValue};
    }
    value.configGroupName = groupName;
    return value;
}

coConfigEntryString coConfigGroup::getValue(const std::string &simpleVariable) const
{
    return getValue("value", simpleVariable);
}

coConfigEntryString coConfigGroup::getValue(const std::string &variable,
                                            const std::string &section) const
{

    coConfigEntryString item;

    for (const auto config : configs)
    {
        coConfigEntryString currentValue = config.second->getValue(variable, section);
        if (!(currentValue == coConfigEntryString{}))             {
            item = currentValue;
            item.configGroupName = groupName;

        }
    }
    COCONFIGDBG_GET_SET("coConfigGroup::getValue info: [" << groupName << "] "
                                                          << section << "." << variable << " = "
                                                          << (item == coConfigEntryString{} ? "*NO VALUE*" : item.entry));

    return item;
}

const char *coConfigGroup::getEntry(const char *variable) const
{
    const char *item = NULL;
    for (auto &config : configs)
    {
        auto currentValue = config.second->getEntry(variable);
        if (currentValue)
            item = currentValue;
    }
    return item;
}

bool coConfigGroup::isOn(const std::string &simpleVariable, bool defaultValue) const
{
    return isOn("value", simpleVariable, defaultValue);
}

bool coConfigGroup::isOn(const std::string &variable, const std::string &section,
                         bool defaultValue) const
{

    auto value = getValue(variable, section).entry;

    if (value.empty())
        return defaultValue;

    if (toLower(value) == "on" || toLower(value) == "true" || atoi(value.c_str()) > 0)
        return true;
    else
        return false;
}

void coConfigGroup::setValue(const std::string &simpleVariable, const std::string &value)
{
    setValue("value", value, simpleVariable);
}

void coConfigGroup::setValue(const std::string &variable, const std::string &value,
                             const std::string &section,
                             const std::string &configuration,
                             const std::string &targetHost, bool move)
{

    if (isReadOnly())
        return;

    std::string configurationName = configuration;

    if (configuration.empty())
    {
        // cerr << "coConfigGroup::setValue info: getting " << variable << " in " << section << endl;
        coConfigEntryString oldValue = getValue(variable, section);

        if (oldValue == coConfigEntryString{})
        {
            assert(!configs.empty());
            configurationName = configs.begin()->first;
            // cerr << "coConfigGroup::setValue info: autoselecting config " << configurationName << endl;
        }
        else
        {
            configurationName = oldValue.configName;
            // cerr << "coConfigGroup::setValue info: autoselecting config (from old value) " << configurationName << endl;
        }
    }

    coConfigRoot *root;

    root = configs[configurationName];

    if (root == 0)
    {
        COCONFIGLOG("coConfigGroup::setValue warn: no such configuration: " << configuration);
    }
    else
    {
        root->setValue(variable, value, section, targetHost, move);
    }
}

bool coConfigGroup::deleteValue(const std::string &variable, const std::string &section,
                                const std::string &configuration, const std::string &targetHost)

{

    if (isReadOnly())
        return false;

    std::string configurationName = configuration;

    if (configuration == std::string())
    {

        coConfigEntryString oldValue = getValue(variable, section);

        if (oldValue == coConfigEntryString{})
        {
            return false;
        }
        else
        {
            configurationName = oldValue.configName;
            COCONFIGDBG("coConfigGroup::deleteValue info: autoselecting config (from old value) " << configurationName);
        }
    }

    coConfigRoot *root;

    root = configs[configurationName];

    if (root == 0)
    {
        COCONFIGLOG("coConfigGroup::deleteValue warn: no such configuration: " << configuration);
        return false;
    }
    else
    {
        return root->deleteValue(variable, section, targetHost);
    }
}

bool coConfigGroup::deleteSection(const std::string &section, const std::string &configuration, const std::string &targetHost)

{

    if (isReadOnly())
        return false;

    bool removed = false;
    std::string configurationName = configuration;

    if (configuration == std::string())
    {
        for(const auto& config : configs)
        {
            coConfigRoot *root = config.second;
            if (root)
                removed |= root->deleteSection(section, targetHost);
        }
    }

    return removed;
}

bool coConfigGroup::save(const std::string &filename) const
{

    if (isReadOnly())
    {
        COCONFIGDBG("coConfigGroup::save warn: not saving read only config");
        return true;
    }

    if (filename == std::string() && configs.size() > 1)
    {
        COCONFIGDBG("coConfigGroup::save warn: saving more than one config group, only the last one will be saved, consider flattening the configuration");
    }

    bool saved = true;

    for(const auto& config : configs)
    {
        COCONFIGDBG("coConfigGroup::save info: saving root " << config.second->getConfigName());
        saved &= config.second->save(filename);
    }

    return saved;
}

const std::string &coConfigGroup::getGroupName() const
{
    return groupName;
}

void coConfigGroup::setReadOnly(bool ro)
{
    readOnly = ro;

    for(const auto& config : configs)
    {
        config.second->setReadOnly(ro);
    }
}

bool coConfigGroup::isReadOnly() const
{
    return readOnly;
}

void coConfigGroup::setReadOnly(const std::string &config, bool ro)
{
    coConfigRoot *root = configs[config];

    if (root == 0)
    {
        COCONFIGLOG("coConfigGroup::setReadOnly warn: no such configuration: " << config);
    }
    else
    {
        root->setReadOnly(ro);
    }
}

bool coConfigGroup::isReadOnly(const std::string &config) const
{
    coConfigRoot *root = nullptr;
    auto c = configs.find(config);
    if (c != configs.end())
        root = c->second;
    if (!root)
    {
        COCONFIGLOG("coConfigGroup::isReadOnly warn: no such configuration: " << config);
        return false;
    }
    else
        return root->isReadOnly();
}

coConfigGroup *coConfigGroup::clone() const
{
    return new coConfigGroup(this);
}

void coConfigGroup::merge(const coConfigGroup *with)
{
    for(const auto &withConfig : with->configs)
    {
        auto config = configs.find(withConfig.first);
        if(config != configs.end())
            config->second->merge(withConfig.second);
        else
        {
            auto root = withConfig.second->clone();
            root->setGroup(this);
        }
    }
}

/**
 * Flattens the configuration structure that only one tree remains.
 * The name of the configuration is taken from the first configuration.
 */

void coConfigGroup::flatten()
{
    if (configs.size() < 2)
        return;

    auto entry = configs.begin();

    std::string mainKey = entry->first;
    coConfigRoot *root = entry->second;

    for (++entry; entry != configs.end(); ++entry)
    {
        root->merge(entry->second);
        delete entry->second;
    }

    configs.clear();
    configs.insert({mainKey, root});
}
