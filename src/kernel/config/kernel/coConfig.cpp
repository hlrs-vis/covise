/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigLog.h>
#include <config/coConfig.h>

#include <math.h>

#include <xercesc/dom/DOM.hpp>
#include <util/string_util.h>
#include <boost/filesystem/operations.hpp>
using namespace std;
using namespace covise;

coConfig *coConfig::config = 0;
coConfig::DebugLevel coConfig::debugLevel = coConfig::DebugOff;

coConfig::coConfig()
{

    isGlobalConfig = false;
    adminMode = false;

    auto dm = getenv("COCONFIG_DEBUG");
    if (dm)
    {
        unsigned int dl = atoi(dm);
        switch (dl)
        {
        case DebugOff:
            debugLevel = DebugOff;
            break;
        case DebugGetSets:
            debugLevel = DebugGetSets;
            break;
        default:
            debugLevel = DebugAll;
        }
    }
    else
    {
        debugLevel = DebugOff;
    }

    COCONFIGDBG_DEFAULT("coConfig::<init> info: debug level is " << debugLevel);

    COCONFIGDBG("coConfigConstants::<init> info: hostname is " << coConfigConstants::getHostname());
    activeHostname = coConfigConstants::getHostname();

    load();
}

coConfig::~coConfig()
{
    for (auto &group : configGroups)
        delete group.second;
}

const std::set<std::string> &coConfig::getHostnameList() const
{
    return hostnames;
}

const std::string &coConfig::getActiveHost() const
{
    return activeHostname;
}

bool coConfig::setActiveHost(const std::string &host)
{

    if (hostnames.find(toLower(host)) != hostnames.end())
    {
        //cerr << "coConfig::setActiveHost info: setting active host "
        //     << host << endl;
        activeHostname = toLower(host);

        for (auto &group : configGroups)
            group.second->setActiveHost(activeHostname);

        return true;
    }
    else
    {
        COCONFIGLOG("coConfig::setActiveHost warn: could not set active host " << toLower(host));
        return false;
    }
}

const std::string &coConfig::getActiveCluster() const
{
    return activeCluster;
}

bool coConfig::setActiveCluster(const std::string &master)
{
    if (masternames.find(toLower(master)) != masternames.end() || master.empty())
    {
        //cerr << "coConfig::setActiveCluster info: setting active cluster "
        //     << host << endl;
        activeCluster = toLower(master);

        for (auto &group : configGroups)
            group.second->setActiveCluster(activeCluster);

        return true;
    }
    else
    {
        COCONFIGDBG("coConfig::setActiveCluster warn: could not set active cluster " << toLower(master));
        return false;
    }
}

void coConfig::reload()
{

    COCONFIGDBG("coConfig::reload info: reloading config");

    for (auto &group : configGroups)
        group.second->reload();
}

void coConfig::load()
{

    std::string configGlobal = coConfigDefaultPaths::getDefaultGlobalConfigFileName();
    std::string configLocal = coConfigDefaultPaths::getDefaultLocalConfigFileName();

    if (!configGlobal.empty())
        COCONFIGDBG("coConfig::load info: global config file in " << configGlobal);
    if (!configLocal.empty())
        COCONFIGDBG("coConfig::load info: local  config file in " << configLocal);

    coConfigGroup *mainGroup = new coConfigGroup("config");

    // Load global configuration
    mainGroup->addConfig(configGlobal, "global", false);
    mainGroup->setReadOnly("global", true);

    // Load user configuration
    mainGroup->addConfig(configLocal, "local", true);

    configGroups["config"] = mainGroup;

    // Set active host

    this->hostnames = mainGroup->getHostnameList();
    this->masternames = mainGroup->getClusterList();

    setActiveCluster(activeCluster);
    setActiveHost(activeHostname);
}

coConfigEntryStringList coConfig::getScopeList(const std::string &section,
                                               const std::string &variableName) const
{

    coConfigEntryStringList merged;

    for (const auto &configGroup : configGroups)
    {
        coConfigEntryStringList list = configGroup.second->getScopeList(section, variableName);
        merged.merge(list);
    }
    if (variableName.empty())
    {
        return merged;
    }
    else
    {
        // FIXME Do I have to?
        return merged.filter(std::regex("^" + variableName + ":.*"));
    }
}

coConfigEntryStringList coConfig::getVariableList(const std::string &section) const
{

    coConfigEntryStringList merged;

    for (const auto &configGroup : configGroups)
    {
        coConfigEntryStringList list = configGroup.second->getVariableList(section);
        merged = merged.merge(list);
    }
    return merged;
}

coConfigEntryString coConfig::getValue(const std::string &variable,
                                       const std::string &section,
                                       const std::string &defaultValue) const
{

    coConfigEntryString value = getValue(variable, section);
    if (value == coConfigEntryString{})
        return coConfigEntryString{defaultValue};
    return value;
}

coConfigEntryString coConfig::getValue(const std::string &simpleVariable) const
{
    return getValue("value", simpleVariable);
}

coConfigEntryString coConfig::getValue(const std::string &variable,
                                       const std::string &section) const
{

    coConfigEntryString item;

    for (const auto configGroup : configGroups)
    {
        coConfigEntryString currentValue = configGroup.second->getValue(variable, section);
        if (!(currentValue == coConfigEntryString{}))
            item = currentValue;
    }

    return item;
}

const char *coConfig::getEntry(const char *simpleVariable) const
{

    const char *item = 0;

    for (const auto configGroup : configGroups)
    {

        const char *currentValue = configGroup.second->getEntry(simpleVariable);
        if (currentValue)
            item = currentValue;
    }

    return item;
}

coConfigString coConfig::getString(const std::string &variable, const std::string &section, const std::string &defaultValue) const
{

    coConfigString value{variable, section};

    if (value.hasValidValue())
    {
        return value;
    }
    else
    {
        value = defaultValue;
        return value;
    }
}

coConfigString coConfig::getString(const std::string &simpleVariable) const
{
    //FIXME
    return getString("value", simpleVariable, "");
}

coConfigInt coConfig::getInt(const std::string &variable, const std::string &section, int defaultValue) const
{

    coConfigInt value(variable, section);

    if (value.hasValidValue())
    {
        return value;
    }
    else
    {
        value = defaultValue;
        return value;
    }
}

coConfigInt coConfig::getInt(const std::string &simpleVariable, int defaultValue) const
{
    return getInt("value", simpleVariable, defaultValue);
}

coConfigInt coConfig::getInt(const std::string &variable, const std::string &section) const
{
    return coConfigInt(variable, section);
}

coConfigInt coConfig::getInt(const std::string &simpleVariable) const
{
    return getInt("value", simpleVariable);
}

coConfigLong coConfig::getLong(const std::string &variable, const std::string &section, long defaultValue) const
{

    coConfigLong value(variable, section);

    if (value.hasValidValue())
    {
        return value;
    }
    else
    {
        value = defaultValue;
        return value;
    }
}

coConfigLong coConfig::getLong(const std::string &simpleVariable, long defaultValue) const
{
    return getLong("value", simpleVariable, defaultValue);
}

coConfigLong coConfig::getLong(const std::string &variable, const std::string &section) const
{
    return coConfigLong(variable, section);
}

coConfigLong coConfig::getLong(const std::string &simpleVariable) const
{
    return getLong("value", simpleVariable);
}

coConfigBool coConfig::getBool(const std::string &variable, const std::string &section, bool defaultValue) const
{

    coConfigBool value(variable, section);

    if (value.hasValidValue())
    {
        return value;
    }
    else
    {
        value = defaultValue;
        return value;
    }
}

coConfigBool coConfig::getBool(const std::string &simpleVariable, bool defaultValue) const
{
    return getBool("value", simpleVariable, defaultValue);
}

coConfigBool coConfig::getBool(const std::string &variable, const std::string &section) const
{
    return coConfigBool(variable, section);
}

coConfigBool coConfig::getBool(const std::string &variable, const char *section) const
{
    return coConfigBool(variable, std::string(section));
}

coConfigBool coConfig::getBool(const std::string &simpleVariable) const
{
    return getBool("value", simpleVariable);
}

/**
 * @brief Get a float value.
 *
 * If the variable is not found, defaultValue is returned instead.
 *
 * @param variable Variable to check.
 * @param section Section of the variable.
 * @param defaultValue Default value to return.
 * @return The value of the variable as float or the default value if the
 *         variable was not found.
 */

coConfigFloat coConfig::getFloat(const std::string &variable,
                                 const std::string &section,
                                 float defaultValue) const
{

    coConfigFloat value(variable, section);
    //cerr << "coConfig::getFloat info: " << section << "." << variable << " = "
    //     << (value.hasValidValue() ? FP_NAN : value) << endl;

    if (value.hasValidValue())
    {
        //cerr << "coConfig::getFloat info: found valid value" << endl;
        return value;
    }
    else
    {
        //cerr << "coConfig::getFloat warn: using default value" << endl;
        value = defaultValue;
        return value;
    }
}

/**
 * @brief Get a float value.
 *
 * If the variable is not found, defaultValue is returned instead.
 * Shortcut for getFloat(&quot;value&quot;, section, defaultValue)
 *
 * @param simpleVariable Section of the variable &quot;value&quot;.
 * @param defaultValue Default value to return.
 * @return The value of the variable as float or the default value if the
 *         variable was not found.
 */
coConfigFloat coConfig::getFloat(const std::string &simpleVariable, float defaultValue) const
{
    return getFloat("value", simpleVariable, defaultValue);
}

/**
 * @brief Get a float value.
 *
 * @param variable Variable to get.
 * @param section Section of the variable.
 * @return The value of the variable as float.
 */
coConfigFloat coConfig::getFloat(const std::string &variable,
                                 const std::string &section) const
{
    return coConfigFloat(variable, section);
}

/**
 * @brief Get a float value.
 *
 * Shortcut for getFloat(&quot;value&quot;, section)
 * @param simpleVariable Section of the variable &quot;value&quot;.
 * @return The value of the variable as float.
 */
coConfigFloat coConfig::getFloat(const std::string &simpleVariable) const
{
    return getFloat("value", simpleVariable);
}

/**
 * @brief Checks if a variable in the configuration is set to &quot;on&quot;.
 *
 * As &quot;on&quot; count the case insensitive values of <I>on</I>, <I>true</I> and <I>1</I>.
 * It is a shortcut for isOn(&quot;value&quot;, section, defaultValue)
 * @param simpleVariable Section of the variable &quot;value&quot;.
 * @param defaultValue Default value to return.
 * @return <I>true</I>, if the variable has above values,
 *         <I>false</I> if not,
 *         <I>defaultValue</I> if the variable does not exist.
 */
bool coConfig::isOn(const std::string &simpleVariable, bool defaultValue) const
{
    return isOn("value", simpleVariable, defaultValue);
}

/**
 * @brief Checks if a variable in the configuration is set to &quot;on&quot;.
 *
 * As &quot;on&quot; count the case insensitive values of <I>on</I>, <I>true</I> and <I>1</I>.
 * It is a shortcut for isOn(&quot;value&quot;, section)
 * @param simpleVariable Section of the variable &quot;value&quot;.
 * @return <I>true</I>, if the variable has above values,
 *         <I>false</I> if not or the variable does not exist.
 */
bool coConfig::isOn(const std::string &simpleVariable) const
{
    return isOn("value", simpleVariable);
}

/**
 * @brief Checks if a variable in the configuration is set to &quot;on&quot;.
 *
 * As &quot;on&quot; count the case insensitive values of <I>on</I>, <I>true</I> and <I>1</I>.
 * @param variable Variable to check.
 * @param section Section of the variable.
 * @param defaultValue Default value to return.
 * @return <I>true</I>, if the variable has above values,
 *         <I>false</I> if not,
 *         <I>defaultValue</I> if the variable does not exist.
 */
bool coConfig::isOn(const std::string &variable, const std::string &section,
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

/**
 * @brief Set a value for a variable for a host.
 *
 * @param targetHost Host to set the value for.
 * @param configGroup Configuration group to set the value in.
 *
 * For other parameters, see
 * coConfig::setValue(const std::string & variable, const std::string & value, const std::string & section, const std::string & config, bool move).
 *
 */
void coConfig::setValueForHost(const std::string &variable, const std::string &value, const std::string &section,
                               const std::string &targetHost, bool move,
                               const std::string &config, const std::string &configGroup)
{

    coConfigEntryString oldValue = getValue(variable, section);

    coConfigGroup *group;
    std::string groupName;
    std::string groupConfigName;

    if (config.empty())
    {
        //if (oldValue.isNull() || oldValue.getConfigGroupName() == "global")
        if (oldValue == coConfigEntryString{} || oldValue.configName == "global")
        {
            groupName = "config";
            groupConfigName = "local";
        }
        else
        {
            groupName = oldValue.configGroupName;
            groupConfigName = oldValue.configName;
        }
    }
    else
    {
        groupName = config;
    }

    if (!configGroup.empty())
        groupConfigName = configGroup;

    group = configGroups[groupName];

    if (group == 0)
    {
        COCONFIGLOG("coConfig::setValue warn: unknown config group " << groupName);
        return;
    }

    //    cerr << "coConfig::setValue info: " << section << "." << variable
    //       << " = " << value << " in config '" << groupName << "'"
    //       << (targetHost ? " on host " + targetHost : std::string("")) << endl;

    group->setValue(variable, value, section, groupConfigName, targetHost, move);

    oldValue = getValue(variable, section);
    //    cerr << "coConfig::setValue info: vrfy " << section << " - " << variable
    //       << " = " << oldValue << " in config '" << oldValue.getConfigGroupName() << "'"
    //       << endl;
}

/**
 * @brief Set a value for a variable in the designated config group.
 *
 * @param configGroup Configuration group to set the value in.
 *
 * For other parameters, see
 * coConfig::setValue(const std::string & variable, const std::string & value, const std::string & section, const std::string & config, bool move).
 */
void coConfig::setValueInConfig(const std::string &variable, const std::string &value, const std::string &section,
                                const std::string &configGroup, const std::string &config, bool move)
{

    setValueForHost(variable, value, section, 0, move, config, configGroup);
}

/**
 * @brief Set a value for a variable.
 *
 * The actual configuration used can be provided as parameter or
 * automatically determined; in the latter case, this is the configuration the
 * entry already resides in or <I>config.local</I> if it resides in the global configuration
 * or not at all.
 *
 * @param variable Variable to set.
 * @param value Value to set to.
 * @param section Section of the variable to set.
 * @param config Configuration to set the variable in. For defaults see above
 * @param move Delete the value from its old location <B>*not implemented yet*</B>
 */
void coConfig::setValue(const std::string &variable, const std::string &value, const std::string &section,
                        const std::string &config, bool move)
{

    setValueForHost(variable, value, section, "", move, config);
}

/**
 * @brief Set a value for a simple variable in the default configuration.
 *
 * The actual configuration used is automatically determined and is the configuration the
 * entry already resides in or <I>config.local</I> if it resides in the global configuration
 * or not at all.
 *
 * @param simpleVariable Section of the variable to set. The variable itself is called &quot;value&quot;
 * @param value Value to set to.
 */
void coConfig::setValue(const std::string &simpleVariable, const std::string &value)
{
    setValue("value", value, simpleVariable);
}

//---- Delete ----

/**
 * @brief Delete a value in the designated configuration group for a host only
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param variable Variable to delete.
 * @param section Section, where the variable to delete is found.
 * @param targetHost The hostname of the host to be affected.
 * @param config Configuration where to delete from.
 * @param configGroup Name of the configuration group to delete from.
 *
 * @return true, if something was deleted.
 */
bool coConfig::deleteValueForHost(const std::string &variable, const std::string &section,
                                  const std::string &targetHost,
                                  const std::string &config, const std::string &configGroup)
{

    coConfigGroup *group;
    std::string groupName;
    std::string groupConfigName;

    if (config.empty())
    {
        groupName = "config";
        groupConfigName = "local";
    }
    else
    {
        groupName = config;
        groupConfigName = "local";
    }

    if (!configGroup.empty())
        groupConfigName = configGroup;

    group = configGroups[groupName];

    if (group == 0)
    {
        COCONFIGLOG("coConfig::deleteValueForHost warn: unknown config group " << groupName);
        return false;
    }

    return group->deleteValue(variable, section, groupConfigName, targetHost);
}

/**
 * @brief Delete a value in the designated configuration group
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param variable Variable to delete.
 * @param section Section, where the variable to delete is found.
 * @param configGroup Name of the configuration group to delete from.
 * @param config Configuration where to delete from.
 *
 * @return true, if something was deleted.
 */
bool coConfig::deleteValueInConfig(const std::string &variable, const std::string &section,
                                   const std::string &configGroup, const std::string &config)
{

    return deleteValueForHost(variable, section, "", config, configGroup);
}

/**
 * @brief Delete a value from the configuration
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param variable Variable to delete.
 * @param section Section, where the variable to delete is found.
 * @param config Configuration where to delete from.
 *
 * @return true, if something was deleted.
 */
bool coConfig::deleteValue(const std::string &variable, const std::string &section, const std::string &config)
{
    return deleteValueForHost(variable, section, "", config);
}

/**
 * @brief Delete a value from the configuration
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param simpleVariable Variable to delete. This method deletes the variable &quot;value&quot;
 * @return true, if something was deleted.
 */
bool coConfig::deleteValue(const std::string &simpleVariable)
{
    return deleteValue("value", simpleVariable);
}

/**
 * @brief Delete a section for a host in the configuration.
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param section Section to delete.
 * @param targetHost The hostname of the host to be affected.
 * @param configGroup Name of the configuration group to delete from.
 * @param config Name of the configuration to delete this section from.
 * @return true, if something was deleted.
 */
bool coConfig::deleteSectionForHost(const std::string &section, const std::string &targetHost,
                                    const std::string &config, const std::string &configGroup)
{

    coConfigGroup *group;
    std::string groupName;
    std::string groupConfigName;

    if (config.empty())
    {
        groupName = "config";
        groupConfigName = "local";
    }
    else
    {
        groupName = config;
        groupConfigName = "local";
    }

    if (!configGroup.empty())
        groupConfigName = configGroup;

    group = configGroups[groupName];

    if (!group)
    {
        COCONFIGLOG("coConfig::deleteValueForHost warn: unknown config group " << groupName);
        return false;
    }

    return group->deleteSection(section, groupConfigName, targetHost);
}

/**
 * @brief Delete a section in the configuration.
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param section Section to delete.
 * @param configGroup Name of the configuration group to delete from.
 * @param config Name of the configuration to delete this section from.
 * @return true, if something was deleted.
 */
bool coConfig::deleteSectionInConfig(const std::string &section, const std::string &configGroup, const std::string &config)
{

    return deleteSectionForHost(section, "", config, configGroup);
}

/**
 * @brief Delete a section in the configuration.
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param section Section to delete.
 * @param config Name of the configuration to delete this section from.
 * @return true, if something was deleted.
 */
bool coConfig::deleteSection(const std::string &section, const std::string &config)
{
    return deleteSectionForHost(section, "", config);
}

/**
 * @brief Save all configurations to disk.
 *
 * Before calling this method, all changes are just in memory.
 *
 * @note You have to turn on the administrative mode if you want to save the global
 * configuration.
 *
 * @return true, if all files were saved successfully.
 */
bool coConfig::save() const
{
    bool saved = true;
    for (const auto &configGroup : configGroups)
    {
        COCONFIGDBG("coConfig::save info: saving group " << configGroup.second->getGroupName());
        saved &= configGroup.second->save();
    }
    return saved;
}

/**
 * @brief Exports all configurations into a single file.
 *
 * @return true, if the configuration was exported successfully.
 */
bool coConfig::save(const std::string &filename) const
{

    bool saved = true;

    auto group = configGroups.begin();

    coConfigGroup *merged = group->second->clone();

    for (++group; group != configGroups.end(); ++group)
    {
        merged->merge(group->second);
    }
    merged->flatten();
    merged->setReadOnly(false);
    merged->save(filename);

    delete merged;

    return saved;
}

/**
 * @brief Turn administrator mode on or off.
 *
 * It is only possible to write to the global configuration after turning this on.
 *
 * @note You can only turn on the administrative mode if you have write access to the global
 * configuration files.
 *
 * @param mode New administrative mode.
 */
void coConfig::setAdminMode(bool mode)
{
    boost::filesystem::perms prms;
    boost::filesystem::permissions(coConfigDefaultPaths::getDefaultGlobalConfigFileName(), prms);
    if ((prms & boost::filesystem::perms::group_write) != boost::filesystem::perms::no_perms)
    {
        adminMode = mode;
        if (adminMode)
            configGroups["config"]->setReadOnly("global", !mode);
    }
    else
    {
        adminMode = false;
        configGroups["config"]->setReadOnly("global", true);
    }
}

/**
 * @brief Is the administrator mode on or off.
 *
 * If the administrative mode is on, the global configuration can be written to.
 * @return true, if the administrative mode is on.
 */
bool coConfig::isAdminMode()
{
    return adminMode;
}

/**
 * @brief Load a configuration from a file and add it to the global configuration.
 *
 * @param filename The file to load. The name is searched for in the local/global config dir and COVISE_PATH
 * @param name Name of the new configuration. Has to be unique.
 * @param create If the file does not exist, create a new, empty configuration.
 */
void coConfig::addConfig(const std::string &filename, const std::string &name, bool create)
{
    configGroups["config"]->addConfig(filename, name, create);
}

/**
 * @brief Add a configuration group for access through the global config.
 *
 * @param group The group to add. The name of the group has to be unique.
 */
void coConfig::addConfig(coConfigGroup *group)
{
    configGroups.insert({group->getGroupName(), group});
    auto hostnameList = group->getHostnameList();
    auto clusterList = group->getClusterList();
    hostnames.insert(hostnameList.begin(), hostnameList.end());
    masternames.insert(clusterList.begin(), clusterList.end());
}

/**
 * @brief Removes a configuration group from the global coConfig.
 *
 * @param name Name of the configuration.
 */
void coConfig::removeConfig(const std::string &name)
{
    configGroups["config"]->removeConfig(name);

    hostnames.clear();
    masternames.clear();
    for (const auto group : configGroups)
    {
        auto hostnameList = group.second->getHostnameList();
        auto clusterList = group.second->getClusterList();
        hostnames.insert(hostnameList.begin(), hostnameList.end());
        masternames.insert(clusterList.begin(), clusterList.end());
    }
}

/**
 * @brief Check if in debug mode.
 *
 * @return true, if any debug level is set.
 *
 * @sa coConfig::getDebugLevel()
 */
bool coConfig::isDebug()
{
    return debugLevel != 0;
}

/**
 * @brief Get the debug level.
 *
 * The debug level is set via the environment variable COCONFIG_DEBUG. Allowed values are:
 * <UL>
 * <LI>0: off</LI>
 * <LI>1: debug gets/sets</LI>
 * <LI>any other: debug all</LI>
 * </UL>
 *
 * @return The debug level
 */
coConfig::DebugLevel coConfig::getDebugLevel()
{
    return debugLevel;
}
