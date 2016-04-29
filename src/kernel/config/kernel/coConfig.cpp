/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigLog.h>
#include <config/coConfig.h>

#include <math.h>

#include <QFile>
#include <QFileInfo>
#include <QDir>

#include <xercesc/dom/DOM.hpp>
#include <QRegExp>
#include <QTextStream>

using namespace std;
using namespace covise;

coConfig *coConfig::config = 0;
coConfig::DebugLevel coConfig::debugLevel = coConfig::DebugOff;

coConfig::coConfig()
{

    isGlobalConfig = false;
    adminMode = false;

    QString debugModeEnv = getenv("COCONFIG_DEBUG");
    if (!debugModeEnv.isNull())
    {
        unsigned int dl = debugModeEnv.toUInt();
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
    QList<QString> keys = configGroups.keys();
    for (QList<QString>::iterator i = keys.begin(); i != keys.end(); ++i)
    {
        delete configGroups.take(*i);
    }
}

const QStringList &coConfig::getHostnameList() const
{
    return hostnames;
}

const QString &coConfig::getActiveHost() const
{
    return activeHostname;
}

bool coConfig::setActiveHost(const QString &host)
{

    if (hostnames.contains(host.toLower()))
    {
        //cerr << "coConfig::setActiveHost info: setting active host "
        //     << host << endl;
        activeHostname = host.toLower();

        for (QHash<QString, coConfigGroup *>::iterator i = configGroups.begin(); i != configGroups.end(); ++i)
        {
            (*i)->setActiveHost(activeHostname);
        }

        return true;
    }
    else
    {

        COCONFIGLOG("coConfig::setActiveHost warn: could not set active host " << host.toLower());
        return false;
    }
}

const QString &coConfig::getActiveCluster() const
{
    return activeCluster;
}

bool coConfig::setActiveCluster(const QString &master)
{
    if (masternames.contains(master.toLower()) || master.isEmpty())
    {
        //cerr << "coConfig::setActiveCluster info: setting active cluster "
        //     << host << endl;
        activeCluster = master.toLower();

        for (QHash<QString, coConfigGroup *>::iterator i = configGroups.begin(); i != configGroups.end(); ++i)
        {
            (*i)->setActiveCluster(activeCluster);
        }

        return true;
    }
    else
    {

        COCONFIGDBG("coConfig::setActiveCluster warn: could not set active cluster " << master.toLower());
        return false;
    }
}

void coConfig::reload()
{

    COCONFIGDBG("coConfig::reload info: reloading config");

    for (QHash<QString, coConfigGroup *>::iterator i = configGroups.begin(); i != configGroups.end(); ++i)
    {
        (*i)->reload();
    }
}

void coConfig::load()
{

    QString configGlobal = coConfigDefaultPaths::getDefaultGlobalConfigFileName();
    QString configLocal = coConfigDefaultPaths::getDefaultLocalConfigFileName();

    if (!configGlobal.isEmpty())
        COCONFIGDBG("coConfig::load info: global config file in " << configGlobal);
    if (!configLocal.isEmpty())
        COCONFIGDBG("coConfig::load info: local  config file in " << configLocal);

    coConfigGroup *mainGroup = new coConfigGroup("config");

    // Load global configuration
    mainGroup->addConfig(configGlobal, "global", false);
    mainGroup->setReadOnly("global", true);

    // Load user configuration
    mainGroup->addConfig(configLocal, "local", true);

    configGroups.insert("config", mainGroup);

    // Set active host

    this->hostnames = mainGroup->getHostnameList();
    this->masternames = mainGroup->getClusterList();

    setActiveCluster(activeCluster);
    setActiveHost(activeHostname);
}

coConfigEntryStringList coConfig::getScopeList(const QString &section,
                                               const QString &variableName) const
{

    coConfigEntryStringList merged;

    for (QHash<QString, coConfigGroup *>::const_iterator i = configGroups.begin(); i != configGroups.end(); ++i)
    {
        coConfigEntryStringList list = (*i)->getScopeList(section, variableName);
        merged = merged.merge(list);
    }

    if (variableName.isEmpty())
    {
        return merged;
    }
    else
    {
        // FIXME Do I have to?
        return merged.filter(QRegExp("^" + variableName + ":.*"));
    }
}

coConfigEntryStringList coConfig::getVariableList(const QString &section) const
{

    coConfigEntryStringList merged;

    for (QHash<QString, coConfigGroup *>::const_iterator i = configGroups.begin(); i != configGroups.end(); ++i)
    {
        coConfigEntryStringList list = (*i)->getVariableList(section);
        merged = merged.merge(list);
    }

    return merged;
}

coConfigEntryString coConfig::getValue(const QString &variable,
                                       const QString &section,
                                       const QString &defaultValue) const
{

    coConfigEntryString value = getValue(variable, section);
    if (value.isNull())
    {
        return coConfigEntryString(defaultValue);
    }

    return value;
}

coConfigEntryString coConfig::getValue(const QString &simpleVariable) const
{
    return getValue("value", simpleVariable);
}

coConfigEntryString coConfig::getValue(const QString &variable,
                                       const QString &section) const
{

    coConfigEntryString item;

    for (QHash<QString, coConfigGroup *>::const_iterator i = configGroups.begin(); i != configGroups.end(); ++i)
    {

        coConfigEntryString currentValue = (*i)->getValue(variable, section);
        if (!currentValue.isNull())
            item = currentValue;
    }

    //cerr << "coConfigGroup::getValue info: " << section << "."
    //     << variable << "=" << (item.isNull() ? "*NULL*" : item) << endl;

    return item;
}

const char *coConfig::getEntry(const char *simpleVariable) const
{

    const char *item = 0;

    for (QHash<QString, coConfigGroup *>::const_iterator i = configGroups.begin(); i != configGroups.end(); ++i)
    {

        const char *currentValue = (*i)->getEntry(simpleVariable);
        if (currentValue)
            item = currentValue;
    }

    return item;
}

coConfigString coConfig::getString(const QString &variable, const QString &section, const QString &defaultValue) const
{

    coConfigString value(variable, section);

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

coConfigString coConfig::getString(const QString &simpleVariable) const
{
    //FIXME
    return getString("value", simpleVariable, "");
}

coConfigInt coConfig::getInt(const QString &variable, const QString &section, int defaultValue) const
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

coConfigInt coConfig::getInt(const QString &simpleVariable, int defaultValue) const
{
    return getInt("value", simpleVariable, defaultValue);
}

coConfigInt coConfig::getInt(const QString &variable, const QString &section) const
{
    return coConfigInt(variable, section);
}

coConfigInt coConfig::getInt(const QString &simpleVariable) const
{
    return getInt("value", simpleVariable);
}

coConfigLong coConfig::getLong(const QString &variable, const QString &section, long defaultValue) const
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

coConfigLong coConfig::getLong(const QString &simpleVariable, long defaultValue) const
{
    return getLong("value", simpleVariable, defaultValue);
}

coConfigLong coConfig::getLong(const QString &variable, const QString &section) const
{
    return coConfigLong(variable, section);
}

coConfigLong coConfig::getLong(const QString &simpleVariable) const
{
    return getLong("value", simpleVariable);
}

coConfigBool coConfig::getBool(const QString &variable, const QString &section, bool defaultValue) const
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

coConfigBool coConfig::getBool(const QString &simpleVariable, bool defaultValue) const
{
    return getBool("value", simpleVariable, defaultValue);
}

coConfigBool coConfig::getBool(const QString &variable, const QString &section) const
{
    return coConfigBool(variable, section);
}

coConfigBool coConfig::getBool(const QString &variable, const char *section) const
{
    return coConfigBool(variable, QString(section));
}

coConfigBool coConfig::getBool(const QString &simpleVariable) const
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

coConfigFloat coConfig::getFloat(const QString &variable,
                                 const QString &section,
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
coConfigFloat coConfig::getFloat(const QString &simpleVariable, float defaultValue) const
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
coConfigFloat coConfig::getFloat(const QString &variable,
                                 const QString &section) const
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
coConfigFloat coConfig::getFloat(const QString &simpleVariable) const
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
bool coConfig::isOn(const QString &simpleVariable, bool defaultValue) const
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
bool coConfig::isOn(const QString &simpleVariable) const
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
bool coConfig::isOn(const QString &variable, const QString &section,
                    bool defaultValue) const
{

    coConfigEntryString value = getValue(variable, section);

    if (value.isNull())
        return defaultValue;

    if ((value.toLower() == "on") || (value.toLower() == "true") || (value.toInt() > 0))
        return true;
    else
        return false;
}

/**
 * @brief Checks if a variable in the configuration is set to &quot;on&quot;.
 *
 * As &quot;on&quot; count the case insensitive values of <I>on</I>, <I>true</I> and <I>1</I>.
 * @param variable Variable to check.
 * @param section Section of the variable.
 * @return <I>true</I>, if the variable has above values,
 *         <I>false</I> if not or the variable does not exist.
 */
bool coConfig::isOn(const QString &variable, const QString &section) const
{

    coConfigEntryString value = getValue(variable, section);

    if ((value.toLower() == "on") || (value.toLower() == "true") || (value == "1"))
        return true;
    else
        return false;
}

bool coConfig::isOn(const QString &variable, const char *section) const
{
    return isOn(variable, QString(section));
}

/**
 * @brief Set a value for a variable for a host.
 *
 * @param targetHost Host to set the value for.
 * @param configGroup Configuration group to set the value in.
 *
 * For other parameters, see
 * coConfig::setValue(const QString & variable, const QString & value, const QString & section, const QString & config, bool move).
 *
 */
void coConfig::setValueForHost(const QString &variable, const QString &value, const QString &section,
                               const QString &targetHost, bool move,
                               const QString &config, const QString &configGroup)
{

    coConfigEntryString oldValue = getValue(variable, section);

    coConfigGroup *group;
    QString groupName;
    QString groupConfigName;

    if (config.isNull())
    {
        //if (oldValue.isNull() || oldValue.getConfigGroupName() == "global")
        if (oldValue.isNull() || oldValue.getConfigName() == "global")
        {
            groupName = "config";
            groupConfigName = "local";
        }
        else
        {
            groupName = oldValue.getConfigGroupName();
            groupConfigName = oldValue.getConfigName();
        }
    }
    else
    {
        groupName = config;
    }

    if (!configGroup.isNull())
        groupConfigName = configGroup;

    group = configGroups[groupName];

    if (group == 0)
    {
        COCONFIGLOG("coConfig::setValue warn: unknown config group " << groupName);
        return;
    }

    //    cerr << "coConfig::setValue info: " << section << "." << variable
    //       << " = " << value << " in config '" << groupName << "'"
    //       << (targetHost ? " on host " + targetHost : QString("")) << endl;

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
 * coConfig::setValue(const QString & variable, const QString & value, const QString & section, const QString & config, bool move).
 */
void coConfig::setValueInConfig(const QString &variable, const QString &value, const QString &section,
                                const QString &configGroup, const QString &config, bool move)
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
void coConfig::setValue(const QString &variable, const QString &value, const QString &section,
                        const QString &config, bool move)
{

    setValueForHost(variable, value, section, 0, move, config);
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
void coConfig::setValue(const QString &simpleVariable, const QString &value)
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
bool coConfig::deleteValueForHost(const QString &variable, const QString &section,
                                  const QString &targetHost,
                                  const QString &config, const QString &configGroup)
{

    coConfigGroup *group;
    QString groupName;
    QString groupConfigName;

    if (config.isNull())
    {
        groupName = "config";
        groupConfigName = "local";
    }
    else
    {
        groupName = config;
        groupConfigName = "local";
    }

    if (!configGroup.isNull())
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
bool coConfig::deleteValueInConfig(const QString &variable, const QString &section,
                                   const QString &configGroup, const QString &config)
{

    return deleteValueForHost(variable, section, 0, config, configGroup);
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
bool coConfig::deleteValue(const QString &variable, const QString &section, const QString &config)
{
    return deleteValueForHost(variable, section, 0, config);
}

/**
 * @brief Delete a value from the configuration
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param simpleVariable Variable to delete. This method deletes the variable &quot;value&quot;
 * @return true, if something was deleted.
 */
bool coConfig::deleteValue(const QString &simpleVariable)
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
bool coConfig::deleteSectionForHost(const QString &section, const QString &targetHost,
                                    const QString &config, const QString &configGroup)
{

    coConfigGroup *group;
    QString groupName;
    QString groupConfigName;

    if (config.isNull())
    {
        groupName = "config";
        groupConfigName = "local";
    }
    else
    {
        groupName = config;
        groupConfigName = "local";
    }

    if (!configGroup.isNull())
        groupConfigName = configGroup;

    group = configGroups[groupName];

    if (group == 0)
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
bool coConfig::deleteSectionInConfig(const QString &section, const QString &configGroup, const QString &config)
{

    return deleteSectionForHost(section, 0, config, configGroup);
}

/**
 * @brief Delete a section in the configuration.
 * @note This is beta. Use at you own risk and report bugs.
 *
 * @param section Section to delete.
 * @param config Name of the configuration to delete this section from.
 * @return true, if something was deleted.
 */
bool coConfig::deleteSection(const QString &section, const QString &config)
{
    return deleteSectionForHost(section, 0, config);
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

    for (QHash<QString, coConfigGroup *>::const_iterator i = configGroups.begin(); i != configGroups.end(); ++i)
    {
        COCONFIGDBG("coConfig::save info: saving group " << (*i)->getGroupName());
        saved &= (*i)->save();
    }

    return saved;
}

/**
 * @brief Exports all configurations into a single file.
 *
 * @return true, if the configuration was exported successfully.
 */
bool coConfig::save(const QString &filename) const
{

    bool saved = true;

    QHash<QString, coConfigGroup *>::const_iterator group = configGroups.begin();

    coConfigGroup *merged = (*group)->clone();

    for (++group; group != configGroups.end(); ++group)
    {
        merged->merge(*group);
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

    if (QFileInfo(coConfigDefaultPaths::getDefaultGlobalConfigFileName()).isWritable())
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
void coConfig::addConfig(const QString &filename, const QString &name, bool create)
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
    configGroups.insert(group->getGroupName(), group);
    this->hostnames.append(group->getHostnameList());
    this->hostnames.removeDuplicates();
    this->masternames.append(group->getClusterList());
    this->masternames.removeDuplicates();
}

/**
 * @brief Removes a configuration group from the global coConfig.
 *
 * @param name Name of the configuration.
 */
void coConfig::removeConfig(const QString &name)
{
    configGroups["config"]->removeConfig(name);

    this->hostnames.clear();
    this->masternames.clear();
    foreach (coConfigGroup *group, configGroups)
    {
        this->hostnames.append(group->getHostnameList());
        this->masternames.append(group->getClusterList());
    }
    this->hostnames.removeDuplicates();
    this->masternames.removeDuplicates();
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
