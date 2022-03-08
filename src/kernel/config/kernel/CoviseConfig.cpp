/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigLog.h>
#include <config/CoviseConfig.h>

#include <config/coConfig.h>

#include <iostream>

using namespace covise;

coCoviseConfig::coCoviseConfig()
{
}

coCoviseConfig::~coCoviseConfig() {}

/**
 * @brief Get a config entry.
 * The entry takes the form "{sectionname.}*entry, whereby sectionname is
 * section[:name]. This method returns the "value"-field of the config entry only.
 * @param entry The section to get the value from.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the section, or an empty string if nothing found.
 */
std::string coCoviseConfig::getEntry(const std::string &entry, bool *exists)
{
    return getEntry("value", entry, "", exists);
}

/**
 * @brief Get a config entry.
 * The entry takes the form "{sectionname.}*entry, whereby sectionname is
 * section[:name]. This method returns the field given by "variable" of the config entry.
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or an empty string if nothing found.
 */

std::string coCoviseConfig::getEntry(const std::string &variable, const std::string &entry, bool *exists)
{
    return getEntry(variable, entry, "", exists);
}

/**
 * @brief Get a config entry.
 * The entry takes the form "{sectionname.}*entry, whereby sectionname is
 * section[:name]. This method returns the field given by "variable" of the config entry.
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */

std::string coCoviseConfig::getEntry(const std::string &variable, const std::string &entry, const std::string &defaultValue, bool *exists)
{
    std::string val = coConfig::getInstance()->getValue(variable, entry).entry;

    if (val.empty())
    {
        if (exists)
            *exists = false;
        return defaultValue;
    }

    if (exists)
        *exists = true;

    COCONFIGDBG("coCoviseConfig::getEntry info: " << entry << "/" << variable << " = " << val);

    return val;
}

/**
 * @brief Get an integer value
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the section, or the default value if nothing found.
 */
int coCoviseConfig::getInt(const std::string &entry, int defaultValue, bool *exists)
{
    return getInt("value", entry, defaultValue, exists);
}

/**
 * @brief Get an integer value
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */
int coCoviseConfig::getInt(const std::string &variable, const std::string &entry, int defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::getInt info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigInt val = coConfig::getInstance()->getInt(variable, entry);
    COCONFIGDBG("coCoviseConfig::getInt info: " << entry.c_str() << "/" << variable.c_str() << " = " << val
                                                << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

/**
 * @brief Get a long integer value
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the section, or the default value if nothing found.
 */
long coCoviseConfig::getLong(const std::string &entry, long defaultValue, bool *exists)
{
    return getLong("value", entry, defaultValue, exists);
}

/**
 * @brief Get a long integer value
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */
long coCoviseConfig::getLong(const std::string &variable, const std::string &entry, long defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::getLong info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigLong val = coConfig::getInstance()->getLong(variable, entry);
    COCONFIGDBG("coCoviseConfig::getLong info: " << entry.c_str() << "/" << variable.c_str() << " = " << val
                                                 << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

/**
 * @brief Checks if an option is on or off. As on counts "on", "true", and "1"
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return true, if "value" in the denoted section is on, or the default value if nothing found.
 */

bool coCoviseConfig::isOn(const std::string &entry, bool defaultValue, bool *exists)
{
    return isOn("value", entry, defaultValue, exists);
}

/**
 * @brief Checks if an option is on or off. As on counts "on", "true", and "1"
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return true, if the variable in the denoted section is on, or the default value if nothing found.
 */
bool coCoviseConfig::isOn(const std::string &variable, const std::string &entry, bool defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::isOn info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigBool val = coConfig::getInstance()->getBool(variable, entry);
    COCONFIGDBG("coCoviseConfig::isOn info: " << entry.c_str() << "/" << variable.c_str() << " = " << val
                                              << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

/**
 * @brief Get a float value
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the denoted section, or the default value if nothing found.
 */
float coCoviseConfig::getFloat(const std::string &entry, float defaultValue, bool *exists)
{
    return getFloat("value", entry, defaultValue, exists);
}

/**
 * @brief Get a float value
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */
float coCoviseConfig::getFloat(const std::string &variable, const std::string &entry, float defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::getFloat info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigFloat val = coConfig::getInstance()->getFloat(variable, entry);
    COCONFIGDBG("coCoviseConfig::getFloat info: " << entry.c_str() << " = " << val
                                                  << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

std::vector<std::string> getScopeNamesHelper(const coCoviseConfig::ScopeEntries &entries)
{
    std::vector<std::string> result;
    for (const auto &pair : entries)
    {
        auto pos = pair.first.find(':');
        if (pos != std::string::npos)
            result.emplace_back(pair.first.substr(pos + 1));
        else
            result.push_back(pair.first);
    }
    return result;
}

std::vector<std::string> coCoviseConfig::getScopeNames(const std::string &scope, const std::string &name)
{
    if (name.empty())
        return getScopeNamesHelper(getScopeEntries(scope));
    else
        return getScopeNamesHelper(getScopeEntries(scope, name));
}

coCoviseConfig::ScopeEntries coCoviseConfig::getScopeEntries(const std::string &scope, const std::string &name)
{

    ScopeEntries entries;
    coConfigEntryStringList list = coConfig::getInstance()->getScopeList(scope, name);

    COCONFIGDBG("coCoviseConfig::ScopeEntries::<init>(" << scope << ", " << name << ": size = " << list.entries().size());

    if (list.entries().size() == 0)
        return entries;

    if (list.getListType() == coConfigEntryStringList::PLAIN_LIST)
    {
        COCONFIGDBG("coCoviseConfig::getScopeEntries info: PLAIN_LIST");
        for (const auto entry : list.entries())
        {
            auto pos = entry.entry.find(' ');
            if (pos != std::string::npos)
            {
                entries.insert({entry.entry.substr(0, pos), entry.entry.substr(pos + 1)});
            }
            // COCONFIGLOG("coCoviseConfig::getScopeEntries info: " << rv_[ctr - 2] << " = " << rv_[ctr - 1]);
        }
    }
    else if (list.getListType() == coConfigEntryStringList::VARIABLE)
    {
        COCONFIGDBG("coCoviseConfig::getScopeEntries info: VARIABLE");
        for (const auto entry : list.entries())
        {
            entries.insert({entry.entry, coConfig::getInstance()->getValue(scope + "." + entry.entry).entry});
        }
    }
    else
    {
        COCONFIGLOG("coCoviseConfig::getScopeEntries warn: UNKNOWN");
    }
    return entries;
}
