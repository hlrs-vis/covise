/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigGroup.h>
#include <config/coConfigLog.h>
#include <config/coConfigConstants.h>
#include "coConfigXercesRoot.h"

#include <iostream>
using namespace std;

#include <QRegExp>
#include <QTextStream>

using namespace covise;

coConfigGroup::coConfigGroup(const QString &groupName)
{

    this->groupName = groupName;
    activeHostname = coConfigConstants::getHostname();
    readOnly = false;
}

coConfigGroup::~coConfigGroup()
{
    for (QHash<QString, coConfigRoot *>::iterator entry = this->configs.begin(); entry != this->configs.end(); ++entry)
    {
        delete entry.value();
    }
    this->configs.clear();
}

coConfigGroup::coConfigGroup(const coConfigGroup *source)
    : activeHostname(source->activeHostname)
    , hostnames(source->hostnames)
    , masternames(source->masternames)
    , activeCluster(source->activeCluster)
    , groupName(source->groupName)
    , readOnly(source->readOnly)
{
    for (QHash<QString, coConfigRoot *>::const_iterator entry = source->configs.begin(); entry != source->configs.end(); ++entry)
    {
        coConfigRoot *root = entry.value()->clone();
        root->setGroup(this);
        this->configs.insert(entry.key(), root);
    }
}

coConfigRoot *coConfigGroup::addConfig(const QString &filename, const QString &name, bool create)
{

    if (configs[name])
    {
        COCONFIGLOG("coConfigGroup::addConfig err: tried to add config " << filename << " twice");
        return configs[name];
    }

    coConfigRoot *configRoot = new coConfigXercesRoot(name, filename, create, this);
    configs.insert(name, configRoot);
    return configRoot;
}

void coConfigGroup::removeConfig(const QString &name)
{
    delete configs.take(name);
}

QStringList coConfigGroup::getHostnameList() /*const*/
{
    hostnames.clear();
    for (QHash<QString, coConfigRoot *>::const_iterator configRoots = configs.begin(); configRoots != configs.end(); ++configRoots)
    {
        hostnames += configRoots.value()->getHostnameList();
    }
    hostnames.removeDuplicates();
    return hostnames;
}

QString coConfigGroup::getActiveHost() const
{
    return activeHostname;
}

bool coConfigGroup::setActiveHost(const QString &host)
{

    activeHostname = host;

    for (QHash<QString, coConfigRoot *>::iterator i = configs.begin(); i != configs.end(); ++i)
    {
        (*i)->setActiveHost(host);
    }

    return true;
}

QStringList coConfigGroup::getClusterList() /*const*/
{
    masternames.clear();
    for (QHash<QString, coConfigRoot *>::const_iterator configRoots = configs.begin(); configRoots != configs.end(); ++configRoots)
    {
        masternames += configRoots.value()->getClusterList();
    }
    masternames.removeDuplicates();
    return masternames;
}

QString coConfigGroup::getActiveCluster() const
{
    return activeCluster;
}

bool coConfigGroup::setActiveCluster(const QString &master)
{

    activeCluster = master;

    for (QHash<QString, coConfigRoot *>::iterator i = configs.begin(); i != configs.end(); ++i)
    {
        (*i)->setActiveCluster(master);
    }

    return true;
}

void coConfigGroup::reload()
{

    COCONFIGDBG("coConfigGroup::reload info: reloading config");

    for (QHash<QString, coConfigRoot *>::iterator i = configs.begin(); i != configs.end(); ++i)
    {
        (*i)->reload();
    }
}

coConfigEntryStringList coConfigGroup::getScopeList(const QString &section,
                                                    const QString &variableName) const
{

    coConfigEntryStringList merged;

    for (QHash<QString, coConfigRoot *>::const_iterator i = configs.begin(); i != configs.end(); ++i)
    {
        coConfigEntryStringList list = (*i)->getScopeList(section, variableName);
        merged.merge(list);
    }

    if (variableName.isEmpty())
    {
        return merged;
    }
    else
    {
        return merged.filter(QRegExp("^" + variableName + ":.*"));
    }
}

coConfigEntryStringList coConfigGroup::getVariableList(const QString &section) const
{

    coConfigEntryStringList merged;

    for (QHash<QString, coConfigRoot *>::const_iterator i = configs.begin(); i != configs.end(); ++i)
    {
        coConfigEntryStringList list = (*i)->getVariableList(section);
        merged.merge(list);
    }

    return merged;
}

coConfigEntryString coConfigGroup::getValue(const QString &variable,
                                            const QString &section,
                                            const QString &defaultValue) const
{

    coConfigEntryString value = getValue(variable, section);

    if (value.isNull())
    {
        return coConfigEntryString(defaultValue);
    }

    value.setConfigGroupName(groupName);
    return value;
}

coConfigEntryString coConfigGroup::getValue(const QString &simpleVariable) const
{
    return getValue("value", simpleVariable);
}

coConfigEntryString coConfigGroup::getValue(const QString &variable,
                                            const QString &section) const
{

    coConfigEntryString item;

    for (QHash<QString, coConfigRoot *>::const_iterator i = configs.begin(); i != configs.end(); ++i)
    {
        coConfigEntryString currentValue = (*i)->getValue(variable, section);
        if (!currentValue.isNull())
            item = currentValue;
    }

    item.setConfigGroupName(groupName);

    COCONFIGDBG_GET_SET("coConfigGroup::getValue info: [" << groupName << "] "
                                                          << section << "." << variable << " = "
                                                          << (item.isNull() ? "*NO VALUE*" : item.toLatin1()));

    return item;
}

const char *coConfigGroup::getEntry(const char *variable) const
{

    const char *item = NULL;

    for (QHash<QString, coConfigRoot *>::const_iterator i = configs.begin(); i != configs.end(); ++i)
    {
        const char *currentValue = (*i)->getEntry(variable);
        if (currentValue)
            item = currentValue;
    }

    return item;
}

bool coConfigGroup::isOn(const QString &simpleVariable, bool defaultValue) const
{
    return isOn("value", simpleVariable, defaultValue);
}

bool coConfigGroup::isOn(const QString &simpleVariable) const
{
    return isOn("value", simpleVariable);
}

bool coConfigGroup::isOn(const QString &variable, const QString &section,
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

bool coConfigGroup::isOn(const QString &variable, const QString &section) const
{

    coConfigEntryString value = getValue(variable, section);

    if ((value.toLower() == "on") || (value.toLower() == "true") || (value == "1"))
        return true;
    else
        return false;
}

void coConfigGroup::setValue(const QString &simpleVariable, const QString &value)
{
    setValue("value", value, simpleVariable);
}

void coConfigGroup::setValue(const QString &variable, const QString &value,
                             const QString &section,
                             const QString &configuration,
                             const QString &targetHost, bool move)
{

    if (isReadOnly())
        return;

    QString configurationName = configuration;

    if (configuration.isEmpty())
    {
        //cerr << "coConfigGroup::setValue info: getting " << variable << " in " << section << endl;
        coConfigEntryString oldValue = getValue(variable, section);

        if (oldValue.isNull())
        {
            QList<QString> keys = configs.keys();
            configurationName = keys.first();
            //cerr << "coConfigGroup::setValue info: autoselecting config " << configurationName << endl;
        }
        else
        {
            configurationName = oldValue.getConfigName();
            //cerr << "coConfigGroup::setValue info: autoselecting config (from old value) " << configurationName << endl;
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

bool coConfigGroup::deleteValue(const QString &variable, const QString &section,
                                const QString &configuration, const QString &targetHost)

{

    if (isReadOnly())
        return false;

    QString configurationName = configuration;

    if (configuration == QString::null)
    {

        coConfigEntryString oldValue = getValue(variable, section);

        if (oldValue.isNull())
        {
            return false;
        }
        else
        {
            configurationName = oldValue.getConfigName();
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

bool coConfigGroup::deleteSection(const QString &section, const QString &configuration, const QString &targetHost)

{

    if (isReadOnly())
        return false;

    bool removed = false;
    QString configurationName = configuration;

    if (configuration == QString::null)
    {
        for (QHash<QString, coConfigRoot *>::iterator i = configs.begin(); i != configs.end(); ++i)
        {
            coConfigRoot *root = *i;
            if (root)
                removed |= root->deleteSection(section, targetHost);
        }
    }

    return removed;
}

bool coConfigGroup::save(const QString &filename) const
{

    if (isReadOnly())
    {
        COCONFIGDBG("coConfigGroup::save warn: not saving read only config");
        return true;
    }

    if (filename == QString::null && configs.count() > 1)
    {
        COCONFIGDBG("coConfigGroup::save warn: saving more than one config group, only the last one will be saved, consider flattening the configuration");
    }

    bool saved = true;

    for (QHash<QString, coConfigRoot *>::const_iterator i = configs.begin(); i != configs.end(); ++i)
    {
        COCONFIGDBG("coConfigGroup::save info: saving root " << (*i)->getConfigName());
        saved &= (*i)->save(filename);
    }

    return saved;
}

const QString &coConfigGroup::getGroupName() const
{
    return groupName;
}

void coConfigGroup::setReadOnly(bool ro)
{
    readOnly = ro;

    for (QHash<QString, coConfigRoot *>::iterator i = configs.begin(); i != configs.end(); ++i)
    {
        (*i)->setReadOnly(ro);
    }
}

bool coConfigGroup::isReadOnly() const
{
    return readOnly;
}

void coConfigGroup::setReadOnly(const QString &config, bool ro)
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

bool coConfigGroup::isReadOnly(const QString &config) const
{
    coConfigRoot *root = configs[config];

    if (root == 0)
    {
        COCONFIGLOG("coConfigGroup::isReadOnly warn: no such configuration: " << config);
        return false;
    }
    else
    {
        return root->isReadOnly();
    }
}

coConfigGroup *coConfigGroup::clone() const
{
    return new coConfigGroup(this);
}

void coConfigGroup::merge(const coConfigGroup *with)
{
    for (QHash<QString, coConfigRoot *>::const_iterator entry = with->configs.begin(); entry != with->configs.end(); ++entry)
    {
        if (configs.contains(entry.key()))
        {
            configs[entry.key()]->merge(entry.value());
        }
        else
        {
            coConfigRoot *root = entry.value()->clone();
            root->setGroup(this);
            this->configs.insert(entry.key(), root);
        }
    }
}

/**
 * Flattens the configuration structure that only one tree remains.
 * The name of the configuration is taken from the first configuration.
 */

void coConfigGroup::flatten()
{
    if (configs.count() < 2)
        return;

    QHash<QString, coConfigRoot *>::const_iterator entry = configs.begin();

    QString mainKey = entry.key();
    coConfigRoot *root = entry.value();

    for (++entry; entry != configs.end(); ++entry)
    {
        root->merge(entry.value());
        delete entry.value();
    }

    configs.clear();
    configs.insert(mainKey, root);
}
