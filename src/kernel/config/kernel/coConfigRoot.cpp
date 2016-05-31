/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigRoot.h>
#include "coConfigXercesRoot.h"
#include "coConfigXercesEntry.h"
#include <config/coConfigLog.h>
#include <config/coConfigConstants.h>
#include "coConfigRootErrorHandler.h"
#include "coConfigTools.h"

#include <QFileInfo>
#include <QDir>

#include <QRegExp>
#include <QTextStream>

#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/internal/XMLGrammarPoolImpl.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/validators/common/GrammarResolver.hpp>
#include <xercesc/validators/schema/SchemaGrammar.hpp>

using namespace covise;

coConfigRoot *coConfigRoot::createNew(const QString &name, const QString &filename,
                                      bool create, coConfigGroup *group)
{
    return new coConfigXercesRoot(name, filename, create, group);
}

coConfigRoot::coConfigRoot(const QString &name, const QString &filename,
                           bool create, coConfigGroup *group)
    : globalConfig(0)
    , clusterConfig(0)
    , hostConfig(0)
{

    this->filename = filename;
    this->create = create;
    configName = name;
    this->group = group;
    readOnly = false;

    init();
}

coConfigRoot::~coConfigRoot()
{
    clear();
}

/**
 * Creates a copy of a coConfigXercesRoot. The group the copy belongs to is set to 0. 
 * Makes a deep copy of all coConfigEntries belonging to this group.
 */

coConfigXercesRoot::coConfigXercesRoot(const coConfigXercesRoot *source)
    : coConfigRoot(source->configName, source->filename, source->create, 0)
{
    this->activeHostname = source->activeHostname;
    this->activeCluster = source->activeCluster;
    this->hostnames = source->hostnames;
    this->masternames = source->masternames;

    this->configName = source->configName;

    this->globalConfig = source->globalConfig->clone();

    for (QHash<QString, coConfigEntry *>::const_iterator entry = source->hostConfigs.begin(); entry != source->hostConfigs.end(); ++entry)
    {
        if (entry.value())
            this->hostConfigs.insert(entry.key(), entry.value()->clone());
        else
            this->hostConfigs.insert(entry.key(), 0);
    }

    if (this->hostConfigs.contains(this->activeHostname))
        this->hostConfig = this->hostConfigs[this->activeHostname];
    else
        this->hostConfig = 0;

    for (QHash<QString, coConfigEntry *>::const_iterator entry = source->clusterConfigs.begin(); entry != source->clusterConfigs.end(); ++entry)
    {
        if (entry.value())
            this->clusterConfigs.insert(entry.key(), entry.value()->clone());
        else
            this->clusterConfigs.insert(entry.key(), 0);
    }

    if (this->clusterConfigs.contains(this->activeCluster))
        this->clusterConfig = this->clusterConfigs[this->activeCluster];
    else
        this->clusterConfig = 0;

    this->create = source->create;
    this->readOnly = source->readOnly;
}

coConfigXercesRoot::coConfigXercesRoot(const QString &name, const QString &filename,
                                       bool create, coConfigGroup *group)
    : coConfigRoot(name, filename, create, group)
{
    load(create);
}

coConfigXercesRoot::coConfigXercesRoot(const xercesc::DOMNode *node, const QString &name,
                                       const QString &filename, coConfigGroup *group)
    : coConfigRoot(name, filename, false, group)
{
    xercesc::DOMNode *globalConfigNode = 0;
    xercesc::DOMNodeList *nodeList = node->getChildNodes();
    for (int it = 0; it < nodeList->getLength(); ++it)
    {
        QString nodeName = QString::fromUtf16(reinterpret_cast<const ushort *>(nodeList->item(it)->getNodeName()));
        if (nodeName == "COCONFIG")
        {
            globalConfigNode = nodeList->item(it);
            COCONFIGDBG("coConfigRoot:coConfigRoot info: COCONFIG found in init");
        }

        if (!globalConfigNode)
        {
            setContentsFromDom(node);
        }
        else
        {
            setContentsFromDom(globalConfigNode);
        }
    }
}

coConfigXercesRoot::~coConfigXercesRoot()
{
}

void coConfigRoot::init()
{

    activeHostname = coConfigConstants::getHostname();
    activeCluster = coConfigConstants::getMaster();
    globalConfig = 0;
}

void coConfigRoot::setGroup(coConfigGroup *group)
{
    this->group = group;
}

QStringList coConfigRoot::getHostnameList() const
{
    return hostnames;
}

QString coConfigRoot::getActiveHost() const
{
    return activeHostname;
}

bool coConfigRoot::setActiveHost(const QString &host)
{
    if (hostnames.contains(host.toLower()))
    {
        //cerr << "coConfigRoot::setActiveHost info: setting active host "
        //     << host << endl;
        activeHostname = host.toLower();

        hostConfig = hostConfigs[activeHostname];
        if (hostConfig == 0)
        {
            hostConfigs[activeHostname] = hostConfig = 0;
        }

        return true;
    }
    else
    {

        COCONFIGLOG("coConfigRoot::setActiveHost warn: could not set active host " << host.toLower() << ", this: " << this);
        return false;
    }
}

QStringList coConfigRoot::getClusterList() const
{
    return masternames;
}

QString coConfigRoot::getActiveCluster() const
{
    return activeCluster;
}

bool coConfigRoot::setActiveCluster(const QString &master)
{
    if (masternames.contains(master.toLower()) || master.isEmpty())
    {
        //cerr << "coConfigRoot::setActiveCluster info: setting active master "
        //     << master << endl;
        activeCluster = master.toLower();

        if (master.isEmpty())
        {
            clusterConfig = 0;
        }
        else
        {
            clusterConfig = clusterConfigs[activeCluster];
            if (clusterConfig == 0)
            {
                clusterConfigs[activeCluster] = clusterConfig = 0;
            }
        }

        return true;
    }
    else
    {

        COCONFIGDBG("coConfigRoot::setActiveCluster warn: could not set active cluster " << master.toLower());
        return false;
    }
}

const QString &coConfigRoot::getConfigName() const
{
    return configName;
}

void coConfigRoot::reload()
{

    COCONFIGDBG("coConfigRoot::reload info: reloading config " << filename);

    clear();
    load();
}

void coConfigXercesRoot::load(bool create)
{

    QFileInfo configfile(findConfigFile(filename));
    if (configfile.isFile())
    {
        xercesc::DOMNode *globalConfigNode = loadFile(configfile.filePath());
        if (globalConfigNode)
        {
            setContentsFromDom(globalConfigNode);
        }
    }
    else
    {

        if (create)
        {
            COCONFIGDBG("coConfigRoot::load info: Creating config file " << filename);
            createGlobalConfig();
        }
        else
        {
            COCONFIGDBG("coConfigRoot::load warn: Could not open config file " << filename);
        }
    }

    if (!hostnames.contains(coConfigConstants::getHostname()))
    {
        hostnames.append(coConfigConstants::getHostname());
    }
    if (!coConfigConstants::getMaster().isEmpty() && !masternames.contains(coConfigConstants::getMaster()))
    {
        masternames.append(coConfigConstants::getMaster());
    }

    if (!hostnames.contains(activeHostname))
    {
        activeHostname = coConfigConstants::getHostname();
    }
    if (!masternames.contains(activeCluster))
    {
        activeCluster = coConfigConstants::getMaster();
    }

    setActiveCluster(activeCluster);
    setActiveHost(activeHostname);
}

void coConfigXercesRoot::setContentsFromDom(const xercesc::DOMNode *node)
{
    //COCONFIGLOG("coConfigRoot::setContentsFromDom info: creating tree from " << configName);

    if (node)
    {
        xercesc::DOMNodeList *nodeList = node->getChildNodes();
        for (int i = 0; i < nodeList->getLength(); ++i)
        {

            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;

            QString nodeName = QString::fromUtf16(reinterpret_cast<const ushort *>(node->getNodeName()));

            if (nodeName == "GLOBAL")
            {
                xercesc::DOMElement *globalConfigNode = node;
                if (!globalConfigNode)
                {
                    COCONFIGLOG("coConfigRoot::setContentsFromDom err: global config no element?");
                }
                else
                {
                    // Temporary attributes
                    globalConfigNode->setAttribute(xercesc::XMLString::transcode("scope"),
                                                   xercesc::XMLString::transcode("global"));

                    globalConfigNode->setAttribute(xercesc::XMLString::transcode("configname"),
                                                   reinterpret_cast<const XMLCh *>(configName.utf16()));

                    if (globalConfig == 0)
                    {
                        globalConfig = coConfigXercesEntry::restoreFromDom(globalConfigNode, configName);
                    }
                    else
                    {
                        globalConfig->merge(coConfigXercesEntry::restoreFromDom(node, configName));
                    }
                    // Temporary attributes cleanup
                    globalConfig->deleteValue("scope", QString::null);
                    globalConfig->deleteValue("configname", QString::null);
                }
            }
            else if (nodeName == "LOCAL")
            {

                // Temporary attributes
                node->setAttribute(xercesc::XMLString::transcode("scope"),
                                   xercesc::XMLString::transcode("host"));

                node->setAttribute(xercesc::XMLString::transcode("configname"),
                                   reinterpret_cast<const XMLCh *>(configName.utf16()));

                QString hostTemp = QString::fromUtf16(reinterpret_cast<const ushort *>(node->getAttribute(xercesc::XMLString::transcode("HOST"))));

                QStringList hosts = hostTemp.split(',', QString::SkipEmptyParts);
                for (QStringList::iterator i = hosts.begin(); i != hosts.end(); ++i)
                {
                    QString hostname = (*i).trimmed().toLower();
                    //cerr << "coConfigRoot::setContentsFromDom info: adding for host " << hostname.latin1() << endl;
                    if (!hostnames.contains(hostname))
                    {
                        hostnames.append(hostname);
                        hostConfigs.insert(hostname, coConfigXercesEntry::restoreFromDom(node, configName));
                    }
                    else
                    {
                        if (hostConfigs[hostname] == 0)
                            hostConfigs.insert(hostname, coConfigXercesEntry::restoreFromDom(node, configName));
                        else
                            hostConfigs[hostname]->merge(coConfigXercesEntry::restoreFromDom(node, configName));

                        // Temporary attributes cleanup
                        if (globalConfig)
                        {
                            globalConfig->deleteValue("scope", QString::null);
                            globalConfig->deleteValue("configname", QString::null);
                        }
                    }
                }
            }
            else if (nodeName == "CLUSTER")
            {

                // Temporary attributes
                node->setAttribute(xercesc::XMLString::transcode("scope"),
                                   xercesc::XMLString::transcode("cluster"));

                node->setAttribute(xercesc::XMLString::transcode("configname"),
                                   reinterpret_cast<const XMLCh *>(configName.utf16()));

                QString hostTemp = QString::fromUtf16(reinterpret_cast<const ushort *>(node->getAttribute(xercesc::XMLString::transcode("MASTER"))));

                QStringList hosts = hostTemp.split(',', QString::SkipEmptyParts);
                for (QStringList::iterator i = hosts.begin(); i != hosts.end(); ++i)
                {
                    QString hostname = (*i).trimmed().toLower();
                    if (!masternames.contains(hostname))
                    {
                        masternames.append(hostname);
                        clusterConfigs.insert(hostname, coConfigXercesEntry::restoreFromDom(node, configName));
                    }
                    else
                    {
                        if (clusterConfigs[hostname] == 0)
                        {
                            clusterConfigs.insert(hostname, coConfigXercesEntry::restoreFromDom(node, configName));
                        }
                        else
                        {
                            clusterConfigs[hostname]->merge(coConfigXercesEntry::restoreFromDom(node, configName));
                        }

                        // Temporary attributes cleanup
                        if (globalConfig)
                        {
                            globalConfig->deleteValue("scope", QString::null);
                            globalConfig->deleteValue("configname", QString::null);
                        }
                    }
                }
            }
            else if (nodeName == "INCLUDE" || nodeName == "TRYINCLUDE")
            {
                QHash<QString, QString *> attributes;
                QString arch, host, rank, master;
                const ushort *archUtf16 = reinterpret_cast<const ushort *>(node->getAttribute(xercesc::XMLString::transcode("arch")));
                const ushort *rankUtf16 = reinterpret_cast<const ushort *>(node->getAttribute(xercesc::XMLString::transcode("rank")));
                const ushort *hostUtf16 = reinterpret_cast<const ushort *>(node->getAttribute(xercesc::XMLString::transcode("host")));
                const ushort *masterUtf16 = reinterpret_cast<const ushort *>(node->getAttribute(xercesc::XMLString::transcode("master")));
                if (archUtf16)
                {
                    arch = QString::fromUtf16(archUtf16);
                    if (!arch.isEmpty())
                        attributes["ARCH"] = &arch;
                }
                if (rankUtf16)
                {
                    rank = QString::fromUtf16(rankUtf16);
                    if (!rank.isEmpty())
                        attributes["RANK"] = &rank;
                }
                if (hostUtf16)
                {
                    host = QString::fromUtf16(hostUtf16);
                    if (!host.isEmpty())
                        attributes["HOST"] = &host;
                }
                if (masterUtf16)
                {
                    master = QString::fromUtf16(masterUtf16);
                    if (!master.isEmpty())
                        attributes["MASTER"] = &master;
                }

                if (coConfigTools::matchingAttributes(attributes))
                {
                    QString filename = QString::fromUtf16(reinterpret_cast<const ushort *>(node->getFirstChild()->getNodeValue())).trimmed();
                    if (!included.contains(filename))
                    {
                        COCONFIGDBG("coConfigRoot::setContentsFromDom info: INCLUDE:  filename: " << filename);

                        xercesc::DOMNode *includeNode = 0;

                        // try in current directory first
                        QDir pwd = QDir(this->filename);
                        pwd.cdUp();
                        QString localPath = pwd.filePath(filename);
                        if (QFileInfo(localPath).isFile())
                        {
                            includeNode = loadFile(localPath);
                        }
                        else if (!node->getAttribute(xercesc::XMLString::transcode("global")) || QString::fromUtf16(reinterpret_cast<const ushort *>(node->getAttribute(xercesc::XMLString::transcode("global")))) == "0")
                            includeNode = loadFile(findConfigFile(filename, false));
                        else
                            includeNode = loadFile(findConfigFile(filename, true));

                        if (!includeNode)
                        {
                            COCONFIGLOG("coConfigRoot::setContentsFromDom error: could not open include file " << filename);
                            if (nodeName != "TRYINCLUDE")
                                exit(1);
                        }
                        else
                        {
                            included.insert(filename);
                            setContentsFromDom(includeNode);
                        }
                    }
                    else
                    {
                        COCONFIGLOG("coConfigRoot::setContentsFromDom info: ALREADY INCLUDED:  filename: " << filename);
                    }
                }
            }
        }
    }

    if (!coConfigConstants::getMaster().isEmpty() && !masternames.contains(coConfigConstants::getMaster()))
    {
        masternames.append(coConfigConstants::getMaster());
    }

    if (!masternames.contains(activeCluster))
    {
        activeCluster = coConfigConstants::getMaster();
    }
    setActiveCluster(activeCluster);
    //
    // Set active host
    //

    if (!hostnames.contains(coConfigConstants::getHostname()))
    {
        hostnames.append(coConfigConstants::getHostname());
    }

    if (!hostnames.contains(activeHostname))
    {
        activeHostname = coConfigConstants::getHostname();
    }
    setActiveHost(activeHostname);
}

coConfigEntryStringList coConfigRoot::getScopeList(const QString &section, const QString &variableName) const
{

    coConfigEntryStringList global;
    if (globalConfig)
        global = globalConfig->getScopeList(section);
    coConfigEntryStringList cluster;
    if (clusterConfig)
        cluster = clusterConfig->getScopeList(section);
    coConfigEntryStringList host;
    if (hostConfig)
        host = hostConfig->getScopeList(section);

    coConfigEntryStringList merged = global.merge(cluster);
    merged = merged.merge(host);

    if (variableName.isEmpty())
    {
        return merged;
    }
    else
    {
        return merged.filter(QRegExp("^" + variableName + ":.*"));
    }
}

coConfigEntryStringList coConfigRoot::getVariableList(const QString &section) const
{

    coConfigEntryStringList global;
    if (globalConfig)
        global = globalConfig->getScopeList(section);
    coConfigEntryStringList cluster;
    if (clusterConfig)
        cluster = clusterConfig->getScopeList(section);
    coConfigEntryStringList host;
    if (hostConfig)
        host = hostConfig->getScopeList(section);

    return global.merge(cluster).merge(host);
}

coConfigEntryString coConfigRoot::getValue(const QString &variable,
                                           const QString &section,
                                           const QString &defaultValue) const
{

    coConfigEntryString value = getValue(variable, section);
    if (value.isNull())
    {
        return coConfigEntryString(defaultValue).setConfigName(configName);
    }
    return value;
}

coConfigEntryString coConfigRoot::getValue(const QString &simpleVariable) const
{
    return getValue("value", simpleVariable);
}

coConfigEntryString coConfigRoot::getValue(const QString &variable,
                                           const QString &section) const
{

    if (hostConfig)
    {
        coConfigEntryString localItem = hostConfig->getValue(variable, section);

        if (!localItem.isNull())
        {
            localItem.setConfigName(configName);
            return localItem;
        }
    }

    if (clusterConfig)
    {
        coConfigEntryString localItem = clusterConfig->getValue(variable, section);

        if (!localItem.isNull())
        {
            localItem.setConfigName(configName);
            return localItem;
        }
    }

    if (globalConfig)
    {
        coConfigEntryString globalItem = globalConfig->getValue(variable, section);

        if (!globalItem.isNull())
        {
            globalItem.setConfigName(configName);
            return globalItem;
        }
    }

    //cerr << "coConfig::getValue info: " << section << "."
    //<< variable << "=" << (item.isNull() ? "*NULL*" : item) << endl;

    return coConfigEntryString();
}

const char *coConfigRoot::getEntry(const char *variable) const
{

    const char *item = hostConfig->getEntry(variable);

    if (!item && clusterConfig)
    {
        item = clusterConfig->getEntry(variable);
    }

    if (!item && globalConfig)
    {
        item = globalConfig->getEntry(variable);
    }

    return item;
}

bool coConfigRoot::isOn(const QString &simpleVariable, bool defaultValue) const
{
    return isOn("value", simpleVariable, defaultValue);
}

bool coConfigRoot::isOn(const QString &simpleVariable) const
{
    return isOn("value", simpleVariable);
}

bool coConfigRoot::isOn(const QString &variable, const QString &section,
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

bool coConfigRoot::isOn(const QString &variable, const QString &section) const
{

    coConfigEntryString value = getValue(variable, section);

    if ((value.toLower() == "on") || (value.toLower() == "true") || (value == "1"))
        return true;
    else
        return false;
}

void coConfigRoot::setValue(const QString &variable, const QString &value,
                            const QString &section,
                            const QString &targetHost, bool move)
{

    if (move)
    {
        COCONFIGLOG("coConfigRoot::setValue fixme: move not implemented, copying");
    }

    if (isReadOnly())
    {
        COCONFIGDBG("coConfigRoot::setValue warn: not setting value in read only root");
        return;
    }

    //coConfigEntryString oldValue = getValue(variable, section);

    coConfigConstants::ConfigScope scope = coConfigConstants::Default;

    //if (oldValue.isNull()) {
    //  scope = coConfigConstants::Global;
    //} else {
    //  scope = oldValue.getConfigScope();
    //}

    if (targetHost == QString::null)
    {
        scope = coConfigConstants::Global;
    }
    else
    {
        scope = coConfigConstants::Host;
    }

    COCONFIGDBG_GET_SET("coConfig::setValue info: " << section << " - " << variable << " = " << value << " in scope " << scope);

    switch (scope)
    {
    case coConfigConstants::Global:
        if (!globalConfig)
            createGlobalConfig();
        if (globalConfig && !globalConfig->setValue(variable, value, section))
            globalConfig->addValue(variable, value, section);
        break;

    case coConfigConstants::Cluster:
        if (targetHost == QString::null)
        {
            if (!clusterConfig->setValue(variable, value, section))
                clusterConfig->addValue(variable, value, section);
        }
        else
        {
            coConfigEntry *clusterConfig = clusterConfigs[coConfigConstants::getMaster()];
            if (clusterConfig == 0)
            {
                createClusterConfig(targetHost);
            }
            if (!clusterConfigs[targetHost]->setValue(variable, value, section))
                clusterConfigs[targetHost]->addValue(variable, value, section);
        }
        break;

    case coConfigConstants::Host:
        if ((targetHost.toLower() == activeHostname) || (targetHost == QString::null))
        {
            if (!hostConfig->setValue(variable, value, section))
                hostConfig->addValue(variable, value, section);
        }
        else
        {
            coConfigEntry *hostConfig = hostConfigs[activeHostname];
            if (hostConfig == 0)
            {
                createHostConfig(targetHost);
            }
            if (!hostConfigs[targetHost]->setValue(variable, value, section))
                hostConfigs[targetHost]->addValue(variable, value, section);
        }
        break;

    default:
        COCONFIGLOG("coConfig::setValue err: no such scope " << scope);
    }

    // oldValue = getValue(variable, section);
    // cerr << "coConfig::setValue info: vrfy " << section << " - " << variable
    //      << " = " << oldValue << " in scope " << oldValue.getConfigScope() << endl;
}

bool coConfigRoot::deleteValue(const QString &variable, const QString &section,
                               const QString &targetHost)
{

    if (isReadOnly())
    {
        COCONFIGDBG("coConfigRoot::deleteValue warn: not deleting value in read only root");
        return false;
    }

    if (targetHost == QString::null)
    {
        if (globalConfig)
            return globalConfig->deleteValue(variable, section);
    }
    else
    {
        coConfigEntry *hostConfig = hostConfigs[targetHost];
        if (hostConfig)
            return hostConfig->deleteValue(variable, section);
    }

    return false;
}

bool coConfigRoot::deleteSection(const QString &section, const QString &targetHost)
{
    if (isReadOnly())
    {
        COCONFIGDBG("coConfigRoot::deleteSection warn: not deleting section in read only root");
        return false;
    }

    if (targetHost == QString::null)
    {
        if (globalConfig)
            return globalConfig->deleteSection(section);
    }
    else
    {
        coConfigEntry *hostConfig = hostConfigs[targetHost];
        if (hostConfig)
            return hostConfig->deleteSection(section);
    }

    return false;
}

bool coConfigRoot::save(const QString &filename) const
{

    if (isReadOnly())
    {
        COCONFIGDBG("coConfigRoot::save warn: not saving read only config");
        return true;
    }

    QString saveToFile = filename;
    if (filename == QString::null)
    {
        saveToFile = this->filename;
    }

    if (saveToFile == QString::null)
    {
        COCONFIGDBG("coConfigRoot::save info: no filename given, skipping save");
        return true;
    }

    xercesc::DOMImplementation *impl = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core"));

    xercesc::DOMDocument *document = impl->createDocument(0, xercesc::XMLString::transcode("COCONFIG"), 0);

    xercesc::DOMElement *rootElement = document->getDocumentElement();

#if XERCES_VERSION_MAJOR < 3
    document->setVersion(xercesc::XMLString::transcode("1.0"));
    document->setStandalone(true);
    document->setEncoding(xercesc::XMLString::transcode("utf8"));
#else
    document->setXmlVersion(xercesc::XMLString::transcode("1.0"));
    document->setXmlStandalone(true);
#endif

    //    xercesc::DOMProcessingInstruction * processingInstruction =   document->createProcessingInstruction(xercesc::XMLString::transcode("xml"),
    //           xercesc::XMLString::transcode("version=\"1.0\""));
    //     rootElement->appendChild(processingInstruction);

    COCONFIGDBG("coConfigRoot::save info: writing config " << configName << " to " << saveToFile);

    if (coConfigXercesEntry *gc = dynamic_cast<coConfigXercesEntry *>(globalConfig))
    {
        COCONFIGDBG("coConfigRoot::save info: saving global config ");
        rootElement->appendChild(document->createTextNode(xercesc::XMLString::transcode("\n")));
        rootElement->appendChild(gc->storeToDom(*document));
        rootElement->appendChild(document->createTextNode(xercesc::XMLString::transcode("\n")));
    }
    else
    {
        COCONFIGDBG("coConfigRoot::save info: skipping save of global config");
    }

    QList<QString> keys = hostConfigs.keys();
    for (QList<QString>::const_iterator i = keys.begin(); i != keys.end(); ++i)
    {
        coConfigEntry *config = hostConfigs[*i];
        if (coConfigXercesEntry *c = dynamic_cast<coConfigXercesEntry *>(config))
        {
            COCONFIGDBG("coConfigRoot::save info: saving host config for " << *i);
            QString confignameAttr = config->getValue("configname", QString());
            // configname attr is only intern. No need to write it to file.
            c->deleteValue("configname", QString());
            rootElement->appendChild(c->storeToDom(*document));
            // set attr again (NOTE shall it be saveToFile), maybe the libary is still in use
            c->setValue("configname", confignameAttr, QString());
            rootElement->appendChild(document->createTextNode(xercesc::XMLString::transcode("\n")));
        }
        else
            COCONFIGDBG("coConfigRoot::save info: null entry for " << *i << "!");
    }

#if XERCES_VERSION_MAJOR < 3
    xercesc::DOMWriter *writer = impl->createDOMWriter();

    // "discard-default-content" "validation" "format-pretty-print"
    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(saveToFile.toLatin1());
    bool written = writer->writeNode(xmlTarget, *rootElement);
    if (!written)
        COCONFIGLOG("coConfigRoot::save info: Could not open file for writing !");

    delete writer;
    delete xmlTarget;
#else

    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();

    writer->getDomConfig();
    //xercesc::DOMConfiguration* dc = writer->getDomConfig();

    //dc->setParameter(xercesc::XMLUni::fgDOMErrorHandler,errorHandler);

    //dc->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent,true);

    xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    theOutput->setEncoding(xercesc::XMLString::transcode("utf8"));

    bool written = writer->writeToURI(rootElement, xercesc::XMLString::transcode(saveToFile.toLatin1()));
    if (!written)
        COCONFIGLOG("coConfigRoot::save info: Could not open file for writing !");
    delete writer;

#endif

    return written;
}

QString coConfigRoot::findConfigFile(const QString &filename, bool preferGlobal)
{

    QFile configfile;

    if (preferGlobal)
    {

        findGlobalConfig(filename, configfile);
        if (!configfile.exists())
            findLocalConfig(filename, configfile);
    }
    else
    {

        findLocalConfig(filename, configfile);
        if (!configfile.exists())
            findGlobalConfig(filename, configfile);
    }

    return configfile.fileName();
}

void coConfigRoot::findLocalConfig(const QString &filename, QFile &configfile)
{

    QString localConfigPath = coConfigDefaultPaths::getDefaultLocalConfigFilePath();
    QDir d;

    d = QDir(localConfigPath);
    if (d.exists())
        configfile.setFileName(d.absoluteFilePath(filename));
}

void coConfigRoot::findGlobalConfig(const QString &filename, QFile &configfile)
{
    QStringList pathEntries = coConfigDefaultPaths::getSearchPath();

    for (QStringList::const_iterator pathEntry = pathEntries.begin(); pathEntry != pathEntries.end(); ++pathEntry)
    {
        QDir d = QDir(*pathEntry + QDir::separator() + "config");
        configfile.setFileName(d.absoluteFilePath(filename));
        COCONFIGDBG("coConfigRoot::findConfigFile info: trying " << configfile.fileName());
        if (!configfile.exists())
        {
            d = QDir(*pathEntry);
            configfile.setFileName(d.absoluteFilePath(filename));
            COCONFIGDBG("coConfigRoot::findConfigFile info: trying " << configfile.fileName());
        }
        if (configfile.exists())
            break;
    }
}

xercesc::DOMNode *coConfigXercesRoot::loadFile(const QString &filename)
{
    //create Parser, get Schema file and return element
    xercesc::DOMElement *globalConfigElement = 0;
    QString schemaFile;
    if (!QFileInfo(filename).isFile())
    {
        COCONFIGDBG("coConfigRoot::loadFile err: non existent filename: " << filename);
        return globalConfigElement;
    }
    else
    {
        COCONFIGDBG("coConfigRoot::loadFile info: loading " << filename);
        xercesc::XercesDOMParser *parser = 0;
        coConfigRootErrorHandler handler;

        xercesc::XMLGrammarPool *grammarPool;
        QString externalSchemaFile = getenv("COCONFIG_SCHEMA");
        if (QFileInfo(externalSchemaFile).isFile())
        {
            schemaFile = externalSchemaFile;
            COCONFIGDBG("coConfigRoot::loadFile info externalSchemaFile: " << schemaFile.toLatin1());
        }

        try
        {
// Getting a XSmodel from the SchemaGrammar, needed to process annotations

#if XERCES_VERSION_MAJOR < 3
            grammarPool = new xercesc::XMLGrammarPoolImpl(xercesc::XMLPlatformUtils::fgMemoryManager);
            parser = new xercesc::XercesDOMParser(0, xercesc::XMLPlatformUtils::fgMemoryManager, grammarPool);
#else
            grammarPool = NULL;
            parser = new xercesc::XercesDOMParser(0, xercesc::XMLPlatformUtils::fgMemoryManager, grammarPool);
#endif
            parser->setExternalNoNamespaceSchemaLocation(reinterpret_cast<const XMLCh *>(schemaFile.utf16()));
            parser->setDoNamespaces(true); // n o change
            parser->useCachedGrammarInParse(true);
            //parser->setDoValidation( true );
            parser->setIncludeIgnorableWhitespace(false);
            parser->setErrorHandler(&handler);
            if (schemaFile.length() > 0)
            {
                parser->setDoSchema(true);
                parser->setValidationSchemaFullChecking(true);
                // change to Validate possible Never, Auto, Always
                parser->setValidationScheme(xercesc::XercesDOMParser::Val_Always);
                parser->loadGrammar(reinterpret_cast<const XMLCh *>(schemaFile.utf16()), xercesc::Grammar::SchemaGrammarType, true);
            }
            //  Try parsing fallback schema file, if there was an error
            // ------------ Disabled for now -----------------
            //          if (handler.getSawErrors())
            //          {
            //             handler.resetErrors();
            //             QString fallbackSchema =  getenv("COVISEDIR") ;
            //             fallbackSchema.append("/src/kernel/config/coEditor/schema.xsd");
            //             COCONFIGDBG ("coConfigRoot::loadFile err: Parsing '"
            //                   << schemaFile <<"' failed. Trying fallback:" << fallbackSchema );
            //             parser->loadGrammar (fallbackSchema.utf16(), xercesc::Grammar::SchemaGrammarType, true);
            //          }

            if (handler.getSawErrors())
            {
                if (schemaFile == "")
                {
                    COCONFIGDBG("coConfigRoot::loadFile warn: schema not set, please set the environment variable COCONFIG_SCHEMA ... not validating");
                }
                else
                {
                    COCONFIGLOG("coConfigRoot::loadFile warn: opening schema '" << schemaFile << "' failed, not validating");
                }
                handler.resetErrors();
                parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);
                parser->setDoSchema(false);
                parser->setValidationSchemaFullChecking(false);
                parser->useCachedGrammarInParse(false);
            }

// try parsing the file

// work around for UNC file names
#ifdef WIN32
            char *fileName = new char[filename.length() + 1];
            strcpy(fileName, filename.toLatin1());
            if (fileName[0] == '/' && fileName[1] == '/')
            {
                fileName[1] = '\\';
                fileName[0] = '\\';
            }
            parser->parse(fileName);
#else
            parser->parse(filename.toLatin1());
#endif
        }
        catch (const xercesc::XMLException &toCatch)
        {
            QString message = QString::fromUtf16(reinterpret_cast<const ushort *>(toCatch.getMessage()));
            COCONFIGLOG("coConfigRoot::loadFile err: xmlparse failed " << message);
        }
        catch (const xercesc::DOMException &toCatch)
        {
            QString message = QString::fromUtf16(reinterpret_cast<const ushort *>(toCatch.getMessage()));
            COCONFIGLOG("coConfigRoot::loadFile err: xmlparse failed " << message);
        }
        catch (...)
        {
            COCONFIGLOG("coConfigRoot::loadFile err: xmlparse failed - unknown exception");
        }

        if (parser == 0)
        {
            COCONFIGLOG("coConfigRoot::loadFile err: failed to create parser");
            return 0;
        }

        xercesc::DOMDocument *xmlDoc = parser->getDocument();
        globalConfigElement = xmlDoc->getDocumentElement();

        if (globalConfigElement == 0 || QString::fromUtf16(reinterpret_cast<const ushort *>(globalConfigElement->getNodeName())) != "COCONFIG")
        {
            COCONFIGLOG("coConfigRoot::loadFile err: COCONFIG not found in " << filename);
        }
        else
        {
            COCONFIGDBG("coConfigRoot::loadFile warn: Parse of Schema failed "
                        << "\n");
        }
    }
    return globalConfigElement;
}

void coConfigRoot::setReadOnly(bool ro)
{
    readOnly = ro;
}

bool coConfigRoot::isReadOnly() const
{
    return readOnly;
}

void coConfigRoot::clear()
{
    {
        QList<QString> keys = hostConfigs.keys();
        for (QList<QString>::iterator key = keys.begin(); key != keys.end(); ++key)
        {
            delete hostConfigs.take(*key);
        }
    }
    {
        QList<QString> keys = clusterConfigs.keys();
        for (QList<QString>::iterator key = keys.begin(); key != keys.end(); ++key)
        {
            delete clusterConfigs.take(*key);
        }
    }

    delete globalConfig;

    globalConfig = 0;
    clusterConfig = 0;
    hostConfig = 0;
}

void coConfigXercesRoot::createGlobalConfig()
{
    COCONFIGDBG("coConfigRoot::createGlobalConfig info: creating global config");

    xercesc::DOMImplementation *implLoad = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core"));
    xercesc::DOMDocument *document = implLoad->createDocument(0, xercesc::XMLString::transcode("COCONFIG"), 0);
    xercesc::DOMElement *rootNode = document->getDocumentElement();
    xercesc::DOMElement *globalElement = document->createElement(xercesc::XMLString::transcode("GLOBAL"));
    rootNode->appendChild(globalElement);
    setContentsFromDom(document->getDocumentElement());
}

void coConfigXercesRoot::createHostConfig(const QString &hostname)
{
    COCONFIGDBG("coConfigRoot::createHostConfig info: creating local config for host " << hostname);

    xercesc::DOMImplementation *implLoad = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core"));
    xercesc::DOMDocument *document = implLoad->createDocument(0, xercesc::XMLString::transcode("COCONFIG"), 0);
    xercesc::DOMElement *rootNode = document->getDocumentElement();
    xercesc::DOMElement *localElement = document->createElement(xercesc::XMLString::transcode("LOCAL"));
    localElement->setAttribute(xercesc::XMLString::transcode("host"), reinterpret_cast<const XMLCh *>(hostname.toLower().utf16()));
    rootNode->appendChild(localElement);
    setContentsFromDom(document->getDocumentElement());
}

void coConfigXercesRoot::createClusterConfig(const QString &hostname)
{
    COCONFIGDBG("coConfigRoot::createClusterConfig info: creating cluster config for master " << hostname);

    xercesc::DOMImplementation *implLoad = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core"));
    xercesc::DOMDocument *document = implLoad->createDocument(0, xercesc::XMLString::transcode("COCONFIG"), 0);
    xercesc::DOMElement *rootNode = document->getDocumentElement();
    xercesc::DOMElement *localElement = document->createElement(xercesc::XMLString::transcode("CLUSTER"));
    localElement->setAttribute(xercesc::XMLString::transcode("master"), reinterpret_cast<const XMLCh *>(hostname.toLower().utf16()));
    rootNode->appendChild(localElement);
    setContentsFromDom(document->getDocumentElement());
}

QStringList coConfigRoot::getHosts()
{
    return hostConfigs.keys();
}

coConfigEntry *coConfigRoot::getConfigForHost(const QString &hostname)
{
    return hostConfigs[hostname.toLower()];
}

coConfigEntry *coConfigRoot::getConfigForCluster(const QString &masterhost)
{
    return clusterConfigs[masterhost.toLower()];
}

coConfigRoot *coConfigXercesRoot::clone() const
{
    return new coConfigXercesRoot(this);
}

void coConfigXercesRoot::merge(const coConfigRoot *with)
{
    this->globalConfig->merge(with->globalConfig);
    for (QHash<QString, coConfigEntry *>::const_iterator entry = with->clusterConfigs.begin(); entry != with->clusterConfigs.end(); ++entry)
    {
        if (this->clusterConfigs.contains(entry.key()))
        {
            if (this->clusterConfigs[entry.key()])
                this->clusterConfigs[entry.key()]->merge(entry.value());
        }
        else
        {
            this->clusterConfigs[entry.key()] = entry.value()->clone();
            this->masternames.append(entry.key());
        }
    }
    for (QHash<QString, coConfigEntry *>::const_iterator entry = with->hostConfigs.begin(); entry != with->hostConfigs.end(); ++entry)
    {
        if (this->hostConfigs.contains(entry.key()))
        {
            if (this->hostConfigs[entry.key()])
                this->hostConfigs[entry.key()]->merge(entry.value());
        }
        else
        {
            this->hostConfigs[entry.key()] = entry.value()->clone();
            this->hostnames.append(entry.key());
        }
    }
}
