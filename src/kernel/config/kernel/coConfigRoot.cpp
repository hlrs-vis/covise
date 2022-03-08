/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coConfigRootErrorHandler.h"
#include "coConfigTools.h"
#include "coConfigXercesConverter.h"
#include "coConfigXercesEntry.h"
#include "coConfigXercesRoot.h"

#include <codecvt>
#include <config/coConfigConstants.h>
#include <config/coConfigLog.h>
#include <config/coConfigRoot.h>
#include <iostream>
#include <util/string_util.h>
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
#include <boost/filesystem.hpp>
#include <array>
using namespace covise;

coConfigRoot *coConfigRoot::createNew(const std::string &name, const std::string &filename,
                                      bool create, coConfigGroup *group)
{
    return new coConfigXercesRoot(name, filename, create, group);
}

coConfigRoot::coConfigRoot(const std::string &name, const std::string &filename,
                           bool create, coConfigGroup *group)
    : globalConfig(0), clusterConfig(0), hostConfig(0)
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
    activeHostname = source->activeHostname;
    activeCluster = source->activeCluster;
    hostnames = source->hostnames;
    masternames = source->masternames;

    configName = source->configName;

    globalConfig = source->globalConfig->clone();

    for (auto entry : source->hostConfigs)
    {
        if (entry.second)
            hostConfigs.insert({entry.first, entry.second->clone()});
        else
            hostConfigs.insert({entry.first, nullptr});
    }

    if (hostConfigs.find(activeHostname) != hostConfigs.end())
        hostConfig = hostConfigs[activeHostname];
    else
        hostConfig = 0;

    for (auto entry : source->clusterConfigs)
    {
        if (entry.second)
            clusterConfigs.insert({entry.first, entry.second->clone()});
        else
            clusterConfigs.insert({entry.first, nullptr});
    }

    if (clusterConfigs.find(activeCluster) != clusterConfigs.end())
        clusterConfig = clusterConfigs[activeCluster];
    else
        clusterConfig = nullptr;

    create = source->create;
    readOnly = source->readOnly;
}

coConfigXercesRoot::coConfigXercesRoot(const std::string &name, const std::string &filename,
                                       bool create, coConfigGroup *group)
    : coConfigRoot(name, filename, create, group)
{
    load(create);
}

coConfigXercesRoot::coConfigXercesRoot(const xercesc::DOMNode *node, const std::string &name,
                                       const std::string &filename, coConfigGroup *group)
    : coConfigRoot(name, filename, false, group)
{
    xercesc::DOMNode *globalConfigNode = 0;
    xercesc::DOMNodeList *nodeList = node->getChildNodes();
    for (int it = 0; it < nodeList->getLength(); ++it)
    {
        auto nodeName = xercescToStdString(nodeList->item(it)->getNodeName());

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

const std::set<std::string> &coConfigRoot::getHostnameList() const
{
    return hostnames;
}

const std::string &coConfigRoot::getActiveHost() const
{
    return activeHostname;
}

bool coConfigRoot::setActiveHost(const std::string &host)
{
    if (hostnames.find(toLower(host)) != hostnames.end())
    {
        // cerr << "coConfigRoot::setActiveHost info: setting active host "
        //      << host << endl;
        activeHostname = toLower(host);

        hostConfig = hostConfigs[activeHostname];
        if (!hostConfig)
        {
            hostConfigs[activeHostname] = hostConfig = nullptr;
        }

        return true;
    }
    else
    {

        COCONFIGLOG("coConfigRoot::setActiveHost warn: could not set active host " << toLower(host) << ", this: " << this);
        return false;
    }
}

const std::set<std::string> &coConfigRoot::getClusterList() const
{
    return masternames;
}

const std::string &coConfigRoot::getActiveCluster() const
{
    return activeCluster;
}

bool coConfigRoot::setActiveCluster(const std::string &master)
{
    if (masternames.find(toLower(master)) != masternames.end() || master.empty())
    {
        // cerr << "coConfigRoot::setActiveCluster info: setting active master "
        //      << master << endl;
        activeCluster = toLower(master);

        if (master.empty())
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

        COCONFIGDBG("coConfigRoot::setActiveCluster warn: could not set active cluster " << toLower(master));
        return false;
    }
}

const std::string &coConfigRoot::getConfigName() const
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
    auto configfile = findConfigFile(filename);
    if (boost::filesystem::exists(configfile))
    {
        xercesc::DOMNode *globalConfigNode = loadFile(configfile);
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

    hostnames.insert(coConfigConstants::getHostname());
    if (!coConfigConstants::getMaster().empty())
    {
        masternames.insert(coConfigConstants::getMaster());
    }

    if (hostnames.find(activeHostname) == hostnames.end())
    {
        activeHostname = coConfigConstants::getHostname();
    }
    if (masternames.find(activeCluster) == masternames.end())
    {
        activeCluster = coConfigConstants::getMaster();
    }

    setActiveCluster(activeCluster);
    setActiveHost(activeHostname);
}

void readHosts(const std::string &hostTemp, std::set<std::string> &hostnames, std::map<std::string, coConfigEntry *> &configs, const std::string &configName, xercesc::DOMElement *node, coConfigEntry *globalConfig)
{
    auto hosts = split(hostTemp, ',', true);
    for (const auto &host : hosts)
    {
        std::string hostname = toLower(strip(host));
        // cerr << "coConfigRoot::setContentsFromDom info: adding for host " << hostname.latin1() << endl;
        if (hostnames.insert(hostname).second)
        {
            configs.insert({hostname, coConfigXercesEntry::restoreFromDom(node, configName)});
        }
        else
        {
            auto &hostConfig = configs[hostname];
            if (!hostConfig)
                hostConfig = coConfigXercesEntry::restoreFromDom(node, configName);
            else
                hostConfig->merge(coConfigXercesEntry::restoreFromDom(node, configName));

            // Temporary attributes cleanup
            if (globalConfig)
            {
                globalConfig->deleteValue("scope", std::string());
                globalConfig->deleteValue("configname", std::string());
            }
        }
    }
}

void coConfigXercesRoot::setContentsFromDom(const xercesc::DOMNode *node)
{
    // COCONFIGLOG("coConfigRoot::setContentsFromDom info: creating tree from " << configName);

    if (node)
    {
        xercesc::DOMNodeList *nodeList = node->getChildNodes();
        for (int i = 0; i < nodeList->getLength(); ++i)
        {

            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;

            auto nodeName = xercescToStdString(node->getNodeName());

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
                    globalConfigNode->setAttribute(stringToXexcesc("scope").get(),
                                                   stringToXexcesc("global").get());
                    globalConfigNode->setAttribute(stringToXexcesc("configname").get(),
                                                   stringToXexcesc(configName).get());

                    if (globalConfig == 0)
                    {
                        globalConfig = coConfigXercesEntry::restoreFromDom(globalConfigNode, configName);
                    }
                    else
                    {
                        globalConfig->merge(coConfigXercesEntry::restoreFromDom(node, configName));
                    }
                    // Temporary attributes cleanup
                    globalConfig->deleteValue("scope", "");
                    globalConfig->deleteValue("configname", "");
                }
            }
            else if (nodeName == "LOCAL")
            {

                // Temporary attributes
                node->setAttribute(stringToXexcesc("scope").get(),
                                   stringToXexcesc("host").get());

                node->setAttribute(stringToXexcesc("configname").get(),
                                   stringToXexcesc(configName).get());

                std::string hostTemp = xercescToStdString(node->getAttribute(stringToXexcesc("HOST").get()));
                readHosts(hostTemp, hostnames, hostConfigs, configName, node, globalConfig);
            }
            else if (nodeName == "CLUSTER")
            {

                // Temporary attributes
                node->setAttribute(stringToXexcesc("scope").get(),
                                   stringToXexcesc("cluster").get());

                node->setAttribute(stringToXexcesc("configname").get(),
                                   stringToXexcesc(configName).get());

                std::string hostTemp = xercescToStdString(node->getAttribute(stringToXexcesc("MASTER").get()));
                readHosts(hostTemp, masternames, clusterConfigs, configName, node, globalConfig);
            }
            else if (nodeName == "INCLUDE" || nodeName == "TRYINCLUDE")
            {
                std::map<std::string, std::string> attributes;
                for (const auto &attributeName : coConfigTools::attributeNames)
                {
                    auto val = xercescToStdString(node->getAttribute(stringToXexcesc(attributeName).get()));
                    if (!val.empty())
                        attributes.insert({attributeName, val});
                }
                if (coConfigTools::matchingAttributes(attributes))
                {
                    std::string filename = strip(xercescToStdString(node->getFirstChild()->getNodeValue()));
                    if (included.find(filename) == included.end())
                    {
                        COCONFIGDBG("coConfigRoot::setContentsFromDom info: INCLUDE:  filename: " << filename);

                        xercesc::DOMNode *includeNode = 0;

                        // try in current directory first
                        boost::filesystem::path p{this->filename};
                        std::string localPath = (p.parent_path() / filename).string();
                        if (boost::filesystem::exists(localPath))
                        {
                            includeNode = loadFile(localPath);
                        }
                        else if (!node->getAttribute(stringToXexcesc("global").get()) || xercescToStdString(node->getAttribute(stringToXexcesc("global").get())) == "0")
                            includeNode = loadFile(findConfigFile(filename, false));
                        else
                            includeNode = loadFile(findConfigFile(filename, true));

                        if (!includeNode)
                        {
                            if (nodeName == "TRYINCLUDE")
                            {
                                COCONFIGLOG("coConfigRoot::setContentsFromDom info: could not open tryinclude file " << filename);
                            }
                            else
                            {
                                COCONFIGLOG("coConfigRoot::setContentsFromDom error: could not open include file " << filename);
                                exit(1);
                            }
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

    if (!coConfigConstants::getMaster().empty())
    {
        masternames.insert(coConfigConstants::getMaster());
    }

    if (masternames.find(activeCluster) == masternames.end())
    {
        activeCluster = coConfigConstants::getMaster();
    }
    setActiveCluster(activeCluster);
    //
    // Set active host
    //

    hostnames.insert(coConfigConstants::getHostname());

    if (hostnames.find(activeHostname) == hostnames.end())
    {
        activeHostname = coConfigConstants::getHostname();
    }
    setActiveHost(activeHostname);
}

coConfigEntryStringList coConfigRoot::getScopeList(const std::string &section, const std::string &variableName) const
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

    if (variableName.empty())
    {
        return merged;
    }
    else
    {
        return merged.filter(std::regex("^" + variableName + ":.*"));
    }
}

coConfigEntryStringList coConfigRoot::getVariableList(const std::string &section) const
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

coConfigEntryString coConfigRoot::getValue(const std::string &variable,
                                           const std::string &section,
                                           const std::string &defaultValue) const
{

    coConfigEntryString value = getValue(variable, section);
    if (value.entry.empty())
        return coConfigEntryString{defaultValue, configName};
    return value;
}

coConfigEntryString coConfigRoot::getValue(const std::string &simpleVariable) const
{
    return getValue("value", simpleVariable);
}

coConfigEntryString coConfigRoot::getValue(const std::string &variable,
                                           const std::string &section) const
{

    if (hostConfig)
    {
        coConfigEntryString localItem = hostConfig->getValue(variable, section);

        if (!localItem.entry.empty())
        {
            localItem.configName = configName;
            return localItem;
        }
    }

    if (clusterConfig)
    {
        coConfigEntryString localItem = clusterConfig->getValue(variable, section);

        if (!localItem.entry.empty())
        {
            localItem.configName = configName;
            return localItem;
        }
    }

    if (globalConfig)
    {
        coConfigEntryString globalItem = globalConfig->getValue(variable, section);

        if (!globalItem.entry.empty())
        {
            globalItem.configName = configName;
            return globalItem;
        }
    }

    // cerr << "coConfig::getValue info: " << section << "."
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

bool coConfigRoot::isOn(const std::string &simpleVariable, bool defaultValue) const
{
    return isOn("value", simpleVariable, defaultValue);
}

bool coConfigRoot::isOn(const std::string &simpleVariable) const
{
    return isOn("value", simpleVariable);
}

bool coConfigRoot::isOn(const std::string &variable, const std::string &section,
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

void coConfigRoot::setValue(const std::string &variable, const std::string &value,
                            const std::string &section,
                            const std::string &targetHost, bool move)
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

    // coConfigEntryString oldValue = getValue(variable, section);

    coConfigConstants::ConfigScope scope = coConfigConstants::Default;

    // if (oldValue.isNull()) {
    //   scope = coConfigConstants::Global;
    // } else {
    //   scope = oldValue.getConfigScope();
    // }

    if (targetHost == std::string())
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
        if (targetHost == std::string())
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
        if ((toLower(targetHost) == activeHostname) || (targetHost == std::string()))
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

bool coConfigRoot::deleteValue(const std::string &variable, const std::string &section,
                               const std::string &targetHost)
{

    if (isReadOnly())
    {
        COCONFIGDBG("coConfigRoot::deleteValue warn: not deleting value in read only root");
        return false;
    }

    if (targetHost == std::string())
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

bool coConfigRoot::deleteSection(const std::string &section, const std::string &targetHost)
{
    if (isReadOnly())
    {
        COCONFIGDBG("coConfigRoot::deleteSection warn: not deleting section in read only root");
        return false;
    }

    if (targetHost == std::string())
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

bool coConfigRoot::save(const std::string &filename) const
{

    if (isReadOnly())
    {
        COCONFIGDBG("coConfigRoot::save warn: not saving read only config");
        return true;
    }

    std::string saveToFile = filename;
    if (filename == std::string())
    {
        saveToFile = this->filename;
    }

    if (saveToFile == std::string())
    {
        COCONFIGDBG("coConfigRoot::save info: no filename given, skipping save");
        return true;
    }

    xercesc::DOMImplementation *impl = xercesc::DOMImplementationRegistry::getDOMImplementation(stringToXexcesc("Core").get());

    xercesc::DOMDocument *document = impl->createDocument(0, stringToXexcesc("COCONFIG").get(), 0);

    xercesc::DOMElement *rootElement = document->getDocumentElement();

#if XERCES_VERSION_MAJOR < 3
    document->setVersion(stringToXexcesc("1.0").get());
    document->setStandalone(true);
    document->setEncoding(stringToXexcesc("utf8").get());
#else
    document->setXmlVersion(stringToXexcesc("1.0").get());
    document->setXmlStandalone(true);
#endif

    //    xercesc::DOMProcessingInstruction * processingInstruction =   document->createProcessingInstruction(stringToXexcesc("xml"),
    //           stringToXexcesc("version=\"1.0\""));
    //     rootElement->appendChild(processingInstruction);

    COCONFIGDBG("coConfigRoot::save info: writing config " << configName << " to " << saveToFile);

    if (coConfigXercesEntry *gc = dynamic_cast<coConfigXercesEntry *>(globalConfig))
    {
        COCONFIGDBG("coConfigRoot::save info: saving global config ");
        rootElement->appendChild(document->createTextNode(stringToXexcesc("\n").get()));
        rootElement->appendChild(gc->storeToDom(*document));
        rootElement->appendChild(document->createTextNode(stringToXexcesc("\n").get()));
    }
    else
    {
        COCONFIGDBG("coConfigRoot::save info: skipping save of global config");
    }

    for (const auto &hostConfig : hostConfigs)
    {
        coConfigEntry *config = hostConfig.second;
        if (coConfigXercesEntry *c = dynamic_cast<coConfigXercesEntry *>(config))
        {
            COCONFIGDBG("coConfigRoot::save info: saving host config for " << hostConfig.first);
            std::string confignameAttr = config->getValue("configname", std::string()).entry;
            // configname attr is only intern. No need to write it to file.
            c->deleteValue("configname", std::string());
            rootElement->appendChild(c->storeToDom(*document));
            // set attr again (NOTE shall it be saveToFile), maybe the libary is still in use
            c->setValue("configname", confignameAttr, std::string());
            rootElement->appendChild(document->createTextNode(stringToXexcesc("\n").get()));
        }
        else
            COCONFIGDBG("coConfigRoot::save info: null entry for " << hostConfig.first << "!");
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
    // xercesc::DOMConfiguration* dc = writer->getDomConfig();

    // dc->setParameter(xercesc::XMLUni::fgDOMErrorHandler,errorHandler);

    // dc->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent,true);

    xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    theOutput->setEncoding(stringToXexcesc("utf8").get());

    bool written = writer->writeToURI(rootElement, stringToXexcesc(saveToFile).get());
    if (!written)
        COCONFIGLOG("coConfigRoot::save info: Could not open file for writing !");
    delete writer;

#endif

    return written;
}

std::string coConfigRoot::findConfigFile(const std::string &filename, bool preferGlobal)
{

    boost::filesystem::path configfile;
    if (preferGlobal)
    {

        auto configfile = findGlobalConfig(filename);
        if (!boost::filesystem::exists(configfile))
            configfile = findLocalConfig(filename);
    }
    else
    {

        configfile = findLocalConfig(filename);
        if (!boost::filesystem::exists(configfile))
            configfile = findGlobalConfig(filename);
    }

    return configfile.string();
}

boost::filesystem::path coConfigRoot::findLocalConfig(const std::string &filename)
{
    std::string localConfigPath = coConfigDefaultPaths::getDefaultLocalConfigFilePath();
    boost::filesystem::path d{localConfigPath};
    if (boost::filesystem::exists(d))
        return boost::filesystem::path{boost::filesystem::absolute(filename, localConfigPath)};
    return boost::filesystem::path{};
}

boost::filesystem::path coConfigRoot::findGlobalConfig(const std::string &filename)
{
    std::set<std::string> pathEntries = coConfigDefaultPaths::getSearchPath();

    for (const auto &pathEntry : pathEntries)
    {
        boost::filesystem::path d{pathEntry + pathSeparator + "config" + pathSeparator + filename};
        COCONFIGDBG("coConfigRoot::findConfigFile info: trying " << d.string());
        if (!boost::filesystem::exists(d))
        {
            d = boost::filesystem::path{boost::filesystem::absolute(filename)};
            COCONFIGDBG("coConfigRoot::findConfigFile info: trying " << d.string());
        }
        if (boost::filesystem::exists(d))
            return d;
    }
    return boost::filesystem::path{};
}

xercesc::DOMNode *coConfigXercesRoot::loadFile(const std::string &filename)
{
    // create Parser, get Schema file and return element
    xercesc::DOMElement *globalConfigElement = 0;
    std::string schemaFile;
    boost::filesystem::path p;
    boost::filesystem::is_regular_file(filename);
    if (!boost::filesystem::is_regular_file(filename))
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
        auto externalSchemaFile = getenv("COCONFIG_SCHEMA");
        if (externalSchemaFile && boost::filesystem::is_regular_file(externalSchemaFile))
        {
            schemaFile = externalSchemaFile;
            COCONFIGDBG("coConfigRoot::loadFile info externalSchemaFile: " << schemaFile);
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
            parser->setExternalNoNamespaceSchemaLocation(stringToXexcesc(schemaFile).get());
            parser->setDoNamespaces(true); // n o change
            parser->useCachedGrammarInParse(true);
            // parser->setDoValidation( true );
            parser->setIncludeIgnorableWhitespace(false);
            parser->setErrorHandler(&handler);
            if (schemaFile.length() > 0)
            {
                parser->setDoSchema(true);
                parser->setValidationSchemaFullChecking(true);
                // change to Validate possible Never, Auto, Always
                parser->setValidationScheme(xercesc::XercesDOMParser::Val_Always);
                parser->loadGrammar(stringToXexcesc(schemaFile).get(), xercesc::Grammar::SchemaGrammarType, true);
            }
            //  Try parsing fallback schema file, if there was an error
            // ------------ Disabled for now -----------------
            //          if (handler.getSawErrors())
            //          {
            //             handler.resetErrors();
            //             std::string fallbackSchema =  getenv("COVISEDIR") ;
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
            auto fileName = filename;
            if (fileName[0] == '/' && fileName[1] == '/')
            {
                fileName[1] = '\\';
                fileName[0] = '\\';
            }
            parser->parse(fileName.c_str());
#else
            parser->parse(stringToXexcesc(filename).get());
#endif
        }
        catch (const xercesc::XMLException &toCatch)
        {
            std::string message = xercescToStdString(toCatch.getMessage());
            COCONFIGLOG("coConfigRoot::loadFile err: xmlparse failed " << message);
        }
        catch (const xercesc::DOMException &toCatch)
        {
            std::string message = xercescToStdString(toCatch.getMessage());
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

        if (globalConfigElement == 0 || xercescToStdString(globalConfigElement->getNodeName()) != "COCONFIG")
        {
            COCONFIGLOG("coConfigRoot::loadFile err: COCONFIG not found in " << filename);
        }
        else
        {
#if 0 // this will happen all the time
            COCONFIGDBG("coConfigRoot::loadFile warn: Parse of Schema failed "
                        << "\n");
#endif
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
    for (const auto &hostConfig : hostConfigs)
        delete hostConfig.second;
    hostConfigs.clear();

    for (const auto &clusterConfig : clusterConfigs)
        delete clusterConfig.second;
    clusterConfigs.clear();

    delete globalConfig;

    globalConfig = nullptr;
    clusterConfig = nullptr;
    hostConfig = nullptr;
}

void coConfigXercesRoot::createGlobalConfig()
{
    COCONFIGDBG("coConfigRoot::createGlobalConfig info: creating global config");

    xercesc::DOMImplementation *implLoad = xercesc::DOMImplementationRegistry::getDOMImplementation(stringToXexcesc("Core").get());
    xercesc::DOMDocument *document = implLoad->createDocument(0, stringToXexcesc("COCONFIG").get(), nullptr);
    xercesc::DOMElement *rootNode = document->getDocumentElement();
    xercesc::DOMElement *globalElement = document->createElement(stringToXexcesc("GLOBAL").get());
    rootNode->appendChild(globalElement);
    setContentsFromDom(document->getDocumentElement());
}

void coConfigXercesRoot::createHostConfig(const std::string &hostname)
{
    COCONFIGDBG("coConfigRoot::createHostConfig info: creating local config for host " << hostname);

    xercesc::DOMImplementation *implLoad = xercesc::DOMImplementationRegistry::getDOMImplementation(stringToXexcesc("Core").get());
    xercesc::DOMDocument *document = implLoad->createDocument(0, stringToXexcesc("COCONFIG").get(), nullptr);
    xercesc::DOMElement *rootNode = document->getDocumentElement();
    xercesc::DOMElement *localElement = document->createElement(stringToXexcesc("LOCAL").get());
    localElement->setAttribute(stringToXexcesc("HOST").get(), stringToXexcesc(toLower(hostname)).get());
    rootNode->appendChild(localElement);
    setContentsFromDom(document->getDocumentElement());
}

void coConfigXercesRoot::createClusterConfig(const std::string &hostname)
{
    COCONFIGDBG("coConfigRoot::createClusterConfig info: creating cluster config for master " << hostname);

    xercesc::DOMImplementation *implLoad = xercesc::DOMImplementationRegistry::getDOMImplementation(stringToXexcesc("Core").get());
    xercesc::DOMDocument *document = implLoad->createDocument(0, stringToXexcesc("COCONFIG").get(), nullptr);
    xercesc::DOMElement *rootNode = document->getDocumentElement();
    xercesc::DOMElement *localElement = document->createElement(stringToXexcesc("CLUSTER").get());
    localElement->setAttribute(stringToXexcesc("MASTER").get(), stringToXexcesc(toLower(hostname)).get());
    rootNode->appendChild(localElement);
    setContentsFromDom(document->getDocumentElement());
}

std::set<std::string> coConfigRoot::getHosts()
{
    std::set<std::string> hosts;
    for (const auto &host : hostConfigs)
        hosts.insert(host.first);
    return hosts;
}

coConfigEntry *coConfigRoot::getConfigForHost(const std::string &hostname)
{
    return hostConfigs[toLower(hostname)];
}

coConfigEntry *coConfigRoot::getConfigForCluster(const std::string &masterhost)
{
    return clusterConfigs[toLower(masterhost)];
}

coConfigRoot *coConfigXercesRoot::clone() const
{
    return new coConfigXercesRoot(this);
}

void merge(std::map<std::string, coConfigEntry *> &toMerge, const std::map<std::string, coConfigEntry *> &mergeWith, std::set<std::string> &masters)
{
    for (const auto &wihtClusterConfig : mergeWith)
    {
        auto clusterConfig = toMerge.find(wihtClusterConfig.first);
        if (clusterConfig != toMerge.end())
        {
            if (clusterConfig->second)
                clusterConfig->second->merge(wihtClusterConfig.second);
        }
        else
        {
            toMerge.insert({wihtClusterConfig.first, wihtClusterConfig.second->clone()});
            masters.insert(wihtClusterConfig.first);
        }
    }
}

void coConfigXercesRoot::merge(const coConfigRoot *with)
{
    globalConfig->merge(with->globalConfig);
    ::merge(clusterConfigs, with->clusterConfigs, masternames);
    ::merge(hostConfigs, with->hostConfigs, hostnames);
}
