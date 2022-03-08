/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigLog.h>
#include <config/coConfigConstants.h>
#include <config/coConfig.h>
#include <util/string_util.h>
#ifdef _WIN32
#include <winsock2.h>
#include <locale.h>
#else
#include <unistd.h>
#endif

#include <stdlib.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <locale>
#include <codecvt>

using namespace std;
using namespace covise;

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>

coConfigDefaultPaths *coConfigDefaultPaths::instance = 0;
coConfigConstants *coConfigConstants::instance = 0;

#ifdef WIN32

namespace detail
{
    wchar_t t = '/';

    char wcharToChar(wchar_t w)
    {
        char c = ' ';
        wctomb(&c, w);
        return c;
    }
}
char covise::pathSeparator = detail::wcharToChar(detail::t);

#else
char covise::pathSeparator = boost::filesystem::path::preferred_separator;
#endif
coConfigConstants::coConfigConstants()
{
    if (instance == 0)
        initXerces();

    instance = this;

    backend = "covise";

    hostname = std::string();

    rank = -1;
}

coConfigConstants::~coConfigConstants()
{
    if (instance == this)
        instance = 0;
}

const std::set<std::string> &coConfigConstants::getArchList()
{

    if (!instance)
        new coConfigConstants();

    if (instance->archlist.empty())
    {
#ifdef _WIN32
        instance->archlist.insert("windows");
#else
        instance->archlist.insert("unix");

#ifdef __APPLE__
        instance->archlist.insert("mac");
#else
        instance->archlist.insert("x11");
#endif
#endif

        const char *archsuffix = getenv("ARCHSUFFIX");
        if (archsuffix)
        {
            std::string as(archsuffix);
            instance->archlist.insert(as);

            if (boost::algorithm::ends_with(as, "opt"))
            {
                instance->archlist.insert(as.substr(0, as.size() - 3));
            }
        }
    }

    return instance->archlist;
}

const std::string &coConfigConstants::getHostname()
{

    if (!instance)
        new coConfigConstants();

    if (instance->hostname.empty())
    {
        auto h = getenv("COVISE_CONFIG");
        if (h)
        {
            instance->hostname = h;
            COCONFIGDBG_DEFAULT("coConfigConstants::getHostname info: LOCAL hostname is '" + instance->hostname + "' (from COVISE_CONFIG)");
        }
    }

    if (instance->hostname == std::string())
    {
#ifdef _WIN32
        WORD wVersionRequested;
        WSADATA wsaData;
        int err;
        wVersionRequested = MAKEWORD(1, 1);

        err = WSAStartup(wVersionRequested, &wsaData);
#endif

        char hostnameTmp[500];
        if (gethostname(hostnameTmp, 500) < 0)
        {
            COCONFIGLOG("coConfigConstants::getHostname err: unable to resolve hostname, using localhost");
            instance->hostname = "localhost";
        }
        else
        {
            instance->hostname = hostnameTmp;
            std::transform(instance->hostname.begin(), instance->hostname.end(), instance->hostname.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            instance->hostname = instance->hostname.substr(0, instance->hostname.find('.'));
        }

        if (!instance->hostname.empty())
        {
            COCONFIGDBG_DEFAULT("coConfigConstants::getHostname info: LOCAL hostname is '" + instance->hostname + "' (from gethostname)");
        }
    }

    return instance->hostname;
}

void coConfigConstants::setMaster(const std::string &hostname)
{
    if (!instance)
        new coConfigConstants();
    instance->master = toLower(hostname);
    instance->master = instance->master.substr(0, instance->master.find('.'));

    COCONFIGDBG_DEFAULT("coConfigConstants::setMaster info: CLUSTER master is '" + hostname);
    coConfig::getInstance()->setActiveCluster(instance->master);
}

const std::string &coConfigConstants::getMaster()
{

    if (!instance)
        new coConfigConstants();

    if (instance->master == std::string())
    {
        // COCONFIGLOG("coConfigConstants::getMaster info: master hostname is not set");
    }
    return instance->master;
}

void coConfigConstants::setRank(int rank, int shmGroupRoot)
{
    if (!instance)
        new coConfigConstants();
    instance->rank = rank;
    instance->shmGroupRoot = shmGroupRoot;
}

int coConfigConstants::getRank()
{
    if (!instance)
        new coConfigConstants();
    return instance->rank;
}

int coConfigConstants::getShmGroupRootRank()
{
    if (!instance)
        new coConfigConstants();
    return instance->shmGroupRoot;
}

void coConfigConstants::setBackend(const std::string &backend)
{
    if (!instance)
        new coConfigConstants();
    instance->backend = backend;
}

const std::string &coConfigConstants::getBackend()
{
    if (!instance)
        new coConfigConstants();
    return instance->backend;
}

coConfigDefaultPaths::coConfigDefaultPaths()
{
    instance = this;
    setNames();
}

coConfigDefaultPaths::~coConfigDefaultPaths()
{
    instance = 0;
}

/**
 * \brief Used to convert an old COVISE config to the new XML format
 */
const std::string &coConfigDefaultPaths::getDefaultTransformFileName()
{
    return getInstance()->defaultTransform;
}

/**
 * \brief Full filename of the local configuration
 */
const std::string &coConfigDefaultPaths::getDefaultLocalConfigFileName()
{
    return getInstance()->configLocal;
}

/**
 * \brief Full filename of the global configuration
 */
const std::string &coConfigDefaultPaths::getDefaultGlobalConfigFileName()
{
    return getInstance()->configGlobal;
}

/**
 * \brief Full path to the local configuration
 */
const std::string &coConfigDefaultPaths::getDefaultLocalConfigFilePath()
{
    return getInstance()->configLocalPath;
}

/**
 * \brief Full path to the global configuration
 */
const std::string &coConfigDefaultPaths::getDefaultGlobalConfigFilePath()
{
    return getInstance()->configGlobalPath;
}

/**
 * \brief Search path to for locating configurations
 */
const std::set<std::string> &coConfigDefaultPaths::getSearchPath()
{
    return getInstance()->searchPath;
}

/**
 * \brief Initialisation of the config file locations
 */
void coConfigDefaultPaths::setNames()
{

    // define global config file
    const char *yacdir;

    // Check for environment variables to override default locations
    auto configGlobalOverride = getenv("COCONFIG");
    auto configLocalOverride = getenv("COCONFIG_LOCAL");
    auto configDir = getenv("COCONFIG_DIR");

    if (coConfigConstants::getBackend() == "covise")
    {
        COCONFIGDBG_DEFAULT("coConfigConstants::setNames info: covise environment");
        yacdir = getenv("COVISEDIR");
    }
    else
    {
        COCONFIGLOG("coConfigConstants::setNames warn: unknown environment");
        yacdir = getenv("COVISEDIR");
    }

    if (yacdir || configDir)
    {
        if (configDir)
            configGlobalPath = configDir + pathSeparator;
        else
            configGlobalPath = std::string(yacdir) + pathSeparator + "config" + pathSeparator;

        // Look for config if override is given
        if (configGlobalOverride && boost::filesystem::exists(configGlobalOverride))
        {
            configGlobal = configGlobalOverride;
        }
        else if (configGlobalOverride && boost::filesystem::exists(configGlobalPath + pathSeparator + configGlobalOverride))
        {
            configGlobal = configGlobalPath + pathSeparator + configGlobalOverride;
        }
        // Look in default locations
        else
        {
            if (configGlobalOverride)
                COCONFIGLOG("coConfigDefaultPaths::setNames warn: global override not found, trying default config");

            if (boost::filesystem::exists(configGlobalPath + "config." + coConfigConstants::getHostname() + ".xml"))
            {
                configGlobal = configGlobalPath + "config." + coConfigConstants::getHostname() + ".xml";
            }
            else
            {
                configGlobal = configGlobalPath + "config.xml";
            }
        }
    }
    else if (!configGlobalOverride)
    {
        COCONFIGLOG("coConfigDefaultPaths::setNames warn: no COVISE_PATH set");
        configGlobal.clear();
    }

// define local config file
#ifndef _WIN32
    auto homedir = getenv("HOME");
    std::string path = "/." + coConfigConstants::getBackend() + "/";
#else
    auto homedir = getenv("USERPROFILE");
    std::string path = "/" + coConfigConstants::getBackend() + "/";
#endif

    if (homedir)
    {
        configLocalPath = boost::filesystem::path{boost::filesystem::path{homedir + path}.make_preferred().native() }.string();

        // Create path to store configuration. For security reasons don't create if override is set.
        if (!boost::filesystem::exists(configLocalPath) && !configLocalOverride)
        {
            if (boost::filesystem::create_directory(configLocalPath))
            {
                COCONFIGDBG("coConfigDefaultPaths::setNames() info: created path:" << configLocalPath);
            }
            else
            {
                COCONFIGLOG("coConfigDefaultPaths::setNames() err: Could not create path:" << configLocalPath);
            }
        }

        // Look for config if override is given
        if (configLocalOverride && boost::filesystem::exists(configLocalOverride))
        {
            configLocal = configLocalOverride;
        }
        else if (configLocalOverride && boost::filesystem::exists(configLocalPath + pathSeparator + configLocalOverride))
        {
            configLocal = configLocalPath + pathSeparator + configLocalOverride;
        }
        // Look in default locations
        else
        {
            if (configLocalOverride)
                COCONFIGLOG("coConfigDefaultPaths::setNames warn: local override not found, trying default config");

            if (boost::filesystem::exists(configLocalPath + "config." + coConfigConstants::getHostname() + ".xml"))
            {
                configLocal = configLocalPath + "config." + coConfigConstants::getHostname() + ".xml";
            }
            else
            {
                configLocal = configLocalPath + "config.xml";
            }
        }
    }
    else if (configLocalOverride)
    {
        COCONFIGLOG("coConfigDefaultPaths::setNames() warn: no HOME set");
        configLocal = configLocalOverride;
    }

    COCONFIGDBG("coConfigDefaultPaths::setNames info: Searching user config in " << configLocal);

    defaultTransform = std::string(yacdir) + pathSeparator + CO_CONFIG_TRANSFORM_FILE;

    std::string sPath;
    auto pathEnv = getenv("COVISE_PATH");
    if (pathEnv && coConfigConstants::getBackend() == "covise")
        sPath = pathEnv;
    else
        COCONFIGDBG("coConfigDefaultPaths::setNames warn: unknown environment, not setting any search path");

    if (configDir)
        searchPath = std::set<std::string>{configDir};
    else
        searchPath = std::set<std::string>{};

#ifdef _WIN32
    char splitter = ';';
#else
    char splitter = ':';
#endif
    auto s = split(sPath, splitter, true);
    searchPath.insert(s.begin(), s.end());
}

void coConfigConstants::initXerces()
{
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *pMsg = xercesc::XMLString::transcode(toCatch.getMessage());
        COCONFIGLOG("coConfigEntry::initXerces init failed" << pMsg);
        xercesc::XMLString::release(&pMsg);
    }
}

// Not used, yet
void coConfigConstants::terminateXerces()
{
    try
    {
        xercesc::XMLPlatformUtils::Terminate();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *pMsg = xercesc::XMLString::transcode(toCatch.getMessage());
        COCONFIGLOG("coConfigEntry::terminteXerces failed" << pMsg);
        xercesc::XMLString::release(&pMsg);
    }
}
