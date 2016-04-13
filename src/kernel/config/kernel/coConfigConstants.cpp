/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigLog.h>
#include <config/coConfigConstants.h>
#include <config/coConfig.h>

#ifdef _WIN32
#include <winsock2.h>
#include <locale.h>
#else
#include <unistd.h>
#endif

#include <stdlib.h>

#include <QDir>

#include <iostream>
using namespace std;
using namespace covise;

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>

coConfigDefaultPaths *coConfigDefaultPaths::instance = 0;
coConfigConstants *coConfigConstants::instance = 0;

coConfigConstants::coConfigConstants()
{
    if (instance == 0)
        initXerces();

    instance = this;

#ifdef YAC
    backend = "yac";
#else
    backend = "covise";
#endif

    hostname = QString::null;

    rank = -1;
}

coConfigConstants::~coConfigConstants()
{
    if (instance == this)
        instance = 0;
}

const QStringList &coConfigConstants::getArchList()
{

    if (!instance)
        new coConfigConstants();

    if (instance->archlist.empty())
    {
#ifdef _WIN32
        instance->archlist << "windows";
#else
        instance->archlist << "unix";

#ifdef __APPLE__
        instance->archlist << "mac";
#else
        instance->archlist << "x11";
#endif
#endif

        const char *archsuffix = getenv("ARCHSUFFIX");
        if (archsuffix)
        {
            QString as(archsuffix);
            instance->archlist << as;

            if (as.endsWith("opt"))
            {
                instance->archlist << as.left(as.length() - 3);
            }
        }
    }

    return instance->archlist;
}

const QString &coConfigConstants::getHostname()
{

    if (!instance)
        new coConfigConstants();

    if (instance->hostname == QString::null)
    {
        instance->hostname = getenv("COVISE_HOST");
        if (instance->hostname != QString::null)
        {
            COCONFIGDBG_DEFAULT("coConfigConstants::getHostname info: LOCAL hostname is '" + instance->hostname + "' (from COVISE_HOST)");
        }
    }
    if (instance->hostname == QString::null)
    {
        instance->hostname = getenv("COVISE_CONFIG");
    }

    if (instance->hostname == QString::null)
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
            instance->hostname = QString(hostnameTmp).toLower();
            if (instance->hostname.contains('.'))
                instance->hostname = instance->hostname.split('.')[0];
        }

        if (instance->hostname != QString::null)
        {
            COCONFIGDBG_DEFAULT("coConfigConstants::getHostname info: LOCAL hostname is '" + instance->hostname + "' (from gethostname)");
        }
    }

    return instance->hostname;
}

void coConfigConstants::setMaster(const QString &hostname)
{
    if (!instance)
        new coConfigConstants();
    instance->master = hostname.toLower();
    coConfig::getInstance()->setActiveCluster(hostname);
}

const QString &coConfigConstants::getMaster()
{

    if (!instance)
        new coConfigConstants();

    if (instance->master == QString::null)
    {
        //COCONFIGLOG("coConfigConstants::getMaster info: master hostname is not set");
    }
    return instance->master;
}

void coConfigConstants::setRank(int rank)
{
    if (!instance)
        new coConfigConstants();
    instance->rank = rank;
}

int coConfigConstants::getRank()
{
    if (!instance)
        new coConfigConstants();
    return instance->rank;
}

void coConfigConstants::setBackend(const QString &backend)
{
    if (!instance)
        new coConfigConstants();
    instance->backend = backend;
}

const QString &coConfigConstants::getBackend()
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
const QString &coConfigDefaultPaths::getDefaultTransformFileName()
{
    return getInstance()->defaultTransform;
}

/**
 * \brief Full filename of the local configuration
 */
const QString &coConfigDefaultPaths::getDefaultLocalConfigFileName()
{
    return getInstance()->configLocal;
}

/**
 * \brief Full filename of the global configuration
 */
const QString &coConfigDefaultPaths::getDefaultGlobalConfigFileName()
{
    return getInstance()->configGlobal;
}

/**
 * \brief Full path to the local configuration
 */
const QString &coConfigDefaultPaths::getDefaultLocalConfigFilePath()
{
    return getInstance()->configLocalPath;
}

/**
 * \brief Full path to the global configuration
 */
const QString &coConfigDefaultPaths::getDefaultGlobalConfigFilePath()
{
    return getInstance()->configGlobalPath;
}

/**
 * \brief Search path to for locating configurations
 */
const QStringList &coConfigDefaultPaths::getSearchPath()
{
    return getInstance()->searchPath;
}

/**
 * \brief Initialisation of the config file locations
 */
void coConfigDefaultPaths::setNames()
{

    // define global config file
    QString yacdir;

    // Check for environment variables to override default locations
    QString configGlobalOverride = getenv("COCONFIG");
    QString configLocalOverride = getenv("COCONFIG_LOCAL");
    QString configDir = getenv("COCONFIG_DIR");

    if (coConfigConstants::getBackend() == "yac")
    {
        COCONFIGDBG_DEFAULT("coConfigConstants::setNames info: yac environment");
        yacdir = getenv("YACDIR");
    }
    else if (coConfigConstants::getBackend() == "covise")
    {
        COCONFIGDBG_DEFAULT("coConfigConstants::setNames info: covise environment");
        yacdir = getenv("COVISEDIR");
    }
    else
    {
        COCONFIGLOG("coConfigConstants::setNames warn: unknown environment");
        yacdir = getenv("YACDIR");
        if (yacdir.isEmpty())
        {
            yacdir = getenv("COVISEDIR");
        }
    }

    if (!yacdir.isEmpty() || !configDir.isEmpty())
    {
        if (!configDir.isEmpty())
            configGlobalPath = configDir + QDir::separator();
        else
            configGlobalPath = yacdir + QDir::separator() + "config" + QDir::separator();

        // Look for config if override is given
        if (!configGlobalOverride.isEmpty() && QFile::exists(configGlobalOverride))
        {
            configGlobal = configGlobalOverride;
        }
        else if (!configGlobalOverride.isEmpty() && QFile::exists(QString("%1%2%3").arg(configGlobalPath, QString(QChar(QDir::separator())), configGlobalOverride)))
        {
            configGlobal = configGlobalPath + QDir::separator() + configGlobalOverride;
        }
        // Look in default locations
        else
        {
            if (!configGlobalOverride.isEmpty())
                COCONFIGLOG("coConfigDefaultPaths::setNames warn: global override not found, trying default config");

            if (QFile::exists(QString("%1config.%2.xml").arg(configGlobalPath, coConfigConstants::getHostname())))
            {
                configGlobal = configGlobalPath + "config." + coConfigConstants::getHostname() + ".xml";
            }
            else
            {
                configGlobal = configGlobalPath + "config.xml";
            }
        }
    }
    else if (configGlobalOverride.isEmpty())
    {
        COCONFIGLOG("coConfigDefaultPaths::setNames warn: no COVISE- or YACPATH set");
        configGlobal = configGlobalOverride;
    }

// define local config file
#ifndef _WIN32
    QString homedir = getenv("HOME");
    QString path = QString("/.%1/").arg(coConfigConstants::getBackend());
#else
    QString homedir = getenv("USERPROFILE");
    QString path = QString("/%1/").arg(coConfigConstants::getBackend());
#endif

    if (!homedir.isEmpty())
    {
#if QT_VERSION >= 0x050000
        configLocalPath = QDir::toNativeSeparators(homedir + path);
#else
        configLocalPath = QDir::convertSeparators(homedir + path);
#endif
        QDir dir;

        // Create path to store configuration. For security reasons don't create if override is set.
        if (!dir.exists(configLocalPath) && configLocalOverride.isEmpty())
        {
            if (dir.mkdir(configLocalPath))
            {
                COCONFIGDBG("coConfigDefaultPaths::setNames() info: created path:" << configLocalPath);
            }
            else
            {
                COCONFIGLOG("coConfigDefaultPaths::setNames() err: Could not create path:" << configLocalPath);
            }
        }

        // Look for config if override is given
        if (!configLocalOverride.isEmpty() && QFile::exists(configLocalOverride))
        {
            configLocal = configLocalOverride;
        }
        else if (!configLocalOverride.isEmpty() && QFile::exists(QString("%1%2%3").arg(configLocalPath, QString(QChar(QDir::separator())), configLocalOverride)))
        {
            configLocal = configLocalPath + QDir::separator() + configLocalOverride;
        }
        // Look in default locations
        else
        {
            if (!configLocalOverride.isEmpty())
                COCONFIGLOG("coConfigDefaultPaths::setNames warn: local override not found, trying default config");

            if (QFile::exists(configLocalPath + "config." + coConfigConstants::getHostname() + ".xml"))
            {
                configLocal = configLocalPath + "config." + coConfigConstants::getHostname() + ".xml";
            }
            else
            {
                configLocal = configLocalPath + "config.xml";
            }
        }
    }
    else if (!configLocalOverride.isEmpty())
    {
        COCONFIGLOG("coConfigDefaultPaths::setNames() warn: no HOME set");
        configLocal = configLocalOverride;
    }

    COCONFIGDBG("coConfigDefaultPaths::setNames info: Searching user config in " << configLocal);

    defaultTransform = yacdir + QDir::separator() + CO_CONFIG_TRANSFORM_FILE;

    QString sPath;

    if (coConfigConstants::getBackend() == "covise")
        sPath = getenv("COVISE_PATH");
    else if (coConfigConstants::getBackend() == "yac")
        sPath = getenv("YAC_PATH");
    else
        COCONFIGDBG("coConfigDefaultPaths::setNames warn: unknown environment, not setting any search path");

    if (!configDir.isEmpty())
        searchPath = QStringList(configDir);
    else
        searchPath = QStringList();

#ifdef _WIN32
    searchPath += sPath.split(';', QString::SkipEmptyParts);
#else
    searchPath += sPath.split(':', QString::SkipEmptyParts);
#endif
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
