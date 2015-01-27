/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FileSysAccess.h"
#include <QDir>
#include <string>
#include <config/coConfig.h>
#include <config/coConfigValue.h>

#ifdef _WIN32
#include "windows.h"
#endif
using namespace opencover;
using namespace covise;

QStringList FileSysAccess::getFileSysList(QString filter, QString &path, bool listFiles)
{
    QDir locDir(path);
    QStringList list;

    if (filter.isEmpty())
        filter = "*";
    locDir.setNameFilters(QStringList(filter));

    if (listFiles)
    {
        if (!locDir.exists() || path.isEmpty())
        {
            locDir.setPath(locDir.current().path());
        }
        list = locDir.entryList(QDir::Files, QDir::Name);
    }
    else
    {
        if (!locDir.exists())
        {
            locDir.setPath(locDir.current().path());
        }
        list = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    }
    path = locDir.path();

    return list;
}

std::string FileSysAccess::getFileName(const std::string dir, const char *dirSep)
{
    std::string::size_type pos = std::string::npos;
    pos = dir.rfind(dirSep);
    std::string fileName = "";

    if (pos != std::string::npos)
        fileName = dir.substr(pos, dir.size() - pos);

    return fileName;
}

std::string FileSysAccess::getTempDir()
{
    std::string tempPath = "";

#ifdef WIN32
    DWORD reqBufferSize = GetTempPathA(0, NULL);
    char *lpPathBuffer = new char[reqBufferSize];
    reqBufferSize = GetTempPathA(reqBufferSize, lpPathBuffer);
    tempPath = lpPathBuffer;
    coConfigString cotempPath = coConfig::getInstance()->getString("value", "COVER.Tmp", lpPathBuffer);
    QString qtemp = (QString)cotempPath;
    tempPath = qtemp.toStdString();
    delete[] lpPathBuffer;
#else
    coConfigString cotempPath = coConfig::getInstance()->getString("value", "COVER.Tmp", "/var/tmp");
    QString qtemp = (QString)cotempPath;
    tempPath = qtemp.toStdString();
#endif

    return tempPath;
}

QStringList FileSysAccess::getLocalHomeDir()
{
    QDir locDir;
    QString homeDir = locDir.homePath();
    return this->getFileSysList("", homeDir, false);
}

QStringList FileSysAccess::getLocalHomeFiles(const std::string filter)
{
    QDir locDir;
    QString homeDir = locDir.homePath();
    return this->getFileSysList(QString::fromStdString(filter), homeDir, true);
}

QString FileSysAccess::getLocalHomeDirStr()
{
    QDir locDir;
    QString homeDir = locDir.homePath();
    return homeDir;
}
