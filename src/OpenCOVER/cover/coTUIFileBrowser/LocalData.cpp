/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LocalData.h"
#include "coVRFileManager.h"
#include <util/coTabletUIMessages.h>
#include <net/tokenbuffer.h>
#include <tui/coAbstractTabletUI.h>
#include <QDir>
#include <QString>
#include <QStringList>

using namespace opencover;
using namespace covise;
//#define FILEBROWSER_DEBUG

LocalData::LocalData(coTUIElement *elem)
    : IData()
{
    this->mTUIElement = elem;
    //	this->mLocationType = type;
}

LocalData::~LocalData(void)
{
}

void LocalData::reqDirectoryList(std::string path, int pId)
{

    QDir locDir(QString(path.c_str()));
    QStringList locDirList;
    if (this->mFilter != "")
    {
        //TODO: Filter bugfix
        QString filter = QString(this->mFilter.c_str());
        locDir.setNameFilters(QStringList(filter));
    }
    else
    {
        locDir.setNameFilters(QStringList(QString("*.*")));
    }

    //TODO: For now directory list is local
    //needs to be converted to vector<string>
    //*(this->mDirectoryList) = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    locDirList = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);

    TokenBuffer rt;
    rt << TABLET_SET_DIRLIST;
    rt << pId;
    rt << (int)locDirList.size();

    for (int i = 0; i < locDirList.size(); i++)
    {
        std::string sdl = locDirList.at(i).toStdString();
        //std::cerr << "reqDirectoryList: Directory entry #" <<  i << " = " << sdl.c_str() << std::endl;
        rt << sdl.c_str();
    }

    Message m(rt);
    //coAbstractTUIFileBrowserButton* tuiElem = (coAbstractTUIFileBrowserButton*) this->mTUIElement;
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);

    //Warum ist dieser Code da?
    if (tuiElem != NULL)
    {
        tuiElem->setDirList(m);

        /*QString newPath(locDir.absolutePath());
      TokenBuffer rt2;
      rt2 << TABLET_SET_CURDIR;
      std::string snewpath =newPath.toStdString();
      std::cerr << "reqDirectoryList: Current Directory" << snewpath.c_str() << std::endl;
      rt2 << snewpath.c_str();

      Message m2(rt2);
      tuiElem->setCurDir(m2);
      std::cerr << "Sent dir update to TUI!" << std::endl;*/
    }
}

void LocalData::setDirectoryList(const Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setDirList(msg);
    }
}

void LocalData::reqFileList(std::string path, int pId)
{
#ifdef FILEBROWSER_DEBUG
    std::cerr << "reqFileList: " << path.c_str() << std::endl;
#endif
    QDir locDir(QString(path.c_str()));
    QStringList locFileList;
    QStringList filters;

    if (!(mFilter == ""))
    {
        //split multiple filters into QStringList
        QString filterString = mFilter.c_str();
        if (filterString.contains(";"))
        {
            filters = filterString.split(";");
        }
        else
            filters.append(filterString);
        locDir.setNameFilters(filters);
    }
    else
    {
        locDir.setNameFilters(QStringList(QString("*.*")));
    }

    //TODO: convert QStringList to vector<string>
    //For now use local variable
    //*(this->mFileList) = locDir.entryList(QDir::Files, QDir::Name);
    locFileList = locDir.entryList(QDir::Files, QDir::Name);

    TokenBuffer rt;
    rt << TABLET_SET_FILELIST;
    rt << pId;
    rt << (int)locFileList.size();

    for (int i = 0; i < locFileList.size(); i++)
    {
        std::string sfl = locFileList.at(i).toStdString();
        /*std::cerr << "reqDirectoryList: File entry #" <<  i << " = " << sfl.c_str() << std::endl;*/
        rt << sfl.data();
    }

    Message m(rt);
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setFileList(m);
    }
}

void LocalData::setFileList(const Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setFileList(msg);
    }
}

void LocalData::reqHomeDir(int pId)
{
    QDir locDir;
    QStringList locDirList;

    locDir.setCurrent(locDir.homePath());

    //TODO: For now directory list is local
    //needs to be converted to vector<string>
    //*(this->mDirectoryList) = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    locDirList = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);

    TokenBuffer rt;
    rt << TABLET_SET_DIRLIST;
    rt << pId;
    rt << (int)locDirList.size();

    for (int i = 0; i < locDirList.size(); i++)
    {
        std::string sdl = locDirList.at(i).toStdString();
        rt << sdl.c_str();
        /*std::cerr << "reqHomeDir: HomeDir entry #" << i << " = "  << sdl.c_str() << std::endl;*/
    }

    Message m(rt);
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setDirList(m);
    }
}

void LocalData::reqHomeFiles(int pId)
{
    QDir locDir;
    QStringList locFileList;

    locDir.setCurrent(locDir.homePath());

    //TODO: convert QStringList to vector<string>
    //For now use local variable
    //*(this->mFileList) = locDir.entryList(QDir::Files, QDir::Name);
    locFileList = locDir.entryList(QDir::Files, QDir::Name);

    TokenBuffer rt;
    rt << TABLET_SET_FILELIST;
    rt << pId;
    rt << (int)locFileList.size();

    for (int i = 0; i < locFileList.size(); i++)
    {
        std::string sfl = locFileList.at(i).toStdString();
        rt << sfl.c_str();
        /*std::cerr << "reqHomeFiles: HomeFile entry #" << i << " = "  << sfl.c_str() << std::endl;*/
    }

    Message m(rt);
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setFileList(m);
    }
}

void LocalData::reqDirUp(std::string basePath, int pId)
{
    /*std::cerr << "reqDirUp: basePath" << basePath.c_str() << std::endl;*/
    QDir locDir(QString(basePath.c_str()));
    QStringList locDirList;
    QStringList locFileList;

    locDir.cdUp();

    //TODO: For now directory list is local
    //needs to be converted to vector<string>
    //*(this->mFileList) = locDir.entryList(QDir::Files, QDir::Name);
    //*(this->mDirectoryList) = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    locDirList = locDir.entryList(QDir::Files, QDir::Name);
    locFileList = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);

    TokenBuffer rt;
    rt << TABLET_SET_DIRLIST;
    rt << pId;
    rt << (int)locDirList.size();

    for (int i = 0; i < locDirList.size(); i++)
    {

        std::string sdl = locDirList.at(i).toStdString();
        rt << sdl.c_str();
        /*std::cerr << "reqDirUp: Directory entry #" << i << " = "  << sdl.c_str() << std::endl;*/
    }

    Message m(rt);
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setDirList(m);

        TokenBuffer tb;
        tb << TABLET_SET_FILELIST;
        tb << pId;
        tb << (int)locFileList.size();

        for (int i = 0; i < locFileList.size(); i++)
        {
            std::string sfl = locFileList.at(i).toStdString();
            tb << sfl.c_str();
            /*std::cerr << "reqDirUp: File entry #" << i << " = "  << sfl.c_str() << std::endl;*/
        }

        Message mf(tb);
        tuiElem->setFileList(mf);
    }
}

void LocalData::setCurDir(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setCurDir(msg);
    }
}

//LocationType LocalData::getType()
//{
////	return this->mLocationType;
//	return;
//}

void *LocalData::getTmpFileHandle(bool /*sync*/)
{
    return NULL;
}

std::string LocalData::getTmpFilename(const std::string url, int)
{

    //Manipulate path for retrieving the file location on local system
    std::string::size_type pos = url.find("//");
    std::string path = url.substr(pos + 1, url.size() - (pos + 1));
    /*std::cerr << "LocalData.cpp: File Location: " << path << std::endl;*/
    if(path.length()>2 && path[2]==':') // this is a windows drive letter
    {
        this->setFile(path.substr(1,path.size()-1));
    }
    else
    {
        this->setFile(path);
    }
    return this->mFile;
}

void LocalData::setHomeDir()
{
    QDir locDir;
    QString homeDir = locDir.homePath();
    std::string strHome = homeDir.toStdString();
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setCurDir(strHome.c_str());
    }
}

void LocalData::reqDrives(int pId)
{
    QDir locDir;
    QFileInfoList list = locDir.drives();

    TokenBuffer tb;
    tb << TABLET_SET_DRIVES;
    tb << pId;
    int locSize = list.size();
    tb << locSize;

    for (int i = 0; i < list.size(); i++)
    {
        QString entry = list.at(i).path();
        std::string drive = entry.toStdString();
        tb << drive.c_str();
        /*std::cerr << "reqDrive: Drive entry #" << i << " = " << drive.c_str() << std::endl;*/
    }

    Message mf(tb);
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setDrives(mf);
    }
}

void LocalData::setFile(std::string file)
{
    this->mFile = file;
}

void LocalData::setSelectedPath(std::string path)
{
    this->mFileLocation = "file://" + path;
}

std::string LocalData::getSelectedPath()
{
    return this->mFileLocation;
}

std::string LocalData::resolveToAbsolute(const std::string &dir)
{
    auto d = dir;
    Url url = Url::fromFileOrUrl(dir);
    if (url.valid() && url.scheme() == "file")
    {
        d = url.authority() + url.path();
    }
    QDir tempDir(QString(d.c_str()));
    return tempDir.absolutePath().toStdString();
}
