/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef FB_USE_AG
#include "AGData.h"
#include <util/cotabletuimessages.h>

#include <QDir>

AGData::AGData(coTUIElement *elem)
    : IRemoteData()
{
    this->mTUIElement = elem;
    mDataStore = new CDataStore("https://141.58.8.10:8000/Venues/default", "http://141.58.8.10:11000/venueclient");
}

AGData::~AGData(void)
{
    delete mDataStore;
}

void AGData::reqDirectoryList(std::string path, int pId)
{
    std::string id = "";
    if (path != "")
    {
        id = mDataStore->getIdFromPath(path);
    }
    if (path == "AccessGrid")
    {
        id = "";
    }
    std::vector<FileDesc> list = mDataStore->getHierarchicalDirList(id);
    TokenBuffer tb;
    tb << TABLET_SET_DIRLIST;
    tb << pId;
    tb << list.size();

    for (int i = 0; i < list.size(); i++)
    {
        tb << list.at(i).filename.c_str();
    }

    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    this->setDirectoryList(m);
}

void AGData::setDirectoryList(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setDirList(msg);
    }
}

void AGData::reqFileList(std::string path, int pId)
{
    std::string id = "";
    if (path != "")
    {
        id = mDataStore->getIdFromPath(path);
    }
    if (path == "AccessGrid")
    {
        id = "";
    }
    std::vector<FileDesc> list = mDataStore->getHierarchicalFileList(id);

    // TODO: Apply filter on file list

    TokenBuffer tb;
    tb << TABLET_SET_FILELIST;
    tb << pId;
    tb << list.size();

    for (int i = 0; i < list.size(); i++)
    {
        tb << list.at(i).filename.c_str();
    }

    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    this->setFileList(m);
}

void AGData::setFileList(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setFileList(msg);
    }
}

void AGData::reqHomeDir(int pId)
{
    std::vector<FileDesc> list = mDataStore->getHierarchicalDirList("-1");
    TokenBuffer tb;
    tb << TABLET_SET_FILELIST;
    tb << pId;
    tb << list.size();

    for (int i = 0; i < list.size(); i++)
    {
        tb << list.at(i).filename.c_str();
    }

    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    this->setDirectoryList(m);
}

void AGData::reqHomeFiles(int pId)
{
    std::vector<FileDesc> list = mDataStore->getHierarchicalFileList("-1");

    TokenBuffer tb;
    tb << TABLET_SET_FILELIST;
    tb << pId;
    tb << list.size();

    for (int i = 0; i < list.size(); i++)
    {
        tb << list.at(i).filename.c_str();
    }

    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    this->setFileList(m);
}

void AGData::reqDirUp(std::string basePath)
{
    std::string id = "";
    if (basePath != "")
    {
        id = mDataStore->getIdFromPath(basePath);
    }
    if (basePath == "AccessGrid")
    {
        id = "";
    }
    std::vector<FileDesc> list = mDataStore->getFlatDataList();
    std::string new_dir;
    for (int i = 0; i < list.size(); i++)
    {
        if (list.at(i).id == id)
        {
            new_dir = list.at(i).filename;
            id = list.at(i).id;
        }
    }

    list = mDataStore->getHierarchicalDirList(id);

    TokenBuffer tb;
    tb << TABLET_SET_DIRLIST;
    tb << 1;
    tb << list.size();

    for (int i = 0; i < list.size(); i++)
    {
        tb << list.at(i).filename.c_str();
    }

    Message m(tb);
    this->mId = 1;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    this->setDirectoryList(m);

    list = mDataStore->getHierarchicalFileList(id);

    TokenBuffer tb2;
    tb2 << TABLET_SET_FILELIST;
    tb2 << 1;
    tb2 << list.size();

    for (int i = 0; i < list.size(); i++)
    {
        tb2 << list.at(i).filename.c_str();
    }

    Message m2(tb2);
    this->mId = 1;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    this->setFileList(m2);
}

void AGData::setVRBC(VRBClient *vrbc)
{
    this->mVrbc = vrbc;
}

VRBClient *AGData::getVRB()
{
    return this->mVrbc;
}

void AGData::setId(int id)
{
    this->mId = id;
}

void AGData::setCurDir(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setCurDir(msg);
    }
}

void *AGData::getTmpFileHandle(bool sync)
{
    return NULL;
}

std::string AGData::getTmpFilename(const std::string url)
{
    return mLastFile;
}

void AGData::reqClientList(int pId)
{
}

void AGData::setClientList(Message &msg)
{
}

void AGData::reqDrives(int pId)
{
}

void AGData::setFile(std::string Filename)
{
    //Connect to AG-DataStore and retrieve file to /tmp
    std::vector<FileDesc> locFileList = mDataStore->getFlatFileList();
    int fileIndex = 0;

    for (int i = 0; i < locFileList.size(); i++)
    {
        if (locFileList.at(i).filename.find(Filename) != std::string::npos)
        {
            fileIndex = i;
            break;
        }
    }

    mLastFile = mDataStore->getLocalFilename(locFileList.at(fileIndex));

    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->dataTransferReady();
    }
}

void AGData::setSelectedPath(std::string path)
{
    this->mFileLocation = "agtk3://";
    this->mFileLocation = this->mFileLocation + path;
}

std::string AGData::getSelectedPath()
{
    return this->mFileLocation;
}
#endif
