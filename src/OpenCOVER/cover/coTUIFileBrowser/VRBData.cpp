/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VRBData.h"
#include <util/coTabletUIMessages.h>
#include <util/coFileUtil.h>
#include <net/tokenbuffer.h>
#include <osgDB/fstream>
//#include <fstream>
#include <QDir>
#include <qtutil/FileSysAccess.h>
#include <cover/coVRCommunication.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <net/message.h>
#include <net/message_types.h>

using namespace opencover;
using namespace covise;

VRBData::VRBData(coTUIElement *elem)
    : IRemoteData()
{
    this->mTUIElement = elem;
}

VRBData::~VRBData(void)
{
}

void VRBData::reqDirectoryList(std::string path, int pId)
{
    TokenBuffer tb;
    tb << pId;
    tb << getVRB()->ID();
    tb << TABLET_REQ_DIRLIST;
    tb << this->mFilter.c_str();
    if (path != "")
    {
        tb << path.c_str();
    }
    else
    {
        tb << this->mCurrentPath.c_str();
    }
    tb << this->mIP.c_str();
    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setDirectoryList(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setDirList(msg);
    }
}

void VRBData::reqFileList(std::string path, int pId)
{
    TokenBuffer tb;
    tb << pId;
    tb << getVRB()->ID();
    tb << TABLET_REQ_FILELIST;
    tb << mFilter.c_str();

    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setCurDir(path.c_str());
    }

    if (path != "")
    {
        tb << path.c_str();
    }
    else
    {
        tb << this->mCurrentPath.c_str();
    }
    tb << this->mIP.c_str();
    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setFileList(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setFileList(msg);
    }
}

//TODO: Implement ReqHomeDir
void VRBData::reqHomeDir(int pId)
{
    TokenBuffer tb;
    tb << pId;
    tb << getVRB()->ID();
    tb << TABLET_REQ_HOMEDIR;
    tb << this->mIP.c_str();

    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}

//TODO: Implement ReqHomeFiles
void VRBData::reqHomeFiles(int pId)
{
    TokenBuffer tb;
    tb << pId;
    tb << getVRB()->ID();
    tb << TABLET_REQ_HOMEFILES;
    tb << mFilter.c_str();
    tb << this->mIP.c_str();

    Message m(tb);
    this->mId = pId;
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}

//TODO: Implement ReqDirUp
void VRBData::reqDirUp(std::string /*basePath*/)
{
}

vrb::VRBClient *VRBData::getVRB()
{
    // Using vrbc from OpenCOVER.h
    return vrbc;
}

void VRBData::setId(int id)
{
    this->mId = id;
}

int VRBData::getId()
{
    return this->mId;
}

void VRBData::setCurDir(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setCurDir(msg);
    }
}

//char* VRBData::getType()
//{
//	return this->mLocationType;
//}

void VRBData::reqClientList(int pId)
{
    TokenBuffer tb;
    tb << pId;
    tb << getVRB()->ID();
    tb << TABLET_REQ_CLIENTS;
    Message m(tb);
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setClientList(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setClientList(msg);
    }
}

void VRBData::reqDrives(int pId)
{
    TokenBuffer tb;
    tb << pId;
    tb << getVRB()->ID();
    tb << TABLET_REQ_DRIVES;
    tb << "";
    tb << "";
    tb << this->mIP.c_str();
    Message m(tb);
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setRemoteDirList(Message &msg)
{
    TokenBuffer tb(&msg);

    int type = 0;
    int recvid = 0;
    int recv_vrbId = 0;
    char *location = NULL;
    char *filter = NULL;

    tb >> type;
    tb >> recvid;
    tb >> recv_vrbId;
    tb >> location;
    tb >> filter;

    QString path(location);
    QString locFilter(filter);

    QDir locDir(path);

    if (this->mFilter != "")
    {
        locDir.setNameFilters(QStringList(locFilter));
    }

    QStringList locDirList = locDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    QString qAbsPath = locDir.absolutePath();

    TokenBuffer rt;
    rt << TABLET_REMSET_DIRLIST;
    rt << recvid;
    rt << recv_vrbId;
    rt << locDirList.size();

    for (int i = 0; i < locDirList.size(); i++)
    {
        std::string sdir = locDirList.at(i).toStdString();
        rt << sdir.c_str();
    }

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_REMREQ;
    cover->sendVrbMessage(&m);

    if (strcmp(location, "") == 0)
    {
        this->setRemoteDir(msg, qAbsPath.toStdString());
    }
}

void VRBData::setRemoteFileList(Message &msg)
{
    TokenBuffer tb(&msg);

    int type = 0;
    int recvid = 0;
    char *location = NULL;
    char *filter = NULL;
    int recv_vrbId = 0;

    tb >> type;
    tb >> recvid;
    tb >> recv_vrbId;
    tb >> location;
    tb >> filter;

    QString path(location);
    QString locFilter(filter);

    QDir locDir(path);

    if (this->mFilter != "")
    {
        locDir.setNameFilters(QStringList(locFilter));
    }

    QStringList locFileList = locDir.entryList(QDir::Files, QDir::Name);

    TokenBuffer rt;
    rt << TABLET_REMSET_FILELIST;
    rt << recvid;
    rt << recv_vrbId;
    int size = locFileList.size();
    rt << size;

    for (int i = 0; i < locFileList.size(); i++)
    {
        std::string sentry = locFileList.at(i).toStdString();
        rt << sentry.c_str();
    }

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_REMREQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setRemoteDir(Message &msg, std::string absPath)
{
    TokenBuffer tb(&msg);

    int type = 0;
    int recvid = 0;
    int id = 0;

    tb >> type;
    tb >> id;
    tb >> recvid;

    TokenBuffer rt;
    rt << TABLET_REMSET_DIRCHANGE;
    rt << id;
    rt << recvid;
    rt << absPath.c_str();

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_REMREQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setRemoteDrives(Message &msg)
{
    TokenBuffer tb(&msg);

    int type = 0;
    int recvid = 0;
    char *location = NULL;
    char *filter = NULL;
    int id = 0;

    tb >> type;
    tb >> id;
    tb >> recvid;
    tb >> location;
    tb >> filter;

    QString path(location);
    QString locFilter(filter);

    QFileInfoList list = QDir::drives();

    TokenBuffer rt;
    rt << TABLET_REMSET_DRIVES;
    rt << id;
    rt << recvid;
    int size = list.size();
    rt << size;

    for (int i = 0; i < list.size(); i++)
    {
        QString qentry = list.at(i).absolutePath();
        std::string sentry = qentry.toStdString();
        rt << sentry.c_str();
    }

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_REMREQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setDrives(Message &msg)
{
    coTUIFileBrowserButton *tuiElem = dynamic_cast<coTUIFileBrowserButton *>(this->mTUIElement);
    if (tuiElem != NULL)
    {
        tuiElem->setDrives(msg);
    }
}

void *VRBData::getTmpFileHandle(bool /*sync*/)
{
    return NULL;
}

std::string VRBData::getTmpFilename(const std::string url, int id)
{
    int pId = id;

    //Request File and wait for arrival
    std::string::size_type pos = url.find("//");
    std::string path = url.substr(pos + 2, url.size() - (pos + 2));
    pos = path.find("/");
    std::string remotePoint = path.substr(0, pos);
    path = path.substr(pos + 1, path.size() - (pos + 1));

    //Determine filename and first check if file exists in local temp cache
    pos = path.rfind("/");
    std::string filename = path.substr(pos, path.size());
    FileSysAccess fileAcc;
    std::string tempDir = fileAcc.getTempDir();
    tempDir = tempDir + filename;

    //Check for existance
    osgDB::ifstream vrbFile(tempDir.c_str());
    if (!(!vrbFile.bad()))
    {
        std::cerr << "VRB-File found in local cache!" << std::endl;
        vrbFile.close();
        return tempDir;
    }

    TokenBuffer rt;
    rt << pId;
    rt << getVRB()->ID();
    rt << TABLET_FB_FILE_SEL;
    rt << remotePoint.c_str();
    rt << path.c_str();
	bool loadAll = false;
	rt << loadAll;

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);

    std::cerr << "Waiting for: " << path.c_str() << std::endl;
    if (this->VRBWait())
    {
        std::cerr << "VRB-Downloaded file: " << mTmpFileName.c_str() << std::endl;
        return this->mTmpFileName;
    }
    else
    {
        std::cerr << "VRB-Download of file failed!" << std::endl;
        return std::string("NoFileData");
    }
}

bool VRBData::VRBWait()
{
    bool bTransferReadyFlag = false;
    int type = 0;
    int id = 0;
    int size = 0;
    const char *data = NULL;
    char *filename = NULL;

    Message *msg = new Message;
    do
    {
        if (coVRMSController::instance()->isMaster())
        {
            getVRB()->wait(msg, COVISE_MESSAGE_VRB_FB_SET);
            coVRMSController::instance()->sendSlaves(msg);
        }
        else
        {
            coVRMSController::instance()->readMaster(msg);
        }
        if (msg->type == COVISE_MESSAGE_VRB_FB_SET)
        {
            TokenBuffer tb(msg);
            tb >> type;

            if (type == TABLET_SET_FILE_NOSUCCESS)
            {
                std::cerr << "Error with file handling on remote endpoint!" << std::endl;
                this->mTmpFileName = "";
                return false;
            }
            else if (type != TABLET_SET_FILE)
            {
                std::cerr << "Wrong message handling in VRBData!" << std::endl;
                coVRCommunication::instance()->handleVRB(msg);
            }

            tb >> id;
            tb >> filename;
            tb >> size;

            data = tb.getBinary(size);
            std::cerr << "Size of retrieved data: " << size << std::endl;

            FileSysAccess fileHelper;
            //Not yet available
            /*coDirectory* dir = coDirectory::getTempDir();
         std::cerr << "Full Dir pathname: " << dir->full_name(0) << std::endl;*/

            std::string stmp = fileHelper.getTempDir();
            std::string sfile = filename;
            std::string::size_type ppos = sfile.find(this->mCurrentPath);

            std::string partPath = sfile.substr(ppos + 1, sfile.size() - (ppos + 1));

            std::string::size_type pos = sfile.rfind("/");
            sfile = sfile.substr(pos + 1, sfile.size() - (pos + 1));

            if (sfile[0] != '/' || sfile[0] != '\\')
            {
                sfile = '/' + sfile;
            }

            stmp = stmp + sfile.c_str();

            std::ofstream vrbFile;
            vrbFile.open(stmp.c_str(), std::ofstream::binary);
            if (vrbFile.fail())
            {
                std::cerr << "Opening of file in tmp failed!" << std::endl;
            }
            vrbFile.write(data, size);
            if (vrbFile.fail())
            {
                std::cerr << "Writing of file data to local file in tmp failed!" << std::endl;
            }
            vrbFile.close();
            std::cerr << "Filename for loading by Covise::FileManager: " << stmp.c_str() << std::endl;
            this->mTmpFileName = stmp;

            bTransferReadyFlag = true;
        }
        else
        {
            coVRCommunication::instance()->handleVRB(msg);
        }
    } while (!bTransferReadyFlag);
    return bTransferReadyFlag;
}

void VRBData::reqRemoteFile(std::string filename, int pId)
{
    //Deprecated

    QDir dir;

    std::string path = this->mCurrentPath + dir.separator().toLatin1() + filename;
    TokenBuffer rt;
    rt << pId;
    rt << getVRB()->ID();
    rt << TABLET_FB_FILE_SEL;
    rt << this->mIP.c_str();
    rt << path.c_str();
	bool loadAll = false;
	rt << loadAll;

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setFile(Message &msg)
{
    std::cerr << "VRBData::setFile entered!" << std::endl;
    TokenBuffer tb(&msg);

    int type;
    int id;
    int size;
    const char *data = NULL;
    char *filename = NULL;

    tb >> type;

    if (type != TABLET_SET_FILE)
    {
        std::cerr << "Wrong message handling in VRBData!" << std::endl;
        return;
    }

    tb >> id;
    tb >> filename;
    tb >> size;
    data = tb.getBinary(size);

    std::cerr << " Start Debug!" << std::endl;
    std::cerr << data << std::endl;
    std::cerr << " End Debug!" << std::endl;

    QDir locDir;

    FileSysAccess fileAcc;
    /*QString tmp = locDir.tempPath();
   std::string sfile = fileAcc.getTempDir();
   tmp = tmp + locDir.separator() + sfile.c_str();
   std::string stmp = tmp.toStdString();*/
    std::string stmp = fileAcc.getTempDir();

    std::ofstream vrbFile;
    vrbFile.open(stmp.c_str(), std::ofstream::binary);
    if (vrbFile.fail())
    {
        //DebugBreak();
    }
    vrbFile.write(data, size);
    if (vrbFile.fail())
    {
        //DebugBreak();
    }

    this->mTmpFileName = stmp;

    vrbFile.close();

    std::cerr << "VRBData::setFile left!" << std::endl;
}

void VRBData::setRemoteFile(Message &msg)
{
    TokenBuffer tb(&msg);
    int type;
    int recvId = 0;
    int id = 0;
    char *path = NULL;

    tb >> type;
    tb >> id;
    tb >> recvId;
    tb >> path;

    int bytes = 0;
    std::ifstream vrbFile;
    //Currently opens files in
    vrbFile.open(path, std::ifstream::binary);
    if (vrbFile.fail())
    {
        //NoSuccess message
        TokenBuffer tb2;
        tb2 << TABLET_REMSET_FILE_NOSUCCESS;
        tb2 << id;
        tb2 << recvId;
        tb2 << "Open failed!";
        std::cerr << "Opening of file for submission failed!" << std::endl;
        Message m(tb2);
        m.type = COVISE_MESSAGE_VRB_FB_REMREQ;
        //Send message
        cover->sendVrbMessage(&m);
        return;
    }
    vrbFile.seekg(0, std::ios::end);
    bytes = vrbFile.tellg();
    char *data = new char[bytes];
    vrbFile.seekg(std::ios::beg);
    vrbFile.read(data, bytes);
    if (vrbFile.fail())
    {
        //NoSuccess message
        TokenBuffer tb2;
        tb2 << TABLET_REMSET_FILE_NOSUCCESS;
        tb2 << id;
        tb2 << recvId;
        tb2 << "Read failed!";
        std::cerr << "Reading of file for submission failed!" << std::endl;
        Message m(tb2);
        m.type = COVISE_MESSAGE_VRB_FB_REMREQ;
        //Send message
        cover->sendVrbMessage(&m);
        return;
    }
    vrbFile.close();

    FileSysAccess file;
    std::string spath = path;
    std::string filename = file.getFileName(spath, "/");

    TokenBuffer rt;
    rt << TABLET_REMSET_FILE;
    rt << id;
    rt << recvId;
    rt << filename.c_str(); // TODO: Needs to be splitted in single filename
    rt << bytes;
    rt.addBinary(data, bytes);

    std::cerr << " ---------------------------------> Sent file to VRB: " << filename.c_str() << std::endl;

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_REMREQ;
    cover->sendVrbMessage(&m);
}

void VRBData::setSelectedPath(std::string path)
{
    this->mFileLocation = "vrb://";
    mFileLocation = mFileLocation + path;
}

std::string VRBData::getSelectedPath()
{
    return this->mFileLocation;
}

void VRBData::reqGlobalLoad(std::string url, int pId)
{
    TokenBuffer rt;
    rt << pId;
    rt << getVRB()->ID();
    rt << TABLET_REQ_GLOBALLOAD;
    rt << url.c_str();

    Message m(rt);
    m.type = COVISE_MESSAGE_VRB_FB_RQ;
    cover->sendVrbMessage(&m);
}
