/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>

#include <util/unixcompat.h>
#include <util/coTabletUIMessages.h>
#include <util/threadname.h>
#include "vvTabletUI.h"
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/message_types.h>
#include <config/CoviseConfig.h>
#include "vvPluginSupport.h"
#include "vvSelectionManager.h"
#include "vvMSController.h"
#include "vvCommunication.h"
#include "vvFileManager.h"
#include "vvPluginList.h"
#include "vvVIVE.h"
#include <iostream>

#ifdef USE_QT
#include "vvTUIFileBrowser/VRBData.h"
#include "vvTUIFileBrowser/LocalData.h"
#include "vvTUIFileBrowser/IRemoteData.h"
#include <qtutil/NetHelp.h>
#ifdef FB_USE_AG
#include "vvTUIFileBrowser/AGData.h"
#endif

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#endif

using namespace covise;
using namespace vive;
//#define FILEBROWSER_DEBUG

vvTUIButton::vvTUIButton(const std::string &n, int pID)
: vvTUIElement(n, pID, TABLET_BUTTON)
{
}

vvTUIButton::vvTUIButton(vvTabletUI *tui, const std::string &n, int pID)
: vvTUIElement(tui, n, pID, TABLET_BUTTON)
{
}

#ifdef USE_QT
vvTUIButton::vvTUIButton(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_BUTTON)
{
}
#endif

vvTUIButton::~vvTUIButton()
{
}

void vvTUIButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIButton::parseMessage: unknown event " << i << endl;
    }
}

#ifdef USE_QT
//TABLET_FILEBROWSER_BUTTON
vvTUIFileBrowserButton::vvTUIFileBrowserButton(const char *n, int pID)
    : vvTUIElement(n, pID, TABLET_FILEBROWSER_BUTTON)
{
    VRBData *locData = new VRBData(this);
    mLocalData = new LocalData(this);
    mData = NULL;
    this->mVRBCId = 0;
    mAGData = NULL;
    mMode = vvTUIFileBrowserButton::OPEN;

#ifdef FB_USE_AG
    AGData *locAGData = new AGData(this);
    mAGData = locAGData;
#endif

    std::string locCurDir = mLocalData->resolveToAbsolute(std::string("."));
    locData->setLocation("127.0.0.1");
    locData->setCurrentPath(locCurDir);
    mLocalData->setCurrentPath(locCurDir);
    Host host;
    std::string shost(host.getAddress());
    this->mId = ID;
    locData->setId(this->mId);
    mLocation = shost;
    mLocalIP = shost;
    mLocalData->setLocation(shost);
    this->mDataObj = locData;
    this->mDataObj->setLocation(shost);

    this->mDataRepo.insert(Data_Pair("vrb", mDataObj));
    this->mDataRepo.insert(Data_Pair("file", mLocalData));
#ifdef FB_USE_AG
    this->mDataRepo.insert(Data_Pair("agtk", mAGData));
#endif

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SET_CURDIR;
    tb << ID;
    std::string path = mDataObj->getCurrentPath();
    tb << path.c_str();

    tui()->send(tb);
}

vvTUIFileBrowserButton::vvTUIFileBrowserButton(vvTabletUI *tui, const char *n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_FILEBROWSER_BUTTON)
{
    VRBData *locData = new VRBData(this);
    mLocalData = new LocalData(this);
    mData = NULL;
    this->mVRBCId = 0;
    mAGData = NULL;
    mMode = vvTUIFileBrowserButton::OPEN;

#ifdef FB_USE_AG
    AGData *locAGData = new AGData(this);
    mAGData = locAGData;
#endif

    std::string locCurDir = mLocalData->resolveToAbsolute(std::string("."));
    locData->setLocation("127.0.0.1");
    locData->setCurrentPath(locCurDir);
    mLocalData->setCurrentPath(locCurDir);
    Host host;
    std::string shost(host.getAddress());
    this->mId = ID;
    locData->setId(this->mId);
    mLocation = shost;
    mLocalIP = shost;
    mLocalData->setLocation(shost);
    this->mDataObj = locData;
    this->mDataObj->setLocation(shost);

    this->mDataRepo.insert(Data_Pair("vrb", mDataObj));
    this->mDataRepo.insert(Data_Pair("file", mLocalData));
#ifdef FB_USE_AG
    this->mDataRepo.insert(Data_Pair("agtk", mAGData));
#endif

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SET_CURDIR;
    tb << ID;
    std::string path = mDataObj->getCurrentPath();
    tb << path.c_str();

    tui->send(tb);
}

vvTUIFileBrowserButton::~vvTUIFileBrowserButton()
{
    this->mFileList.clear();
    this->mDirList.clear();
}

void vvTUIFileBrowserButton::setClientList(const covise::Message &msg)
{
    //transmits a list of vrb clients as received by VRBServer
    //to the filebrowser gui
    TokenBuffer tb(&msg);
    //std::string entry;
    const char *entry = NULL;
    int subtype;
    int size;
    int id;

    tb >> subtype;
    tb >> id;
    tb >> size;

    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_CLIENTS;
    rt << ID;
    rt << size;
    this->mClientList.clear();

    for (int i = 0; i < size; i++)
    {
        tb >> entry;

        this->mClientList.push_back(entry);
    }

    for (int i = 0; i < size; i++)
    {
        tb >> entry;

        std::string locString = (this->mClientList.at(i));
        locString = locString + " - ";
        locString = locString + entry;
        entry = (char *)(this->mClientList.at(i)).c_str();
        rt << locString.c_str();
    }

    tui()->send(rt);
}

void vvTUIFileBrowserButton::parseMessage(TokenBuffer &tb)
{

    //Variable declaration
    int i;
    int locId = 0;

    locId = this->getID();
    tb >> i; //Which event occured?

    std::string backupLocation = this->mLocation; //Backup

    if (i == TABLET_PRESSED)
    {
        if (listener)
            listener->tabletPressEvent(this);
    }
    else if (i == TABLET_RELEASED)
    {
        if (listener)
            listener->tabletReleaseEvent(this);
    }
    else if (i == TABLET_FB_FILE_SEL)
    {
        //File selected for opening in VIVE
        const char *cstrFile = NULL;
        const char *cstrDirectory = NULL;
		bool bLoadAll=false;
        std::string protocol;

        tb >> cstrFile;
        std::string strFile = cstrFile;
        tb >> cstrDirectory;
        tb >> bLoadAll;
        std::string strDirectory = cstrDirectory;

        VRBData *locData = NULL;

        //Retrieve file based upon current location setting
        if (this->mLocation == this->mLocalIP)
        {
            this->getData("file")->setSelectedPath(strDirectory + FILESYS_SEP + strFile);
            mData = this->mLocalData;
            protocol = "file://";
        }
        else if (this->mLocation == "AccessGrid")
        {
            mData = this->mAGData;
            this->getData("agtk")->setSelectedPath(this->mLocation + FILESYS_SEP + strDirectory + FILESYS_SEP + strFile);
            protocol = "agtk://";
        }
        else
        {
            locData = dynamic_cast<VRBData *>(this->getData("vrb"));
            locData->setSelectedPath(this->mLocation + FILESYS_SEP + strDirectory + FILESYS_SEP + strFile);
            protocol = "vrb://";
            mData = locData;
        }

        //Decide whether to load for all partners or just locally
        if (bLoadAll)
        {
            if (!locData)
            {
                locData = dynamic_cast<VRBData *>(this->getData("vrb"));
            }
            //Send new message to indicate that file should be loaded for all
            // 1st case: file://
            //   - Requires url modification to vrb://<localIP>
            // 2nd case agtk:// and vrb://
            //   - Pass url along unmodified
            if (protocol == "file://")
            {
                protocol = "vrb://";
            }
            std::string url = protocol + this->mLocation + FILESYS_SEP + strDirectory + FILESYS_SEP + strFile;

            locData->reqGlobalLoad(url, locId);
        }

        listener->tabletEvent(this);
    }
    else if (i == TABLET_REQ_FILELIST)
    {
        //Retrieve new filelist based on filter and location
        //use instance of IData

        if (this->mLocation == this->mLocalIP)
        {
            this->getData("file")->reqFileList(this->mCurDir, locId);
        }
        else if (this->mLocation == "AccessGrid")
        {
            //Call AG methods here
            if (mAGData)
            {
                this->getData("agtk")->reqFileList(this->mLocation, locId);
            }
            else
            {
                std::cerr << "AccessGrid support currently not available!" << std::endl;
                this->mLocation = backupLocation;
            }
        }
        else
        {
            this->getData("vrb")->reqFileList(this->mCurDir, locId);
        }
    }
    else if (i == TABLET_REQ_DIRLIST)
    {
        //Retrieve new directory based on location
        //use instance of IData

        if (this->mLocation == this->mLocalIP)
        {
            this->getData("file")->reqDirectoryList(this->mCurDir, locId);
        }
        else if (this->mLocation == "AccessGrid")
        {
            //Call AG methods here
            if (mAGData)
            {
                this->getData("agtk")->reqDirectoryList(this->mLocation, locId);
            }
            else
            {
                std::cerr << "AccessGrid support currently not available!" << std::endl;
                this->mLocation = backupLocation;
            }
        }
        else
        {
            this->getData("vrb")->reqDirectoryList(this->mCurDir, locId);
        }
    }
    else if (i == TABLET_REQ_FILTERCHANGE)
    {
        // TUI indicates that the selected filters in the dialog have changed
        // and therefore the content of the filter member in the data objects
        // is adjusted accordingly.
        // This also triggers a refresh of the file list in the filebrowser

        const char *filter = NULL;
        std::string strFilter;

        tb >> filter;
        strFilter = filter;

        if (this->mLocation == this->mLocalIP)
        {
            this->getData("file")->setFilter(strFilter);
            this->getData("file")->reqFileList(this->mCurDir, locId);
        }
        else
        {
            this->getData("vrb")->setFilter(strFilter);
            this->getData("vrb")->reqFileList(this->mCurDir, locId);
        }
    }
    else if (i == TABLET_REQ_DIRCHANGE)
    {
        const char *dir = NULL;
        std::string strDir;

        tb >> dir;
        strDir = dir;

        if (this->mLocation == this->mLocalIP)
        {
            this->getData("file")->setCurrentPath(strDir);
            this->mCurDir = strDir;
            this->getData("file")->reqDirectoryList(strDir, locId);
            this->getData("file")->reqFileList(strDir, locId);
        }
        else if (this->mLocation == "AccessGrid")
        {
            //Call AG methods here
            if (mAGData)
            {
                this->getData("agtk")->setCurrentPath(strDir);
                this->mCurDir = strDir;
                this->getData("agtk")->reqDirectoryList(this->mCurDir, locId);
                this->getData("agtk")->reqFileList(this->mCurDir, locId);
                std::string sdir = strDir;
                this->setCurDir(sdir.c_str());
            }
            else
            {
                std::cerr << "AccessGrid support currently not available!" << std::endl;
                this->mLocation = backupLocation;
            }
        }
        else
        {
            this->getData("vrb")->setCurrentPath(strDir);
            this->mCurDir = strDir;
            this->getData("vrb")->reqDirectoryList(this->mCurDir, locId);
            this->getData("vrb")->reqFileList(this->mCurDir, locId);
        }
        this->setCurDir(dir);
    }
    else if (i == TABLET_REQ_CLIENTS)
    {
        ((VRBData *)this->getData("vrb"))->reqClientList(locId);
    }
    else if (i == TABLET_REQ_MASTER)
    {
        if (vvCommunication::instance()->collaborative())
        {
            //Transmit Master/Slave state to TUI
            TokenBuffer rtb;
            rtb << TABLET_SET_VALUE;
            rtb << TABLET_SET_MASTER;
            rtb << ID;
            rtb << vvCommunication::instance()->isMaster();
            tui()->send(rtb);
        }
    }
    else if (i == TABLET_REQ_LOCATION)
    {
        TokenBuffer rtb;
        rtb << TABLET_SET_VALUE;
        rtb << TABLET_SET_LOCATION;
        rtb << ID;
        rtb << this->mLocation;

#ifdef FILEBROWSER_DEBUG
        std::cerr << "Host to be used for file lists: = " << this->mLocation.c_str() << std::endl;
#endif

        tui()->send(rtb);
    }
    else if (i == TABLET_SET_LOCATION)
    {
        const char *location = NULL;
        tb >> location;
#ifdef FILEBROWSER_DEBUG
        std::cerr << "Setting new location!" << std::endl;
        std::cerr << " New location = " << location << std::endl;
#endif
        NetHelp net;
        std::string slocation = net.GetIpAddress(location).toStdString();
        this->mDataObj->setLocation(slocation);
        this->mLocation = slocation;

        this->setCurDir("/");

        if (this->mLocation == "AccessGrid")
        {
            if (mAGData)
            {
                this->mAGData->setCurrentPath("");
                this->mCurDir = "";
                this->mAGData->reqDirectoryList("", locId);
                this->mAGData->reqFileList("", locId);
                this->setCurDir("");
            }
            else
            {
                std::cerr << "AccessGrid support currently not available!" << std::endl;
                this->mLocation = backupLocation;
            }
        }
    }
    else if (i == TABLET_REQ_HOME)
    {
        if (this->mLocation == this->mLocalIP)
        {
            ((LocalData *)this->getData("file"))->setHomeDir();
            ((LocalData *)this->getData("file"))->reqHomeDir(locId);
            ((LocalData *)this->getData("file"))->reqHomeFiles(locId);
        }
        else if (this->mLocation == "AccessGrid")
        {
            //Call AG methods here
            if (mAGData)
            {
                this->getData("agtk")->reqHomeDir(locId);
                this->getData("agtk")->reqHomeFiles(locId);
            }
            else
            {
                std::cerr << "AccessGrid support currently not available!" << std::endl;
                this->mLocation = backupLocation;
            }
        }
        else
        {
            this->getData("vrb")->reqHomeDir(locId);
            this->getData("vrb")->reqHomeFiles(locId);
        }
    }
    else if (i == TABLET_FB_PATH_SELECT)
    {
        std::string protocol;
        VRBData *locData = NULL;
        const char *location = NULL;
        const char *file = NULL;
        tb >> location;
        tb >> file;

        std::string sLocation = location;
        std::string sFile = file;

        if (this->mLocation == this->mLocalIP)
        {
            this->getData("file")->setSelectedPath(sLocation + FILESYS_SEP + sFile);
            mData = this->mLocalData;
            protocol = "file://";
        }
        else if (this->mLocation == "AccessGrid")
        {
            mData = this->mAGData;
            this->getData("agtk")->setSelectedPath(this->mLocation + FILESYS_SEP + sLocation + FILESYS_SEP + sFile);
            protocol = "agtk://";
        }
        else
        {
            locData = dynamic_cast<VRBData *>(this->getData("vrb"));
            locData->setSelectedPath(this->mLocation + FILESYS_SEP + sLocation + FILESYS_SEP + sFile);
            protocol = "vrb://";
            mData = locData;
        }

        this->listener->tabletEvent(this);
    }
    else if (i == TABLET_REQ_DRIVES)
    {
        if (this->mLocation == this->mLocalIP)
        {
            this->getData("file")->reqDrives(locId);
        }
        else if (this->mLocation == "AccessGrid")
        {
            //Call AG methods here
            if (mAGData)
            {
                this->getData("agtk")->reqHomeDir(locId);
                this->getData("agtk")->reqHomeFiles(locId);
            }
            else
            {
                std::cerr << "AccessGrid support currently not available!" << std::endl;
                this->mLocation = backupLocation;
            }
        }
        else
        {
            this->getData("vrb")->reqDrives(locId);
        }
    }
    else if (i == TABLET_REQ_VRBSTAT)
    {
        // Signals the availability of a VRB server to file dialog
        // used for enabling of RemoteClient button in FileBrowser
        TokenBuffer rtb;
        rtb << TABLET_SET_VALUE;
        rtb << TABLET_SET_VRBSTAT;
        rtb << ID;
        rtb << static_cast<int>(vvVIVE::instance()->isVRBconnected());

        tui()->send(rtb);
    }
    else
    {
        cerr << "vvTUIFileBrowserButton::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIFileBrowserButton::resend(bool create)
{
    vvTUIElement::resend(create);
    TokenBuffer rt;

    // Send current Directory
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_CURDIR;
    rt << ID;
    rt << this->getData("vrb")->getCurrentPath().c_str();
    //std::cerr << "Resend: Current directory: " << this->mDataObj->getCurrentPath().c_str() << std::endl;
    tui()->send(rt);

    rt.reset();
    //Send FileList
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_FILELIST;
    rt << ID;
    rt << static_cast<int>(this->mFileList.size());
    //std::cerr << "Resend: Set FileList: " << std::endl;
    //this->mFileList.clear(); // don't delete file list before sending it to client (TUI)

    for (size_t i = 0; i < this->mFileList.size(); i++)
    {
        std::string sfl = this->mFileList.at(i);
        rt << sfl.c_str();
        //std::cerr << "Resend: FileList entry #" << i << " = " << sfl.c_str() << std::endl;
    }
    tui()->send(rt);

    rt.reset();
    //Send DirList
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_DIRLIST;
    rt << ID;
    rt << static_cast<int>(this->mDirList.size());
    //this->mDirList.clear(); //don't delete directory list before sending it to client (TUI)

    //std::cerr << "Resend: Set DirList: " << std::endl;

    for (size_t i = 0; i < this->mDirList.size(); i++)
    {
        std::string sdl = mDirList.at(i);
        rt << sdl.c_str();
        //std::cerr << "Resend: DirList entry #" << i << " = " << sdl.c_str() << std::endl;
    }
    tui()->send(rt);

    //Send DirList
    rt.reset();
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_MODE;
    rt << ID;
    rt << (int)this->mMode;

    tui()->send(rt);

    rt.reset();
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_FILTERLIST;
    rt << ID;
    rt << mFilterList.c_str();
    tui()->send(rt);
}

void vvTUIFileBrowserButton::setFileList(const covise::Message &msg)
{
    TokenBuffer tb(&msg);
    //std::string entry;
    const char *entry = NULL;
    int subtype;
    int size;
    int id;

    tb >> subtype;
    tb >> id;
    tb >> size;

    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_FILELIST;
    rt << ID;
    rt << size;
    this->mFileList.clear();

    for (int i = 0; i < size; i++)
    {
        tb >> entry;

        rt << entry;
        this->mFileList.push_back(entry);
    }

    tui()->send(rt);
}

IData *vvTUIFileBrowserButton::getData(std::string protocol)
{

    if (protocol.compare("") != 0)
    {
        return mDataRepo[protocol];
    }

    return mData;
}

IData *vvTUIFileBrowserButton::getVRBData()
{
    return this->getData("vrb");
}

void vvTUIFileBrowserButton::setDirList(const covise::Message &msg)
{
    TokenBuffer tb(&msg);
    const char *entry;
    int subtype;
    int size;
    int id;

    tb >> subtype;
    tb >> id;
    tb >> size;

    this->mDirList.clear();

    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_DIRLIST;
    rt << ID;
    rt << size;

    for (int i = 0; i < size; i++)
    {
        tb >> entry;

        rt << entry;
        this->mDirList.push_back(entry);
    }

    tui()->send(rt);
}

void vvTUIFileBrowserButton::setDrives(const Message &ms)
{
    TokenBuffer tb(&ms);
    const char *entry;
    int subtype;
    int size;
    int id;

    tb >> subtype;
    tb >> id;
    tb >> size;

    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_DRIVES;
    rt << ID;
    rt << size;

    for (int i = 0; i < size; i++)
    {
        tb >> entry;
        rt << entry;
    }

    tui()->send(rt);
}

void vvTUIFileBrowserButton::setCurDir(const covise::Message &msg)
{
    TokenBuffer tb(&msg);
    int subtype = 0;
    int id = 0;
    const char *dirValue = NULL;

    tb >> subtype;
    tb >> id;
    tb >> dirValue;

    //Save current path to Data object
    this->setCurDir(dirValue);
}

void vvTUIFileBrowserButton::setCurDir(const char *dir)
{
    //Save current path to Data object
    std::string sdir = std::string(dir);
    if ((sdir.compare("") || sdir.compare(".")) && (this->mLocation == this->mLocalIP))
    {
        sdir = this->mLocalData->resolveToAbsolute(std::string(dir));
        QFileInfo fi(sdir.c_str());
        if(fi.isFile())
            sdir = fi.absoluteDir().absolutePath().toStdString();
#ifdef FILEBROWSER_DEBUG
        std::cerr << "Adjusted current directory!" << std::endl;
#endif
    }
    this->mCurDir = sdir.c_str();
    this->mDataObj->setCurrentPath(string(sdir.c_str()));
    this->mLocalData->setCurrentPath(string(sdir.c_str()));

    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_CURDIR;
    rt << ID;
    rt << sdir.c_str();

#ifdef FILEBROWSER_DEBUG
    std::cerr << "Sent parsed message to TUI!" << std::endl;
    std::cerr << "Contains path = " << sdir.c_str() << std::endl;
#endif

    tui()->send(rt);
}

void vvTUIFileBrowserButton::sendList(TokenBuffer & /*tb*/)
{
}

std::string vvTUIFileBrowserButton::getFilename(const std::string url)
{
    IData *locData = NULL;

    std::string::size_type pos = 0;
    pos = url.find(':');
    std::string protocol = url.substr(0, pos);

    locData = mDataRepo[protocol];

    if (locData)
    {
        return locData->getTmpFilename(url, this->getID());
    }
    else
        locData = mDataRepo["file"];
    if (locData)
    {
        return locData->getTmpFilename(url, this->getID());
    }
    return std::string("");
}

void *vvTUIFileBrowserButton::getFileHandle(bool sync)
{
    if (mData)
    {
        return this->mData->getTmpFileHandle(sync);
    }
    return NULL;
}

void vvTUIFileBrowserButton::setMode(DialogMode mode)
{
    mMode = mode;
    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_MODE;
    rt << ID;
    rt << (int)mMode;

    tui()->send(rt);
}

// Method which is called from external of vvTabletUI to allow
// VIVE to initially set the range of available filter extensions
// used in the file dialog in the TUI.
void vvTUIFileBrowserButton::setFilterList(std::string filterList)
{
    mFilterList = filterList;
    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_FILTERLIST;
    rt << ID;
    rt << filterList.c_str();

    tui()->send(rt);
}

std::string vvTUIFileBrowserButton::getSelectedPath()
{
    return this->mData->getSelectedPath();
}
#else
//TABLET_FILEBROWSER_BUTTON
vvTUIFileBrowserButton::vvTUIFileBrowserButton(const char *n, int pID): vvTUIElement(n, pID, TABLET_FILEBROWSER_BUTTON)
{
    mLocalData = nullptr;
    mData = NULL;
    this->mVRBCId = 0;
    mAGData = NULL;
    mMode = vvTUIFileBrowserButton::OPEN;

    this->mId = ID;
    mLocation = "";
    mLocalIP = "";
    this->mDataObj = nullptr;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SET_CURDIR;
    tb << ID;
    std::string path;
    tb << path.c_str();

    tui()->send(tb);
}

vvTUIFileBrowserButton::vvTUIFileBrowserButton(vvTabletUI *tui, const char *n, int pID)
: vvTUIElement(tui, n, pID, TABLET_FILEBROWSER_BUTTON)
{
    mLocalData = nullptr;
    mData = NULL;
    this->mVRBCId = 0;
    mAGData = NULL;
    mMode = vvTUIFileBrowserButton::OPEN;

    this->mId = ID;
    mLocation = "";
    mLocalIP = "";
    this->mDataObj = nullptr;


    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SET_CURDIR;
    tb << ID;
    std::string path;
    tb << path.c_str();

    tui->send(tb);
}

vvTUIFileBrowserButton::~vvTUIFileBrowserButton()
{
}

void vvTUIFileBrowserButton::setClientList(const covise::Message &msg)
{
}

void vvTUIFileBrowserButton::parseMessage(TokenBuffer &tb)
{
}

void vvTUIFileBrowserButton::resend(bool create)
{
}

void vvTUIFileBrowserButton::setFileList(const covise::Message &msg)
{
}

IData *vvTUIFileBrowserButton::getData(std::string protocol)
{
    return nullptr;
}

IData *vvTUIFileBrowserButton::getVRBData()
{
    return nullptr;
}

void vvTUIFileBrowserButton::setDirList(const covise::Message &msg)
{
}

void vvTUIFileBrowserButton::setDrives(const Message &ms)
{
}

void vvTUIFileBrowserButton::setCurDir(const covise::Message &msg)
{
}

void vvTUIFileBrowserButton::setCurDir(const char *dir)
{
}

void vvTUIFileBrowserButton::sendList(TokenBuffer & /*tb*/)
{
}

std::string vvTUIFileBrowserButton::getFilename(const std::string url)
{
    return std::string("");
}

void *vvTUIFileBrowserButton::getFileHandle(bool sync)
{
    return NULL;
}

void vvTUIFileBrowserButton::setMode(DialogMode mode)
{
    mMode = mode;
    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_MODE;
    rt << ID;
    rt << (int)mMode;

    tui()->send(rt);
}

// Method which is called from external of vvTabletUI to allow
// VIVE to initially set the range of available filter extensions
// used in the file dialog in the TUI.
void vvTUIFileBrowserButton::setFilterList(std::string filterList)
{
    mFilterList = filterList;
    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_FILTERLIST;
    rt << ID;
    rt << filterList.c_str();

    tui()->send(rt);
}

std::string vvTUIFileBrowserButton::getSelectedPath()
{
    return "";
}
#endif

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIColorTriangle::vvTUIColorTriangle(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_COLOR_TRIANGLE)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
}

#ifdef USE_QT
vvTUIColorTriangle::vvTUIColorTriangle(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_COLOR_TRIANGLE)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
}
#endif

vvTUIColorTriangle::~vvTUIColorTriangle()
{
}

void vvTUIColorTriangle::parseMessage(TokenBuffer &tb)
{
    int i, j;
    tb >> i;
    tb >> j;

    if (i == TABLET_RGBA)
    {
        tb >> red;
        tb >> green;
        tb >> blue;

        if (j == TABLET_RELEASED)
        {
#ifdef USE_QT
            emit tabletReleaseEvent();
#endif
            if (listener)
                listener->tabletReleaseEvent(this);
        }
        if (j == TABLET_PRESSED)
        {
#ifdef USE_QT
            emit tabletEvent();
#endif
            if (listener)
                listener->tabletEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIColorTriangle::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIColorTriangle::setColor(float r, float g, float b)
{
    red = r;
    green = g;
    blue = b;
    setVal(TABLET_RED, r);
    setVal(TABLET_GREEN, g);
    setVal(TABLET_BLUE, b);
}

void vvTUIColorTriangle::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(TABLET_RED, red);
    setVal(TABLET_GREEN, green);
    setVal(TABLET_BLUE, blue);
}

vvTUIColorButton::vvTUIColorButton(const std::string &n, int pID): vvTUIElement(n, pID, TABLET_COLOR_BUTTON)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

vvTUIColorButton::vvTUIColorButton(vvTabletUI *tui, const std::string &n, int pID)
: vvTUIElement(tui, n, pID, TABLET_COLOR_BUTTON)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

#ifdef USE_QT
vvTUIColorButton::vvTUIColorButton(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_COLOR_TRIANGLE)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}
#endif

vvTUIColorButton::~vvTUIColorButton()
{
}

void vvTUIColorButton::parseMessage(TokenBuffer &tb)
{
    int i, j;
    tb >> i;
    tb >> j;

    if (i == TABLET_RGBA)
    {
        tb >> red;
        tb >> green;
        tb >> blue;
        tb >> alpha;

        if (j == TABLET_RELEASED)
        {
#ifdef USE_QT
            emit tabletEvent();
#endif
            if (listener)
                listener->tabletReleaseEvent(this);
        }
        if (j == TABLET_PRESSED)
        {
#ifdef USE_QT
            emit tabletReleaseEvent();
#endif
            if (listener)
                listener->tabletEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIColorButton::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIColorButton::setColor(float r, float g, float b, float a)
{
    red = r;
    green = g;
    blue = b;
    alpha = a;
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_RGBA;
    t << ID;
    t << red;
    t << green;
    t << blue;
    t << alpha;
    tui()->send(t);
}

void vvTUIColorButton::resend(bool create)
{
    vvTUIElement::resend(create);

    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_RGBA;
    t << ID;
    t << red;
    t << green;
    t << blue;
    t << alpha;
    tui()->send(t);
}

#ifdef USE_QT
//----------------------------------------------------------
//----------------------------------------------------------

vvTUIColorTab::vvTUIColorTab(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_COLOR_TAB)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

vvTUIColorTab::vvTUIColorTab(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_COLOR_TAB)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

vvTUIColorTab::~vvTUIColorTab()
{
}

void vvTUIColorTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_RGBA)
    {
        tb >> red;
        tb >> green;
        tb >> blue;
        tb >> alpha;

#ifdef USE_QT
        emit tabletEvent();
#endif
        if (listener)
            listener->tabletEvent(this);
    }
    else
    {
        cerr << "vvTUIColorTab::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIColorTab::setColor(float r, float g, float b, float a)
{
    red = r;
    green = g;
    blue = b;
    alpha = a;

    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_RGBA;
    t << ID;
    t << r;
    t << g;
    t << b;
    t << a;
    tui()->send(t);
}

void vvTUIColorTab::resend(bool create)
{
    vvTUIElement::resend(create);
    setColor(red, green, blue, alpha);
}
#endif

//----------------------------------------------------------
//---------------------------------------------------------

vvTUINav::vvTUINav(vvTabletUI *tui, const char *n, int pID): vvTUIElement(tui, n, pID, TABLET_NAV_ELEMENT)
{
    down = false;
    x = 0;
    y = 0;
}

vvTUINav::~vvTUINav()
{
}

void vvTUINav::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
        tb >> x;
        tb >> y;
        if (listener)
            listener->tabletPressEvent(this);
        down = true;
    }
    else if (i == TABLET_RELEASED)
    {
        tb >> x;
        tb >> y;
        if (listener)
            listener->tabletReleaseEvent(this);
        down = false;
    }
    else if (i == TABLET_POS)
    {
        tb >> x;
        tb >> y;
        if (listener)
            listener->tabletEvent(this);
    }
    else
    {
        cerr << "vvTUINav::parseMessage: unknown event " << i << endl;
    }
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIBitmapButton::vvTUIBitmapButton(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_BITMAP_BUTTON)
{
}

vvTUIBitmapButton::vvTUIBitmapButton(vvTabletUI *tui, const std::string &n, int pID)
: vvTUIElement(tui, n, pID, TABLET_BITMAP_BUTTON)
{
}

#ifdef USE_QT
vvTUIBitmapButton::vvTUIBitmapButton(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_BITMAP_BUTTON)
{
}
#endif

vvTUIBitmapButton::~vvTUIBitmapButton()
{
}

void vvTUIBitmapButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIBitmapButton::parseMessage: unknown event " << i << endl;
    }
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUILabel::vvTUILabel(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_TEXT_FIELD)
{
    color = Qt::black;
}

vvTUILabel::vvTUILabel(vvTabletUI *tui, const std::string &n, int pID)
: vvTUIElement(tui, n, pID, TABLET_TEXT_FIELD)
{
    color = Qt::black;
}

#ifdef USE_QT
vvTUILabel::vvTUILabel(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_TEXT_FIELD)
{
    color = Qt::black;
}
#endif

vvTUILabel::~vvTUILabel()
{
}

void vvTUILabel::resend(bool create)
{
    vvTUIElement::resend(create);
    setColor(color);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUITabFolder::vvTUITabFolder(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_TAB_FOLDER)
{
}

vvTUITabFolder::vvTUITabFolder(vvTabletUI *tui, const std::string &n, int pID)
: vvTUIElement(tui, n, pID, TABLET_TAB_FOLDER)
{
}

#ifdef USE_QT
vvTUITabFolder::vvTUITabFolder(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_TAB_FOLDER)
{
}
#endif

vvTUITabFolder::~vvTUITabFolder()
{
}

void vvTUITabFolder::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUITabFolder::parseMessage: unknown event " << i << endl;
    }
}


#ifdef USE_QT
//----------------------------------------------------------
//----------------------------------------------------------

vvTUIUITab::vvTUIUITab(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_UI_TAB)
{
}

vvTUIUITab::vvTUIUITab(vvTabletUI *tui, const std::string &n, int pID)
: vvTUIElement(tui, n, pID, TABLET_UI_TAB)
{

}

vvTUIUITab::vvTUIUITab(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_UI_TAB)
{
}

vvTUIUITab::~vvTUIUITab()
{
}

void vvTUIUITab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        emit tabletEvent();
        emit tabletPressEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
        emit tabletEvent();
        emit tabletReleaseEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else if (i == TABLET_UI_COMMAND)
    {
        std::string target;
        uint64_t commandSize;

        QString command;

        tb >> target;
        tb >> commandSize;
        command = QString::fromUtf16((const ushort *)tb.getBinary(commandSize));

        emit tabletUICommand(QString::fromStdString(target), command);
    }
    else
    {
        cerr << "vvTUIUITab::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIUITab::sendEvent(const QString &source, const QString &event)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_UI_COMMAND;
    tb << ID;
    tb << source.toStdString();
    tb << (int)(event.size() + 1) * 2;
    tb.addBinary((const char *)event.utf16(), (event.size() + 1) * 2);

    tui()->send(tb);
}

bool vvTUIUITab::loadUIFile(const std::string &filename)
{
    QFile uiFile(QString::fromStdString(filename));

    if (!uiFile.exists())
    {
        std::cerr << "vvTUIUITab::loadFile err: file " << filename << " does not exist" << std::endl;
        return false;
    }

    if (!uiFile.open(QIODevice::ReadOnly))
    {
        std::cerr << "vvTUIUITab::loadFile err: cannot open file " << filename << std::endl;
        return false;
    }

    QTextStream inStream(&uiFile);

    this->uiDescription = inStream.readAll();

    QString jsFileName(QString::fromStdString(filename) + ".js");
    QFile jsFile(jsFileName);
    if (jsFile.exists() && jsFile.open(QIODevice::ReadOnly))
    {
        vive::vvFileManager::instance()->loadFile(jsFileName.toLocal8Bit().data());
    }

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_UI_USE_DESCRIPTION;
    tb << ID;
    tb << (int)(this->uiDescription.size() + 1) * 2;
    tb.addBinary((const char *)this->uiDescription.utf16(), (this->uiDescription.size() + 1) * 2);

    tui()->send(tb);

    return true;
}
#endif

//----------------------------------------------------------
//----------------------------------------------------------

vvTUITab::vvTUITab(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_TAB)
{
}

vvTUITab::vvTUITab(vvTabletUI *tui, const std::string &n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_TAB)
{

}

#ifdef USE_QT
vvTUITab::vvTUITab(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_TAB)
{
}
#endif

vvTUITab::~vvTUITab()
{
}

void vvTUITab::allowRelayout(bool rl)
{
    m_allowRelayout = rl;
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_RELAYOUT;
    tb << ID;
    tb << m_allowRelayout;

    tui()->send(tb);
}

void vvTUITab::resend(bool create)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_RELAYOUT;
    tb << ID;
    tb << m_allowRelayout;

    vvTUIElement::resend(create);
    tui()->send(tb);
}

void vvTUITab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUITab::parseMessage: unknown event " << i << endl;
    }
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIAnnotationTab::vvTUIAnnotationTab(const char *n, int pID)
    : vvTUIElement(n, pID, TABLET_ANNOTATION_TAB)
{
}

vvTUIAnnotationTab::~vvTUIAnnotationTab()
{
}

void vvTUIAnnotationTab::parseMessage(TokenBuffer &tb)
{
    listener->tabletDataEvent(this, tb);
}

void vvTUIAnnotationTab::setNewButtonState(bool state)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ANNOTATION_CHANGE_NEW_BUTTON_STATE;
    tb << ID;
    tb << (char)state;

    tui()->send(tb);
}

void vvTUIAnnotationTab::addAnnotation(int id)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ANNOTATION_NEW;
    tb << ID;
    tb << id;

    tui()->send(tb);
}

void vvTUIAnnotationTab::deleteAnnotation(int mode, int id)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ANNOTATION_DELETE;
    tb << ID;
    tb << mode;
    tb << id;

    tui()->send(tb);
}

void vvTUIAnnotationTab::setSelectedAnnotation(int id)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ANNOTATION_SET_SELECTION;
    tb << ID;
    tb << id;

    tui()->send(tb);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIFunctionEditorTab::vvTUIFunctionEditorTab(const char *tabName, int pID)
    : vvTUIElement(tabName, pID, TABLET_FUNCEDIT_TAB)
{
    tfDim = 1;
    histogramData = NULL;
}

vvTUIFunctionEditorTab::~vvTUIFunctionEditorTab()
{
    if (histogramData)
        delete[] histogramData;
}

int vvTUIFunctionEditorTab::getDimension() const
{
    return tfDim;
    ;
}

void vvTUIFunctionEditorTab::setDimension(int dim)
{
    tfDim = dim;
}

void vvTUIFunctionEditorTab::resend(bool create)
{
    vvTUIElement::resend(create);

    //resend the transfer function information
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TF_WIDGET_LIST;
    tb << ID;

    //send dimension
    int dim = getDimension();
    tb << dim;

    switch (dim)
    {
    case 1:
    {
        tb << (uint32_t)colorPoints.size();
        for (uint32_t i = 0; i < colorPoints.size(); ++i)
        {
            tb << colorPoints[i].r;
            tb << colorPoints[i].g;
            tb << colorPoints[i].b;
            tb << colorPoints[i].x;
        }

        // then, alpha widgets
        tb << (uint32_t)alphaPoints.size();
        for (uint32_t i = 0; i < alphaPoints.size(); ++i)
        {
            tb << alphaPoints[i].kind;

            switch (alphaPoints[i].kind)
            {
            case TF_PYRAMID:
            {
                tb << alphaPoints[i].alpha;
                tb << alphaPoints[i].xPos;
                tb << alphaPoints[i].xParam1; //xb
                tb << alphaPoints[i].xParam2; //xt
            }
            break;

            case TF_FREE:
            {
                tb << (uint32_t)alphaPoints[i].additionalDataElems; //data lenght;

                //every elem has a position and an alpha value
                for (int j = 0; j < alphaPoints[i].additionalDataElems * 2; j += 2)
                {
                    tb << alphaPoints[i].additionalData[j];
                    tb << alphaPoints[i].additionalData[j + 1];
                }
            }
            break;
            default:
                break;
            }
        }
    }
    break;

    case 2:
    {
        tb << (uint32_t)colorPoints.size();
        for (uint32_t i = 0; i < colorPoints.size(); ++i)
        {
            tb << colorPoints[i].r;
            tb << colorPoints[i].g;
            tb << colorPoints[i].b;
            tb << colorPoints[i].x;
            tb << colorPoints[i].y;
        }

        // then, alpha widgets
        tb << (uint32_t)alphaPoints.size();
        for (uint32_t i = 0; i < alphaPoints.size(); ++i)
        {
            tb << alphaPoints[i].kind;

            switch (alphaPoints[i].kind)
            {
            case TF_PYRAMID:
            {
                tb << alphaPoints[i].alpha;
                tb << alphaPoints[i].xPos;
                tb << alphaPoints[i].xParam1; //xb
                tb << alphaPoints[i].xParam2; //xt
                tb << alphaPoints[i].yPos;
                tb << alphaPoints[i].yParam1; //xb
                tb << alphaPoints[i].yParam2; //xt

                tb << alphaPoints[i].ownColor;
                if (alphaPoints[i].ownColor)
                {
                    tb << alphaPoints[i].r;
                    tb << alphaPoints[i].g;
                    tb << alphaPoints[i].b;
                }
            }
            break;

            case TF_BELL:
            {
                tb << alphaPoints[i].alpha;
                tb << alphaPoints[i].xPos;
                tb << alphaPoints[i].xParam1; //xb
                tb << alphaPoints[i].yPos;
                tb << alphaPoints[i].yParam1; //xb

                tb << alphaPoints[i].ownColor;
                if (alphaPoints[i].ownColor)
                {
                    tb << alphaPoints[i].r;
                    tb << alphaPoints[i].g;
                    tb << alphaPoints[i].b;
                }
            }
            break;

            case TF_CUSTOM_2D:
            {
                tb << alphaPoints[i].alpha;
                tb << alphaPoints[i].alpha; //alpha2
                tb << alphaPoints[i].xPos;
                tb << alphaPoints[i].yPos;
                //tb << extrude; //TODO!
                tb << 1;

                tb << alphaPoints[i].ownColor;
                if (alphaPoints[i].ownColor)
                {
                    tb << alphaPoints[i].r;
                    tb << alphaPoints[i].g;
                    tb << alphaPoints[i].b;
                }

                tb << alphaPoints[i].additionalDataElems;
                for (int j = 0; j < alphaPoints[i].additionalDataElems * 3; j += 3)
                {
                    tb << alphaPoints[i].additionalData[j];
                    tb << alphaPoints[i].additionalData[j + 1];
                    tb << alphaPoints[i].additionalData[j + 2];
                }
            }
            break;

            case TF_MAP:
            {
                tb << alphaPoints[i].alpha;
                tb << alphaPoints[i].xPos;
                tb << alphaPoints[i].xParam1; //xb
                tb << alphaPoints[i].yPos;
                tb << alphaPoints[i].yParam1; //xb

                tb << alphaPoints[i].ownColor;
                if (alphaPoints[i].ownColor)
                {
                    tb << alphaPoints[i].r;
                    tb << alphaPoints[i].g;
                    tb << alphaPoints[i].b;
                }

                // store map info
                tb << alphaPoints[i].additionalDataElems;
                for (int j = 0; j < alphaPoints[i].additionalDataElems; ++j)
                {
                    tb << alphaPoints[i].additionalData[j];
                }
            }
            break;

            default:
                break;
            }
        }
    }
    break;

    default:
        // we do not handle higher dimension for now
        break;
    }

    // send TFE info
    tui()->send(tb);

    // send histogram
    sendHistogramData();
}

void vvTUIFunctionEditorTab::sendHistogramData()
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TF_HISTOGRAM;
    tb << ID;

    tb << tfDim;

    if (tfDim == 1)
    {
        if (histogramData == NULL)
            tb << 0;
        else
        {
            tb << (uint32_t)histogramBuckets;
            for (uint32_t i = 0; i < histogramBuckets; ++i)
            {
                tb << histogramData[i];
            }
        }
    }
    else //if (tfDim == 2) send anyway for volume dimensions > 2
    {
        if (histogramData == NULL)
        {
            tb << 0;
            tb << 0;
        }
        else
        {
            tb << (uint32_t)histogramBuckets;
            tb << (uint32_t)histogramBuckets;
            for (uint32_t i = 0; i < histogramBuckets * histogramBuckets; ++i)
            {
                tb << histogramData[i];
            }
        }
    }

    tui()->send(tb);
}

void vvTUIFunctionEditorTab::parseMessage(TokenBuffer &tb)
{
    int type;
    tb >> type;

    if (type == TABLET_TF_WIDGET_LIST)
    {
        // get TF size (1 = 1D, 2 = 2D)
        tb >> tfDim;

        // if the dimensionality is different (do not match that of data)
        // adjust that.

        // This is done in VolumePlugin::tabletPressEvent

        int numPoints;
        colorPoints.clear();
        for (size_t i = 0; i < alphaPoints.size(); ++i)
            if (alphaPoints[i].additionalDataElems > 0)
                delete[] alphaPoints[i].additionalData;

        alphaPoints.clear();

        if (tfDim == 1)
        {
            //1) color points
            tb >> numPoints;
            for (int i = 0; i < numPoints; ++i)
            {
                // for each entry: r, g, b channels (float), pos (float)
                // but the updateColorMap function expects rgbax, so lets
                // add an opaque alpha component. We deal with alpha below

                colorPoint cp;

                tb >> cp.r;
                tb >> cp.g;
                tb >> cp.b;
                tb >> cp.x;
                cp.y = -1.0f;

                colorPoints.push_back(cp);
            }

            // 2) the alpha widgets
            tb >> numPoints;

            for (int i = 0; i < numPoints; ++i)
            {
                int widgetType;
                //TF_PYRAMID == 1, TF_CUSTOM == 4
                //
                tb >> widgetType;
                switch (widgetType)
                {
                case TF_PYRAMID:
                {
                    alphaPoint ap;
                    ap.kind = widgetType;
                    tb >> ap.alpha;
                    tb >> ap.xPos;
                    tb >> ap.xParam1;
                    tb >> ap.xParam2;
                    ap.yPos = -1.0f;
                    ap.additionalDataElems = 0;
                    ap.additionalData = NULL;
                    alphaPoints.push_back(ap);
                }
                break;

                case TF_FREE:
                {
                    alphaPoint ap;
                    ap.kind = widgetType;
                    ap.alpha = 1.0f;
                    ap.xPos = 0.5f;
                    ap.xParam1 = 1.0f;
                    ap.xParam2 = 1.0f;
                    ap.yPos = -1.0f;
                    tb >> ap.additionalDataElems;

                    // each "element" has 2 components (pos and alpha)
                    if (ap.additionalDataElems > 0)
                    {
                        ap.additionalData = new float[ap.additionalDataElems * 2];
                        for (int j = 0; j < ap.additionalDataElems * 2; j += 2)
                        {
                            float x, alpha;
                            tb >> x; //pos
                            tb >> alpha; //alpha value;
                            ap.additionalData[j] = x;
                            ap.additionalData[j + 1] = alpha;
                        }
                    }
                    else
                    {
                        ap.additionalData = NULL;
                    }
                    alphaPoints.push_back(ap);
                }
                break;
                }
            }
        }
        else //dim == 2
        {
            //1) color points
            tb >> numPoints;
            for (int i = 0; i < numPoints; ++i)
            {
                // for each entry: r, g, b channels (float), pos (float)
                // but the updateColorMap function expects rgbax, so lets
                // add an opaque alpha component. We deal with alpha below

                colorPoint cp;

                tb >> cp.r;
                tb >> cp.g;
                tb >> cp.b;
                tb >> cp.x;
                tb >> cp.y;

                colorPoints.push_back(cp);
            }

            // 2) the alpha widgets
            tb >> numPoints;

            for (int i = 0; i < numPoints; ++i)
            {
                int widgetType;
                //TF_PYRAMID == 1, TF_CUSTOM == 4
                //
                tb >> widgetType;
                switch (widgetType)
                {
                case TF_PYRAMID:
                {
                    alphaPoint ap;
                    ap.kind = widgetType;
                    tb >> ap.alpha;
                    tb >> ap.xPos;
                    tb >> ap.xParam1;
                    tb >> ap.xParam2;
                    tb >> ap.yPos;
                    tb >> ap.yParam1;
                    tb >> ap.yParam2;

                    tb >> ap.ownColor;
                    if (ap.ownColor)
                    {
                        tb >> ap.r;
                        tb >> ap.g;
                        tb >> ap.b;
                    }

                    ap.additionalDataElems = 0;
                    ap.additionalData = NULL;
                    alphaPoints.push_back(ap);
                }
                break;

                case TF_BELL:
                {
                    alphaPoint ap;
                    ap.kind = widgetType;
                    tb >> ap.alpha;
                    tb >> ap.xPos;
                    tb >> ap.xParam1;
                    tb >> ap.yPos;
                    tb >> ap.yParam1;

                    tb >> ap.ownColor;
                    if (ap.ownColor)
                    {
                        tb >> ap.r;
                        tb >> ap.g;
                        tb >> ap.b;
                    }

                    ap.additionalDataElems = 0;
                    ap.additionalData = NULL;
                    alphaPoints.push_back(ap);
                }
                break;

                case TF_CUSTOM_2D:
                    assert(false && "TODO!");
                    break;

                case TF_MAP:
                {
                    alphaPoint ap;
                    ap.kind = widgetType;
                    tb >> ap.alpha;
                    tb >> ap.xPos;
                    tb >> ap.xParam1;
                    tb >> ap.yPos;
                    tb >> ap.yParam1;

                    tb >> ap.ownColor;
                    if (ap.ownColor)
                    {
                        tb >> ap.r;
                        tb >> ap.g;
                        tb >> ap.b;
                    }

                    // store map info
                    tb >> ap.additionalDataElems;
                    if (ap.additionalDataElems > 0)
                    {
                        ap.additionalData = new float[ap.additionalDataElems];

                        for (int j = 0; j < ap.additionalDataElems; ++j)
                            tb >> ap.additionalData[j];
                    }
                    else
                    {
                        ap.additionalData = NULL;
                    }

                    alphaPoints.push_back(ap);
                }
                break;
                }
            }
        }
    }

    listener->tabletPressEvent(this);
}

//----------------------------------------------------------
//----------------------------------------------------------
vvTUISplitter::vvTUISplitter(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_SPLITTER)
{
    shape = vvTUIFrame::StyledPanel;
    style = vvTUIFrame::Sunken;
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}

#ifdef USE_QT
vvTUISplitter::vvTUISplitter(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_SPLITTER)
{
    shape = vvTUIFrame::StyledPanel;
    style = vvTUIFrame::Sunken;
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}
#endif

vvTUISplitter::~vvTUISplitter()
{
}

void vvTUISplitter::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUISplitter::parseMessage: unknown event " << i << endl;
    }
}

void vvTUISplitter::resend(bool create)
{
    vvTUIElement::resend(create);
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}

void vvTUISplitter::setShape(int s)
{
    TokenBuffer tb;
    shape = s;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SHAPE;
    tb << ID;
    tb << shape;
    tui()->send(tb);
}

void vvTUISplitter::setStyle(int t)
{
    TokenBuffer tb;
    style = t;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STYLE;
    tb << ID;
    tb << (style | shape);
    tui()->send(tb);
}

void vvTUISplitter::setOrientation(int orient)
{
    TokenBuffer tb;
    orientation = orient;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ORIENTATION;
    tb << ID;
    tb << orientation;
    tui()->send(tb);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIFrame::vvTUIFrame(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_FRAME)
{
    style = Sunken;
    shape = StyledPanel;
    setShape(shape);
    setStyle(style);
}

vvTUIFrame::vvTUIFrame(vvTabletUI *tui, const std::string &n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_FRAME)
{
    style = Sunken;
    shape = StyledPanel;
    setShape(shape);
    setStyle(style);
}

#ifdef USE_QT
vvTUIFrame::vvTUIFrame(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_FRAME)
{
    style = Sunken;
    shape = StyledPanel;
    setShape(shape);
    setStyle(style);
}
#endif

vvTUIFrame::~vvTUIFrame()
{
}

void vvTUIFrame::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIFrame::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIFrame::setShape(int s)
{
    TokenBuffer tb;
    shape = s;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SHAPE;
    tb << ID;
    tb << shape;
    tui()->send(tb);
}

void vvTUIFrame::setStyle(int t)
{
    TokenBuffer tb;
    style = t;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STYLE;
    tb << ID;
    tb << (style | shape);
    tui()->send(tb);
}

void vvTUIFrame::resend(bool create)
{
    vvTUIElement::resend(create);
    setShape(shape);
    setStyle(style);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIToggleButton::vvTUIToggleButton(const std::string &n, int pID, bool s)
    : vvTUIElement(n, pID, TABLET_TOGGLE_BUTTON)
{
    state = s;
    setVal(state);
}

vvTUIToggleButton::vvTUIToggleButton(vvTabletUI *tui, const std::string &n, int pID, bool s)
    : vvTUIElement(tui, n, pID, TABLET_TOGGLE_BUTTON)
{
    state = s;
    setVal(state);
}

#ifdef USE_QT
vvTUIToggleButton::vvTUIToggleButton(QObject *parent, const std::string &n, int pID, bool s)
    : vvTUIElement(parent, n, pID, TABLET_TOGGLE_BUTTON)
{
    state = s;
    setVal(state);
}
#endif

vvTUIToggleButton::~vvTUIToggleButton()
{
}

void vvTUIToggleButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        state = true;
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
        state = false;
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIToggleButton::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIToggleButton::setState(bool s)
{
    if (s != state) // don't send unnecessary state changes
    {
        state = s;
        setVal(state);
    }
}

bool vvTUIToggleButton::getState() const
{
    return state;
}

void vvTUIToggleButton::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(state);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIToggleBitmapButton::vvTUIToggleBitmapButton(const std::string &n, const std::string &down, int pID, bool state)
    : vvTUIElement(n, pID, TABLET_BITMAP_TOGGLE_BUTTON)
{
    bmpUp = n;
    bmpDown = down;

    setVal(bmpDown);
    setVal(state);
}

#ifdef USE_QT
vvTUIToggleBitmapButton::vvTUIToggleBitmapButton(QObject *parent, const std::string &n, const std::string &down, int pID, bool state)
    : vvTUIElement(parent, n, pID, TABLET_BITMAP_TOGGLE_BUTTON)
{
    bmpUp = n;
    bmpDown = down;

    setVal(bmpDown);
    setVal(state);
}
#endif

vvTUIToggleBitmapButton::~vvTUIToggleBitmapButton()
{
}

void vvTUIToggleBitmapButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        state = true;
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
        state = false;
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIToggleBitmapButton::parseMessage: unknown event " << i << endl;
    }
}

void vvTUIToggleBitmapButton::setState(bool s)
{
    if (s != state) // don't send unnecessary state changes
    {
        state = s;
        setVal(state);
    }
}

bool vvTUIToggleBitmapButton::getState() const
{
    return state;
}

void vvTUIToggleBitmapButton::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(bmpDown);
    setVal(state);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIMessageBox::vvTUIMessageBox(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_MESSAGE_BOX)
{
}

#ifdef USE_QT
vvTUIMessageBox::vvTUIMessageBox(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_MESSAGE_BOX)
{
}
#endif

vvTUIMessageBox::~vvTUIMessageBox()
{
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIEditField::vvTUIEditField(const std::string &n, int pID, const std::string &def)
    : vvTUIElement(n, pID, TABLET_EDIT_FIELD)
{
    this->text = name;
    immediate = false;
    setText(def);
}

vvTUIEditField::vvTUIEditField(vvTabletUI *tui, const std::string &n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_EDIT_FIELD)
{
    this->text = name;
    immediate = false;
}

#ifdef USE_QT
vvTUIEditField::vvTUIEditField(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_EDIT_FIELD)
{
    this->text = name;
    immediate = false;
}
#endif

vvTUIEditField::~vvTUIEditField()
{
}

void vvTUIEditField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void vvTUIEditField::parseMessage(TokenBuffer &tb)
{
    const char *m;
    tb >> m;
    text = m;
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUIEditField::setPasswordMode(bool b)
{
    setVal(TABLET_ECHOMODE, (int)b);
}

void vvTUIEditField::setIPAddressMode(bool b)
{
    setVal(TABLET_IPADDRESS, (int)b);
}

void vvTUIEditField::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

const std::string &vvTUIEditField::getText() const
{
    return text;
}

void vvTUIEditField::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(text);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------
//##########################################################

vvTUIEditTextField::vvTUIEditTextField(const std::string &n, int pID, const std::string &def)
    : vvTUIElement(n, pID, TABLET_TEXT_EDIT_FIELD)
{
    text = name;
    immediate = false;
    setText(def);
}

vvTUIEditTextField::vvTUIEditTextField(vvTabletUI *tui, const std::string &n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_TEXT_EDIT_FIELD)
{
    text = name;
    immediate = false;
}

#ifdef USE_QT
vvTUIEditTextField::vvTUIEditTextField(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_TEXT_EDIT_FIELD)
{
    text = name;
    immediate = false;
}
#endif

vvTUIEditTextField::~vvTUIEditTextField()
{
}

void vvTUIEditTextField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void vvTUIEditTextField::parseMessage(TokenBuffer &tb)
{
    const char *m;
    tb >> m;
    text = m;
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUIEditTextField::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

const std::string &vvTUIEditTextField::getText() const
{
    return text;
}

void vvTUIEditTextField::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(text);
    setVal(immediate);
}

//##########################################################
//----------------------------------------------------------
//----------------------------------------------------------

vvTUIEditIntField::vvTUIEditIntField(const std::string &n, int pID, int def)
    : vvTUIElement(n, pID, TABLET_INT_EDIT_FIELD)
{
    value = def;
    immediate = 0;
    setVal(value);
    setLabel("");
}

vvTUIEditIntField::vvTUIEditIntField(vvTabletUI *tui, const std::string &n, int pID, int def)
    : vvTUIElement(tui, n, pID, TABLET_INT_EDIT_FIELD)
{
    value = def;
    immediate = 0;
    setVal(value);
    setLabel("");
}

#ifdef USE_QT
vvTUIEditIntField::vvTUIEditIntField(QObject *parent, const std::string &n, int pID, int def)
    : vvTUIElement(parent, n, pID, TABLET_INT_EDIT_FIELD)
{
    value = def;
    immediate = 0;
    setVal(value);
    setLabel("");
}
#endif

vvTUIEditIntField::~vvTUIEditIntField()
{
}

void vvTUIEditIntField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void vvTUIEditIntField::parseMessage(TokenBuffer &tb)
{
    tb >> value;
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

std::string vvTUIEditIntField::getText() const
{
    return "";
}

void vvTUIEditIntField::setMin(int min)
{
    //cerr << "vvTUIEditIntField::setMin " << min << endl;
    this->min = min;
    setVal(TABLET_MIN, min);
}

void vvTUIEditIntField::setMax(int max)
{
    //cerr << "vvTUIEditIntField::setMax " << max << endl;
    this->max = max;
    setVal(TABLET_MAX, max);
}

void vvTUIEditIntField::setValue(int val)
{
    if (value != val)
    {
        value = val;
        setVal(value);
    }
}

void vvTUIEditIntField::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(TABLET_MIN, min);
    setVal(TABLET_MAX, max);
    setVal(value);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIEditFloatField::vvTUIEditFloatField(const std::string &n, int pID, float def)
    : vvTUIElement(n, pID, TABLET_FLOAT_EDIT_FIELD)
{
    value = def;
    setVal(value);
    immediate = 0;
    setLabel("");
}

vvTUIEditFloatField::vvTUIEditFloatField(vvTabletUI *tui, const std::string &n, int pID, float def)
    : vvTUIElement(tui, n, pID, TABLET_FLOAT_EDIT_FIELD)
{
    value = def;
    setVal(value);
    immediate = 0;
    setLabel("");
}

#ifdef USE_QT
vvTUIEditFloatField::vvTUIEditFloatField(QObject *parent, const std::string &n, int pID, float def)
    : vvTUIElement(parent, n, pID, TABLET_FLOAT_EDIT_FIELD)
{
    value = def;
    setVal(value);
    immediate = 0;
    setLabel("");
}
#endif

vvTUIEditFloatField::~vvTUIEditFloatField()
{
}

void vvTUIEditFloatField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void vvTUIEditFloatField::parseMessage(TokenBuffer &tb)
{
    tb >> value;
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUIEditFloatField::setValue(float val)
{
    if (value != val)
    {
        value = val;
        setVal(value);
    }
}

void vvTUIEditFloatField::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(value);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUISpinEditfield::vvTUISpinEditfield(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_SPIN_EDIT_FIELD)
{
    actValue = 0;
    minValue = 0;
    maxValue = 100;
    step = 1;
    setVal(actValue);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
}

#ifdef USE_QT
vvTUISpinEditfield::vvTUISpinEditfield(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_SPIN_EDIT_FIELD)
{
    actValue = 0;
    minValue = 0;
    maxValue = 100;
    step = 1;
    setVal(actValue);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
}
#endif

vvTUISpinEditfield::~vvTUISpinEditfield()
{
}

void vvTUISpinEditfield::parseMessage(TokenBuffer &tb)
{
    tb >> actValue;
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUISpinEditfield::setPosition(int newV)
{

    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void vvTUISpinEditfield::setStep(int newV)
{
    step = newV;
    setVal(TABLET_STEP, step);
}

void vvTUISpinEditfield::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void vvTUISpinEditfield::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void vvTUISpinEditfield::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
    setVal(actValue);
}

//----------------------------------------------------------
//----------------------------------------------------------
vvTUITextSpinEditField::vvTUITextSpinEditField(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_TEXT_SPIN_EDIT_FIELD)
{
    text = "";
    minValue = 0;
    maxValue = 100;
    step = 1;
    setVal(text);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
}

#ifdef USE_QT
vvTUITextSpinEditField::vvTUITextSpinEditField(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_TEXT_SPIN_EDIT_FIELD)
{
    text = "";
    minValue = 0;
    maxValue = 100;
    step = 1;
    setVal(text);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
}
#endif

vvTUITextSpinEditField::~vvTUITextSpinEditField()
{
}

void vvTUITextSpinEditField::parseMessage(TokenBuffer &tb)
{
    const char *m;
    tb >> m;
    text = m;
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUITextSpinEditField::setStep(int newV)
{
    step = newV;
    setVal(TABLET_STEP, step);
}

void vvTUITextSpinEditField::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void vvTUITextSpinEditField::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void vvTUITextSpinEditField::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
    setVal(text);
}

void vvTUITextSpinEditField::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIProgressBar::vvTUIProgressBar(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_PROGRESS_BAR)
{
    actValue = 0;
    maxValue = 100;
}

#ifdef USE_QT
vvTUIProgressBar::vvTUIProgressBar(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_PROGRESS_BAR)
{
    actValue = 0;
    maxValue = 100;
}
#endif

vvTUIProgressBar::~vvTUIProgressBar()
{
}

void vvTUIProgressBar::setValue(int newV)
{
    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void vvTUIProgressBar::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void vvTUIProgressBar::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(TABLET_MAX, maxValue);
    setVal(actValue);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIFloatSlider::vvTUIFloatSlider(const std::string &n, int pID, bool s)
    : vvTUIElement(n, pID, TABLET_FLOAT_SLIDER)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    ticks = 10;

    orientation = s;
    setVal(orientation);
}

vvTUIFloatSlider::vvTUIFloatSlider(vvTabletUI *tui, const std::string &n, int pID, bool s)
: vvTUIElement(tui, n, pID, TABLET_FLOAT_SLIDER)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    ticks = 10;

    orientation = s;
    setVal(orientation);
}

#ifdef USE_QT
vvTUIFloatSlider::vvTUIFloatSlider(QObject *parent, const std::string &n, int pID, bool s)
    : vvTUIElement(parent, n, pID, TABLET_FLOAT_SLIDER)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    ticks = 10;

    orientation = s;
    setVal(orientation);
}
#endif

vvTUIFloatSlider::~vvTUIFloatSlider()
{
}

void vvTUIFloatSlider::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    tb >> actValue;
    if (i == TABLET_PRESSED)
    {
#ifdef USE_QT
        emit tabletPressEvent();
        emit tabletEvent();
#endif
        if (listener)
        {
            listener->tabletPressEvent(this);
            listener->tabletEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
#ifdef USE_QT
        emit tabletReleaseEvent();
        emit tabletEvent();
#endif
        if (listener)
        {
            listener->tabletReleaseEvent(this);
            listener->tabletEvent(this);
        }
    }
    else
    {
#ifdef USE_QT
        emit tabletEvent();
#endif
        if (listener)
            listener->tabletEvent(this);
    }
}

void vvTUIFloatSlider::setValue(float newV)
{
    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void vvTUIFloatSlider::setTicks(int newV)
{
    if (ticks != newV)
    {
        ticks = newV;
        setVal(TABLET_NUM_TICKS, ticks);
    }
}

void vvTUIFloatSlider::setMin(float minV)
{
    if (minValue != minV)
    {
        minValue = minV;
        setVal(TABLET_MIN, minValue);
    }
}

void vvTUIFloatSlider::setMax(float maxV)
{
    if (maxValue != maxV)
    {
        maxValue = maxV;
        setVal(TABLET_MAX, maxValue);
    }
}

void vvTUIFloatSlider::setRange(float minV, float maxV)
{
    minValue = minV;
    maxValue = maxV;
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
}

void vvTUIFloatSlider::setOrientation(bool o)
{
    orientation = o;
    setVal(orientation);
}

void vvTUIFloatSlider::setLogarithmic(bool val)
{
    logarithmic = val;
    setVal(TABLET_SLIDER_SCALE, logarithmic ? TABLET_SLIDER_LOGARITHMIC : TABLET_SLIDER_LINEAR);
}

void vvTUIFloatSlider::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_NUM_TICKS, ticks);
    setVal(TABLET_SLIDER_SCALE, logarithmic ? TABLET_SLIDER_LOGARITHMIC : TABLET_SLIDER_LINEAR);
    setVal(actValue);
    setVal(orientation);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUISlider::vvTUISlider(const std::string &n, int pID, bool s)
    : vvTUIElement(n, pID, TABLET_SLIDER)
    , actValue(0)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    orientation = s;
    setVal(orientation);
}

vvTUISlider::vvTUISlider(vvTabletUI *tui, const std::string &n, int pID, bool s)
    : vvTUIElement(tui, n, pID, TABLET_SLIDER)
    , actValue(0)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    orientation = s;
    setVal(orientation);
}

#ifdef USE_QT
vvTUISlider::vvTUISlider(QObject *parent, const std::string &n, int pID, bool s)
    : vvTUIElement(parent, n, pID, TABLET_SLIDER)
    , actValue(0)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    orientation = s;
    setVal(orientation);
}
#endif

vvTUISlider::~vvTUISlider()
{
}

void vvTUISlider::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    tb >> actValue;
    if (i == TABLET_PRESSED)
    {
#ifdef USE_QT
        emit tabletPressEvent();
        emit tabletEvent();
#endif
        if (listener)
        {
            listener->tabletPressEvent(this);
            listener->tabletEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
#ifdef USE_QT
        emit tabletReleaseEvent();
        emit tabletEvent();
#endif
        if (listener)
        {
            listener->tabletReleaseEvent(this);
            listener->tabletEvent(this);
        }
    }
    else
    {
#ifdef USE_QT
        emit tabletEvent();
#endif
        if (listener)
            listener->tabletEvent(this);
    }
}

void vvTUISlider::setValue(int newV)
{

    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void vvTUISlider::setTicks(int newV)
{
    if (ticks != newV)
    {
        ticks = newV;
        setVal(TABLET_NUM_TICKS, ticks);
    }
}

void vvTUISlider::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void vvTUISlider::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void vvTUISlider::setRange(int minV, int maxV)
{
    minValue = minV;
    maxValue = maxV;
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
}

void vvTUISlider::setOrientation(bool o)
{
    orientation = o;
    setVal(orientation);
}

void vvTUISlider::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_NUM_TICKS, ticks);
    setVal(actValue);
    setVal(orientation);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIComboBox::vvTUIComboBox(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_COMBOBOX)
{
    label = "";
    text = "";
    selection = -1;
}

vvTUIComboBox::vvTUIComboBox(vvTabletUI *tui, const std::string &n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_COMBOBOX)
{
    label = "";
    text = "";
    selection = -1;
}

#ifdef USE_QT
vvTUIComboBox::vvTUIComboBox(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_COMBOBOX)
{
    label = "";
    text = "";
    selection = -1;
}
#endif

vvTUIComboBox::~vvTUIComboBox()
{
}

void vvTUIComboBox::parseMessage(TokenBuffer &tb)
{
    tb >> text;
    int i = 0;
    selection = -1;
	for (const auto& it : elements)
	{
		if (it == text)
		{
			selection = i;
			break;
		}
		i++;
	}
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUIComboBox::addEntry(const std::string &t)
{
    if (selection == -1  && elements.size() == 0)
        selection = 0;
	elements.push_back(t);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
}

void vvTUIComboBox::delEntry(const std::string &t)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
	for (const auto& it : elements)
	{
		if (it == t)
		{
			elements.remove(it);
			break;
		}
	}
}

int vvTUIComboBox::getNumEntries()
{
    return (int)elements.size();
}

void vvTUIComboBox::clear()
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ALL;
    tb << ID;
    tui()->send(tb);
    elements.clear();
}

void vvTUIComboBox::setSelectedText(const std::string &t)
{
    text = t;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
	int i = 0;
	selection = -1;
	for (const auto& it : elements)
	{
		if (it == text)
		{
			selection = i;
			break;
		}
		i++;
	}
}

const std::string &vvTUIComboBox::getSelectedText() const
{
    return text;
}

int vvTUIComboBox::getSelectedEntry() const
{
    return selection;
}

void vvTUIComboBox::setSelectedEntry(int e)
{
    selection = e;
    if (e >= elements.size())
        selection = (int)elements.size() - 1;
    if (selection < 0)
        return;
	std::string selectedEntry;
	int i = 0;
	for (const auto& it : elements)
	{
		if (i == selection)
		{
			selectedEntry = it;
			break;
		}
		i++;
	}
    text = selectedEntry;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
}

void vvTUIComboBox::resend(bool create)
{
    vvTUIElement::resend(create);
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_REMOVE_ALL;
        tb << ID;
        tui()->send(tb);
    }
	for (const auto& it : elements)
	{
		TokenBuffer tb;
		tb << TABLET_SET_VALUE;
		tb << TABLET_ADD_ENTRY;
		tb << ID;
		tb << it;
		tui()->send(tb);
	}
    if (text != "")
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_SELECT_ENTRY;
        tb << ID;
        tb << text.c_str();
        tui()->send(tb);
    }
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTUIListBox::vvTUIListBox(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_LISTBOX)
{
    text = "";
    selection = -1;
}

#ifdef USE_QT
vvTUIListBox::vvTUIListBox(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_LISTBOX)
{
    text = "";
    selection = -1;
}
#endif

vvTUIListBox::~vvTUIListBox()
{
}

void vvTUIListBox::parseMessage(TokenBuffer &tb)
{
	tb >> text;
	int i = 0;
	for (const auto& it : elements)
	{
		if (it == text)
		{
			selection = i;
			break;
		}
		i++;
	}
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUIListBox::addEntry(const std::string &t)
{
    elements.push_back(t);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
}

void vvTUIListBox::delEntry(const std::string &t)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
	for (const auto& it : elements)
	{
		if (it == t)
		{
			elements.remove(it);
			break;
		}
	}
}

void vvTUIListBox::setSelectedText(const std::string &t)
{
    text = t;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
	int i = 0;
	for (const auto& it : elements)
	{
		if (it == text)
		{
			selection = i;
			break;
		}
		i++;
	}
}

const std::string &vvTUIListBox::getSelectedText() const
{
    return text;
}

int vvTUIListBox::getSelectedEntry() const
{
    return selection;
}

void vvTUIListBox::setSelectedEntry(int e)
{
    selection = e;
    if (e >= elements.size())
        selection = (int)elements.size() - 1;
    if (selection < 0)
        return;
	std::string selectedEntry;
	int i = 0;
	for (const auto& it : elements)
	{
		if (i == selection)
		{
			selectedEntry = it;
			break;
		}
		i++;
	}
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
}

void vvTUIListBox::resend(bool create)
{
    vvTUIElement::resend(create);

	for (const auto& it : elements)
	{
		TokenBuffer tb;
		tb << TABLET_SET_VALUE;
		tb << TABLET_ADD_ENTRY;
		tb << ID;
		tb << it;
		tui()->send(tb);
	}
    if (text != "")
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_SELECT_ENTRY;
        tb << ID;
        tb << text;
        tui()->send(tb);
    }
}

//----------------------------------------------------------
//----------------------------------------------------------

MapData::MapData(const char *pname, float pox, float poy, float pxSize, float pySize, float pheight)
{
    name = new char[strlen(pname) + 1];
    strcpy(name, pname);
    ox = pox;
    oy = poy;
    xSize = pxSize;
    ySize = pySize;
    height = pheight;
}

MapData::~MapData()
{
    delete[] name;
}

vvTUIMap::vvTUIMap(const char *n, int pID)
    : vvTUIElement(n, pID, TABLET_MAP)
{
}

vvTUIMap::~vvTUIMap()
{
	for (const auto& it : maps)
	{
		delete[] it;
	}
	maps.clear();
}

void vvTUIMap::parseMessage(TokenBuffer &tb)
{
    tb >> mapNum;
    tb >> xPos;
    tb >> yPos;
    tb >> height;

    if (listener)
        listener->tabletEvent(this);
}

void vvTUIMap::addMap(const char *name, float ox, float oy, float xSize, float ySize, float height)
{
    MapData *md = new MapData(name, ox, oy, xSize, ySize, height);
	maps.push_back(md);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_MAP;
    tb << ID;
    tb << md->name;
    tb << md->ox;
    tb << md->oy;
    tb << md->xSize;
    tb << md->ySize;
    tb << md->height;
    tui()->send(tb);
}

void vvTUIMap::resend(bool create)
{
    vvTUIElement::resend(create);


	for (const auto& it : maps)
	{
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_ADD_MAP;
        tb << ID;
        tb << it->name;
        tb << it->ox;
        tb << it->oy;
        tb << it->xSize;
        tb << it->ySize;
        tb << it->height;
        tui()->send(tb);
    }
}



vvTUIEarthMap::vvTUIEarthMap(const char *n, int pID)
    : vvTUIElement(n, pID, TABLET_EARTHMAP)
{
}

vvTUIEarthMap::~vvTUIEarthMap()
{
    
}

void vvTUIEarthMap::parseMessage(TokenBuffer &tb)
{
    tb >> latitude;
    tb >> longitude;
    tb >> altitude;

    if (listener)
        listener->tabletEvent(this);
}

void vvTUIEarthMap::setPosition(float lat, float longi, float alt)
{
    latitude = lat;
    longitude = longi;
    altitude = alt;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_FLOAT;
    tb << ID;
    tb << latitude;
    tb << longitude;
    tb << altitude;
    tui()->send(tb);
}

void vvTUIEarthMap::addPathNode(float latitude, float longitude)
{
    path.push_back(pair<float, float>(latitude, longitude));
}

void vvTUIEarthMap::updatePath()
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_GEO_PATH;
    tb << ID;
    tb << (unsigned)path.size();
    for (auto p = path.begin(); p != path.end(); p++)
    {
        tb << p->first;
        tb << p->second;
    }
    tui()->send(tb);
}
void vvTUIEarthMap::setMinMax(float minH, float maxH)
{
    minHeight = minH;
    maxHeight = maxH;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_MIN_MAX;
    tb << ID;
    tb << minHeight;
    tb << maxHeight;
    tui()->send(tb);
}

void vvTUIEarthMap::resend(bool create)
{
    vvTUIElement::resend(create);

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_FLOAT;
    tb << ID;
    tb << latitude;
    tb << longitude;
    tb << altitude;
    tui()->send(tb);
    setMinMax(minHeight,maxHeight);
    updatePath();
}

//----------------------------------------------------------
//----------------------------------------------------------
//##########################################################

vvTUIPopUp::vvTUIPopUp(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_POPUP)
{
    text = "";
    immediate = false;
}

#ifdef USE_QT
vvTUIPopUp::vvTUIPopUp(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_POPUP)
{
    text = "";
    immediate = false;
}
#endif

vvTUIPopUp::~vvTUIPopUp()
{
}

void vvTUIPopUp::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void vvTUIPopUp::parseMessage(TokenBuffer &tb)
{
    const char *m;
    tb >> m;
    text = m;
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
        listener->tabletEvent(this);
}

void vvTUIPopUp::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

void vvTUIPopUp::resend(bool create)
{
    vvTUIElement::resend(create);
    setVal(text);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------

vvTabletUI *vvTabletUI::tUI = NULL;

vvTabletUI *vvTabletUI::instance()
{
    if (tUI == NULL)
        tUI = new vvTabletUI();
    return tUI;
}
//----------------------------------------------------------
//----------------------------------------------------------

vvTUIElement::vvTUIElement(const std::string &n, int pID, int type)
: QT(QObject(0) D_COMMA) type(type), m_tui(vvTabletUI::instance())
{
    xs = -1;
    ys = -1;
    xp = 0;
    yp = 0;
    parentID = pID;
    name = n;
    label = n;
    ID = tui()->getID();
    listener = NULL;
    tui()->addElement(this);
    createSimple(type);
    if(tui()->debugTUI())
    {
        vvMSController::instance()->agreeString(name);
        vvMSController::instance()->agreeInt(ID);
    }
}

vvTUIElement::vvTUIElement(vvTabletUI *tabletUI, const std::string &n, int pID, int type)
: QT(QObject(0) D_COMMA) type(type), m_tui(tabletUI)
{
    xs = -1;
    ys = -1;
    xp = 0;
    yp = 0;
    parentID = pID;
    name = n;
    label = n;
    ID = tui()->getID();
    listener = NULL;
    tui()->addElement(this);
    createSimple(type);
    if(tui()->debugTUI())
    {
        vvMSController::instance()->agreeString(name);
        vvMSController::instance()->agreeInt(ID);
    }
}

#if 0
vvTUIElement::vvTUIElement(QObject *parent, const std::string &n, int pID)
: QObject(parent)
{
    xs = -1;
    ys = -1;
    xp = 0;
    yp = 0;
    parentID = pID;
    name = n;
    label = n;
    ID = tui()->getID();
    tui()->addElement(this);
    listener = NULL;
    if(tui()->debugTUI())
    {
        vvMSController::instance()->agreeString(name);
        vvMSController::instance()->agreeInt(ID);
    }
}
#endif

#ifdef USE_QT
vvTUIElement::vvTUIElement(QObject *parent, const std::string &n, int pID, int type)
: QObject(parent)
, type(type)
, m_tui(vvTabletUI::instance())
{
    xs = -1;
    ys = -1;
    xp = 0;
    yp = 0;
    parentID = pID;
    name = n;
    label = n;
    ID = tui()->getID();
    listener = NULL;
    tui()->addElement(this);
    if(tui()->debugTUI())
    {
        vvMSController::instance()->agreeString(name);
        vvMSController::instance()->agreeInt(ID);
    }
}
#endif

vvTUIElement::~vvTUIElement()
{
    TokenBuffer tb;
    tb << TABLET_REMOVE;
    tb << ID;
    tui()->send(tb);
    tui()->removeElement(this);
}

bool vvTUIElement::createSimple(int type)
{
    TokenBuffer tb;
    tb << TABLET_CREATE;
    tb << ID;
    tb << type;
    tb << parentID;
    tb << name.c_str();
    return tui()->send(tb);
}

vvTabletUI *vvTUIElement::tui() const
{
    if (m_tui)
        return m_tui;
    else
        return vvTabletUI::instance();
}

void vvTUIElement::setLabel(const char *l)
{
    if (l)
        setLabel(std::string(l));
    else
        setLabel(std::string(""));
}

void vvTUIElement::setLabel(const std::string &l)
{
    label = l;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_LABEL;
    tb << ID;
    tb << label.c_str();
    tui()->send(tb);
}

int vvTUIElement::getID() const
{
    return ID;
}

void vvTUIElement::setEventListener(vvTUIListener *l)
{
    listener = l;
}

void vvTUIElement::parseMessage(TokenBuffer &)
{
}

vvTUIListener *vvTUIElement::getMenuListener()
{
    return listener;
}

void vvTUIElement::setVal(float value)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_FLOAT;
    tb << ID;
    tb << value;
    tui()->send(tb);
}

void vvTUIElement::setVal(bool value)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BOOL;
    tb << ID;
    tb << (char)value;
    tui()->send(tb);
}

void vvTUIElement::setVal(int value)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_INT;
    tb << ID;
    tb << value;
    tui()->send(tb);
}

void vvTUIElement::setVal(const std::string &value)
{
    if (!tui()->isConnected())
        return;

    //cerr << "vvTUIElement::setVal info: " << (value ? value : "*NULL*") << endl;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STRING;
    tb << ID;
    tb << value.c_str();
    tui()->send(tb);
}

void vvTUIElement::setVal(int type, int value)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tui()->send(tb);
}

void vvTUIElement::setVal(int type, float value)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tui()->send(tb);
}
void vvTUIElement::setVal(int type, int value, const std::string &nodePath)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tb << nodePath.c_str();
    tui()->send(tb);
}
void vvTUIElement::setVal(int type, const std::string &nodePath, const std::string &simPath, const std::string &simName)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << nodePath.c_str();
    tb << simPath.c_str();
    tb << simName.c_str();
    tui()->send(tb);
}

void vvTUIElement::setVal(int type, int value, const std::string &nodePath, const std::string &parentPath)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tb << nodePath.c_str();
    tb << parentPath.c_str();
    tui()->send(tb);
}
void vvTUIElement::resend(bool create)
{
    if (create)
    {
        if (!createSimple(type))
        {
            std::cerr << "vvTUIElement::resend(create=true): createSimple failed for ID=" << ID
                      << ", parent=" << parentID << std::endl;
        }
    }

    TokenBuffer tb;

    if (xs > 0)
    {
        tb.reset();
        tb << TABLET_SET_VALUE;
        tb << TABLET_SIZE;
        tb << ID;
        tb << xs;
        tb << ys;
        tui()->send(tb);
    }

    tb.reset();
    tb << TABLET_SET_VALUE;
    tb << TABLET_POS;
    tb << ID;
    tb << xp;
    tb << yp;
    tui()->send(tb);

    tb.reset();
    tb << TABLET_SET_VALUE;
    tb << TABLET_LABEL;
    tb << ID;
    tb << label.c_str();
    tui()->send(tb);

    if (!enabled)
    {
        tb.reset();
        tb << TABLET_SET_VALUE;
        tb << TABLET_SET_ENABLED;
        tb << ID;
        tb << enabled;
        tui()->send(tb);
    }

    if (hidden)
    {
        tb.reset();
        tb << TABLET_SET_VALUE;
        tb << TABLET_SET_HIDDEN;
        tb << ID;
        tb << hidden;
        tui()->send(tb);
    }
}

void vvTUIElement::setPos(int x, int y)
{
    xp = x;
    yp = y;
    if ((x > 10000 || x < -10000) || (y > 10000 || y < -10000))
    {
        fprintf(stderr, "coordinates out of range!, x=%d, y=%d\n", x, y);
#ifdef _WIN32
        DebugBreak();
#else
        abort();
#endif
    }
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_POS;
    tb << ID;
    tb << xp;
    tb << yp;
    tui()->send(tb);
}

void vvTUIElement::setHidden(bool newState)
{
    //std::cerr << "vvTUIElement::setHidden(hide=" << hidden << " -> " << newState << "): ID=" << ID << std::endl;
    if (hidden == newState)
        return;

    hidden = newState;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SET_HIDDEN;
    tb << ID;
    tb << hidden;
    tui()->send(tb);
}

void vvTUIElement::setEnabled(bool newState)
{
    if (enabled == newState)
        return;

    enabled = newState;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SET_ENABLED;
    tb << ID;
    tb << enabled;
    tui()->send(tb);
}

void vvTUIElement::setColor(Qt::GlobalColor color)
{

    this->color = color;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_COLOR;
    tb << ID;
    tb << this->color;
    tui()->send(tb);
}

void vvTUIElement::setSize(int x, int y)
{
    xs = x;
    ys = y;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SIZE;
    tb << ID;
    tb << xs;
    tb << ys;
    tui()->send(tb);
}

vvTabletUI::vvTabletUI()
{
    assert(!tUI);
    tUI = this;

    config();
    init();
}

vvTabletUI::vvTabletUI(const std::string &host, int port): connectionConfig(host, port)
{
    config();
    init();
}

vvTabletUI::vvTabletUI(int fd, int fdSg): connectionConfig(fd, fdSg)
{
    config();
    init();
}

void vvTabletUI::config()
{
    debugTUIState = coCoviseConfig::isOn("VIVE.DebugTUI", debugTUIState);
    timeout = coCoviseConfig::getFloat("VIVE.TabletPC.Timeout", timeout);
}

void vvTabletUI::init()
{
    reinit(connectionConfig);
}

void vvTabletUI::reinit(const ConfigData &cd)
{
    assert(cd.mode != None);

    lock();

    if (mode != None)
    {
        close();
    }

    switch (cd.mode)
    {
    case None:
        // just to get warnings about missing other cases
        break;
    case Config:
    {
        port = 31802;
        std::string line;
        if (getenv("COVER_TABLETUI"))
        {
            std::string env(getenv("COVER_TABLETUI"));
            std::string::size_type p = env.find(':');
            if (p != std::string::npos)
            {
                port = atoi(env.substr(p + 1).c_str());
                env = env.substr(0, p);
            }
            line = env;
            std::cerr << "getting TabletPC configuration from $COVER_TABLETUI: " << line << ":" << port << std::endl;
            serverMode = false;
        }
        else
        {
            port = coCoviseConfig::getInt("port", "VIVE.TabletUI", port);
            line = coCoviseConfig::getEntry("host", "VIVE.TabletUI");
            serverMode = coCoviseConfig::isOn("VIVE.TabletUI.ServerMode", false);
        }

        if (!line.empty())
        {
            if (strcasecmp(line.c_str(), "NONE") != 0)
            {
                serverHost = new Host(line.c_str());
                localHost = new Host("localhost");
            }
        }
        else
        {
            localHost = new Host("localhost");
        }
        break;
    }
    case Client:
    {
        serverMode = false;
        serverHost = new Host(cd.host.c_str());
        port = cd.port;

        break;
    }
    case ConnectedSocket:
    {
        serverMode = false;

        tryConnect();

        localHost = new Host("localhost");
        connectedHost = localHost;
        port = 0;

        conn = std::make_unique<Connection>(cd.fd);
        conn->set_sendertype(0);
        conn->getSocket()->setNonBlocking(false);
        sgConn = std::make_unique<Connection>(cd.fdSg);
        sgConn->set_sendertype(0);
        sgConn->getSocket()->setNonBlocking(false);

        sendThread = std::thread(
            [this]()
            {
                setThreadName("coTabletIU:send");
                //std::cerr << "vvTabletUI: sendThread: entering loop" << std::endl;
                for (;;)
                {
                    std::unique_lock<std::mutex> lock(sendMutex);
                    if (sendQueue.empty())
                    {
                        sendCond.wait(lock);
                    }
                    if (sendQueue.empty())
                    {
                        continue;
                    }
                    DataHandle data = std::move(sendQueue.front());
                    sendQueue.pop_front();
                    lock.unlock();
                    if (data.length() == 0)
                    {
                        std::cerr << "vvTabletUI: sendThread: zero-length message, terminating" << std::endl;
                        break;
                    }
                    Message msg(COVISE_MESSAGE_TABLET_UI, data);
                    if (!conn->sendMessage(&msg))
                    {
                        std::cerr << "vvTabletUI: sendThread: connection failed" << std::endl;
                        break;
                    }
                }
            });

        resendAll();
        break;
    }
    }

    mode = cd.mode;

    unlock();
}

// resend all ui Elements to the TabletPC
void vvTabletUI::resendAll()
{
    for (auto el: elements)
    {
        el->resend(true);
    }
    newElements.clear();
}

bool vvTabletUI::isConnected() const
{
    return connectedHost != nullptr;
}

void vvTabletUI::close()
{
    if (connThread.joinable())
    {
        lock();
        if (conn)
            conn->cancel();
        unlock();
        connThread.join();
    }

    {
        std::unique_lock<std::mutex> guard(sendMutex);
        sendQueue.clear();
        if (sendThread.joinable())
        {
            // send zero-length message to terminate sendThread
            sendQueue.emplace_back();
            sendCond.notify_one();
            guard.unlock();
            sendThread.join();
        }
    }

    connectedHost = NULL;
    conn.reset();
    sgConn.reset();

    delete serverConn;
    serverConn = NULL;

    delete serverHost;
    serverHost = NULL;

    delete localHost;
    localHost = NULL;
}

void vvTabletUI::tryConnect()
{
    lock();
    if (connThread.joinable())
    {
        if (conn)
            conn->cancel();
        connThread.join();
    }

    {
        std::unique_lock<std::mutex> guard(sendMutex);
        sendQueue.clear();
        if (sendThread.joinable())
        {
            // send zero-length message to terminate sendThread
            sendQueue.emplace_back();
            sendCond.notify_one();
            guard.unlock();
            sendThread.join();
        }
    }

    connectedHost = NULL;

    delete serverConn;
    serverConn = NULL;
    unlock();
}

bool vvTabletUI::debugTUI()
{
    return debugTUIState;
}

vvTabletUI::~vvTabletUI()
{
    close();

    if (tUI == this)
        tUI = nullptr;

    if (connThread.joinable())
    {
        lock();
        if (conn)
            conn->cancel();
        connThread.join();
        unlock();
    }

    connectedHost = nullptr;
    conn.reset();

    delete serverHost;
    delete localHost;
}

int vvTabletUI::getID()
{
    return ID++;
}

bool vvTabletUI::send(TokenBuffer &tb)
{
    if (!connectedHost)
    {
        return false;
    }
    assert(conn);
    if (sendThread.joinable())
    {
        std::unique_lock<std::mutex> lock(sendMutex);
        sendQueue.emplace_back(tb.getData());
        sendCond.notify_one();
        return true;
    }

    Message m(tb);
    m.type = COVISE_MESSAGE_TABLET_UI;
    if (!conn->sendMessage(&m))
    {
        std::cerr << "vvTabletUI::send: connection failed" << std::endl;
        return false;
    }
    return true;
}

bool vvTabletUI::update()
{
    if (vvMSController::instance() == NULL)
        return false;

    if (connThread.joinable())
    {
        lock();
        if (!connecting)
        {
            connThread.join();
            if (connectingHost)
            {
                assert(conn);
                assert(sgConn);
            }
            else
            {
                conn.reset();
                sgConn.reset();
            }
        }
        unlock();
    }

    bool hasConnected = false;
    lock();
    if (connectingHost)
    {
        assert(conn);
        assert(sgConn);
        assert(!connectedHost);
        std::swap(connectedHost, connectingHost);
        assert(!connectingHost);
        hasConnected = true;
    }
    unlock();

    if (connectedHost)
    {
    }
    else if (vvMSController::instance()->isMaster() && serverMode)
    {
        if (serverConn == NULL)
        {
            serverConn = new ServerConnection(port, 0, (sender_type)0);
            serverConn->listen();
        }
    }
    else if ((vvMSController::instance()->isMaster()) && (serverHost != NULL || localHost != NULL))
    {
        if (abs(vv->frameRealTime() - oldTime) > 2.)
        {
            oldTime = vv->frameRealTime();
            {
                Message msg(Message::UI, "WANT_TABLETUI");
                vvPluginList::instance()->sendVisMessage(&msg);
            }

            lock();
            if (!connThread.joinable())
            {
                connectingHost = nullptr;
                connecting = true;
                connThread = std::thread(
                    [this]()
                    {
                        setThreadName("vvTabletUI:conn");

                        ClientConnection *nconn = nullptr;
                        Host *host = nullptr;
                        for (auto h: {serverHost, localHost})
                        {
                            if (!h)
                                continue;
                            if ((firstConnection && vv->debugLevel(1)) || vv->debugLevel(3))
                                std::cerr << "Trying tablet UI connection to " << h->getPrintable() << ":" << port
                                          << "... " << std::flush;
                            nconn = new ClientConnection(h, port, 0, (sender_type)0, 0, timeout);
                            if ((firstConnection && vv->debugLevel(1)) || vv->debugLevel(3))
                                std::cerr << (nconn->is_connected() ? "success" : "failed") << "." << std::endl;
                            firstConnection = false;
                            if (nconn && nconn->is_connected())
                            {
                                lock();
                                conn.reset(nconn);
                                unlock();
                                host = h;
                                break;
                            }
                            else if (nconn) // could not open server port
                            {
                                delete nconn;
                                nconn = NULL;
                            }
                        }

                        if (!conn || !host)
                        {
                            connecting = false;
                            return;
                        }

                        // create Texture and SGBrowser Connections
                        Message msg;
                        conn->recv_msg(&msg);
                        if (msg.type == COVISE_MESSAGE_SOCKET_CLOSED)
                        {
                            lock();
                            conn.reset();
                            unlock();
                            sgConn.reset();
                            connecting = false;
                            return;
                        }
                        if (msg.type == covise::COVISE_MESSAGE_TABLET_UI)
                        {
                            TokenBuffer stb(&msg);
                            int sgPort = 0;
                            stb >> sgPort;

                            ClientConnection *cconn = new ClientConnection(host, sgPort, 0, (sender_type)0, 2, 1);
                            if (!cconn->is_connected()) // could not open server port
                            {
#ifndef _WIN32
                                if (errno != ECONNREFUSED)
                                {
                                    fprintf(stderr, "Could not connect to TabletPC SGBrowser %s; port %d: %s\n",
                                            host->getPrintable(), sgPort, strerror(errno));
                                }
#else
                                fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", host->getPrintable(),
                                        sgPort);
#endif
                                lock();
                                conn.reset();
                                unlock();
                                delete cconn;
                                cconn = NULL;
                                connecting = false;
                                return;
                            }
                            sgConn.reset(cconn);
                        }
                        else
                        {
                            lock();
                            conn.reset();
                            unlock();
                            sgConn.reset();
                            connecting = false;
                            return;
                        }

                        lock();
                        connectingHost = host;
                        connecting = false;
                        unlock();
                        return;
                    });
            }
            unlock();
        }
    }

    if (connectedHost && conn && serverMode)
    {
        if (conn->is_connected() && !serverHost)
        {
            Message m;
            conn->recv_msg(&m);
            TokenBuffer tb(&m);
            const char *hostName;
            tb >> hostName;
            serverHost = new Host(hostName);

            hasConnected = true;
        }
    }
    else if (serverConn && serverConn->check_for_input())
    {
        if (conn)
        {
            connectedHost = nullptr;
            sgConn.reset();
        }
        conn = serverConn->spawn_connection();
        if (conn && conn->is_connected())
        {
            Message m;
            conn->recv_msg(&m);
            TokenBuffer tb(&m);
            const char *hostName;
            tb >> hostName;
            serverHost = new Host(hostName);

            hasConnected = true;
        }
    }

    if (hasConnected)
    {
        std::cerr << "vvTabletUI: new connection - sending all elements" << std::endl;
        resendAll();
    }

    for (auto el: newElements)
    {
        el->resend(false);
    }
    newElements.clear();

    bool changed = false;
    bool gotMessage = false;
    do
    {
        gotMessage = false;
        Message m;
        if (vvMSController::instance()->isMaster())
        {
            if (conn && connectedHost)
            {
                if (conn->check_for_input())
                {
                    conn->recv_msg(&m);
                    gotMessage = true;
                }
            }
        }
        gotMessage = vvMSController::instance()->syncBool(gotMessage);
        if (gotMessage)
        {
            if (vvMSController::instance()->isMaster())
            {
                vvMSController::instance()->sendSlaves(&m);
            }
            else
            {
                if (vvMSController::instance()->readMaster(&m) < 0)
                {
                    cerr << "vvTabletUI::update: could not read message from Master" << endl;
                    //cerr << "sync_exit13 " << myID << endl;
                    exit(0);
                }
            }

            changed = true;

            TokenBuffer tb(&m);
            switch (m.type)
            {
            case COVISE_MESSAGE_SOCKET_CLOSED:
            case COVISE_MESSAGE_CLOSE_SOCKET:
            {
                connectedHost = nullptr;
                conn.reset();
                sgConn.reset();
            }
            break;
            case COVISE_MESSAGE_TABLET_UI:
            {

                int ID;
                tb >> ID;
                if (ID >= 0)
                {
                    for(size_t i=0;i<elements.size();i++)
                    {
                        auto el = elements[i];
                        if (el && el->getID() == ID)
                        {
                            el->parseMessage(tb);
                            break;
                        }
                    }
                }
            }
            break;
            default:
            {
                cerr << "vvTabletUI::updates: unknown Message type " << m.type << endl;
            }
            break;
            }
        }
    } while (gotMessage);

    return changed;
}

void vvTabletUI::addElement(vvTUIElement *e)
{
    elements.push_back(e);
    newElements.push_back(e);
}

void vvTabletUI::removeElement(vvTUIElement *e)
{
    {
        auto it = std::find(newElements.begin(), newElements.end(), e);
        if (it != newElements.end())
            newElements.erase(it);
    }
    {
        auto it = std::find(elements.begin(), elements.end(), e);
        if (it != elements.end())
            elements.erase(it);
    }
}


vvTUIGroupBox::vvTUIGroupBox(const std::string &n, int pID)
    : vvTUIElement(n, pID, TABLET_GROUPBOX)
{

}

vvTUIGroupBox::vvTUIGroupBox(vvTabletUI *tui, const std::string &n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_GROUPBOX)
{

}

#ifdef USE_QT
vvTUIGroupBox::vvTUIGroupBox(QObject *parent, const std::string &n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_GROUPBOX)
{

}
#endif

vvTUIGroupBox::~vvTUIGroupBox()
{

}

void vvTUIGroupBox::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletPressEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_DISACTIVATED)
    {
#ifdef USE_QT
        emit tabletEvent();
        emit tabletReleaseEvent();
#endif
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "vvTUIGroupBox::parseMessage: unknown event " << i << endl;
    }
}

vvTUIWebview::vvTUIWebview(const std::string& n, int pID) : vvTUIElement(n, pID, TABLET_WEBVIEW)
{
}

vvTUIWebview::vvTUIWebview(vvTabletUI* tui, const std::string& n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_WEBVIEW)
{
}

#ifdef USE_QT
vvTUIWebview::vvTUIWebview(QObject* parent, const std::string& n, int pID)
    : vvTUIElement(parent, n, pID, TABLET_WEBVIEW)
{
}
#endif

vvTUIWebview::~vvTUIWebview()
{
}

void vvTUIWebview::parseMessage(TokenBuffer& tb)
{
    fprintf(stderr, "vvTUIWebview::parseMessage\n");
    int i;
    tb >> i;
    //url speichern
    //getLoadedURL
#ifdef USE_QT
    emit tabletEvent();
#endif
    if (listener)
    {
        listener->tabletEvent(this);
    }

}

void vvTUIWebview::setURL(const std::string& url)
{
    setVal(url);  ///url is passed to virtual function setVal of baseclass
}

void vvTUIWebview::doSomething()
{
    fprintf(stderr, "message was send from tui to cover\n");
}
