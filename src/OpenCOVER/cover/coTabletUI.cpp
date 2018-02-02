/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>

#include <util/unixcompat.h>
#include <util/coTabletUIMessages.h>
#include "coTabletUI.h"
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/message_types.h>
#include <config/CoviseConfig.h>
#include "coVRPluginSupport.h"
#include "coVRSelectionManager.h"
#include "coVRMSController.h"
#include "coVRCommunication.h"
#include "coVRFileManager.h"
#include "coVRPluginList.h"

#include "coTUIFileBrowser/VRBData.h"
#include "coTUIFileBrowser/LocalData.h"
#include "coTUIFileBrowser/IRemoteData.h"
#include "coTUIFileBrowser/NetHelp.h"
#include "OpenCOVER.h"
#ifdef FB_USE_AG
#include "coTUIFileBrowser/AGData.h"
#endif

#include <QTextStream>
#include <QFile>

using namespace covise;
using namespace opencover;
//#define FILEBROWSER_DEBUG

coTUIButton::coTUIButton(const std::string &n, int pID)
: coTUIElement(n, pID, TABLET_BUTTON)
{
}

coTUIButton::coTUIButton(coTabletUI *tui, const std::string &n, int pID)
: coTUIElement(tui, n, pID, TABLET_BUTTON)
{
}

coTUIButton::coTUIButton(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_BUTTON)
{
}

coTUIButton::~coTUIButton()
{
}

void coTUIButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
        emit tabletEvent();
        emit tabletPressEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
        emit tabletEvent();
        emit tabletReleaseEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIButton::resend(bool create)
{
    coTUIElement::resend(create);
}

//TABLET_FILEBROWSER_BUTTON
coTUIFileBrowserButton::coTUIFileBrowserButton(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_FILEBROWSER_BUTTON)
{
    VRBData *locData = new VRBData(this);
    mLocalData = new LocalData(this);
    mData = NULL;
    this->mVRBCId = 0;
    mAGData = NULL;
    mMode = coTUIFileBrowserButton::OPEN;

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

coTUIFileBrowserButton::~coTUIFileBrowserButton()
{
    this->mFileList.clear();
    this->mDirList.clear();
}

void coTUIFileBrowserButton::setClientList(Message &msg)
{
    //transmits a list of vrb clients as received by VRBServer
    //to the filebrowser gui
    TokenBuffer tb(&msg);
    //std::string entry;
    char *entry = NULL;
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

void coTUIFileBrowserButton::parseMessage(TokenBuffer &tb)
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
        //File selected for opening in OpenCOVER
        char *cstrFile = NULL;
        char *cstrDirectory = NULL;
        int iLoadAll;
        std::string protocol;

        tb >> cstrFile;
        std::string strFile = cstrFile;
        tb >> cstrDirectory;
        tb >> iLoadAll;
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

        bool bLoadAll = (bool)iLoadAll;
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

        char *filter = NULL;
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
        char *dir = NULL;
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
        if (coVRCommunication::instance()->collaborative())
        {
            //Transmit Master/Slave state to TUI
            TokenBuffer rtb;
            rtb << TABLET_SET_VALUE;
            rtb << TABLET_SET_MASTER;
            rtb << ID;
            rtb << coVRCommunication::instance()->isMaster();
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
        char *location = NULL;
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
        char *location = NULL;
        char *file = NULL;
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

        if (vrbc != NULL && vrbc->isConnected())
        {
            rtb << (int)true;
        }
        else
        {
            rtb << (int)false;
        }

        tui()->send(rtb);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIFileBrowserButton::resend(bool create)
{
    coTUIElement::resend(create);
    TokenBuffer rt;

    // Send current Directory
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_CURDIR;
    rt << ID;
    rt << this->getData("vrb")->getCurrentPath().c_str();
    //std::cerr << "Resend: Current directory: " << this->mDataObj->getCurrentPath().c_str() << std::endl;
    tui()->send(rt);

    rt.delete_data();
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

    rt.delete_data();
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
    rt.delete_data();
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_MODE;
    rt << ID;
    rt << (int)this->mMode;

    tui()->send(rt);

    rt.delete_data();
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_FILTERLIST;
    rt << ID;
    rt << mFilterList.c_str();
    tui()->send(rt);
}

void coTUIFileBrowserButton::setFileList(Message &msg)
{
    TokenBuffer tb(&msg);
    //std::string entry;
    char *entry = NULL;
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

IData *coTUIFileBrowserButton::getData(std::string protocol)
{

    if (protocol.compare("") != 0)
    {
        return mDataRepo[protocol];
    }

    return mData;
}

IData *coTUIFileBrowserButton::getVRBData()
{
    return this->getData("vrb");
}

void coTUIFileBrowserButton::setDirList(Message &msg)
{
    TokenBuffer tb(&msg);
    char *entry;
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

void coTUIFileBrowserButton::setDrives(Message &ms)
{
    TokenBuffer tb(&ms);
    char *entry;
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

void coTUIFileBrowserButton::setCurDir(Message &msg)
{
    TokenBuffer tb(&msg);
    int subtype = 0;
    int id = 0;
    char *dirValue = NULL;

    tb >> subtype;
    tb >> id;
    tb >> dirValue;

    //Save current path to Data object
    this->setCurDir(dirValue);
}

void coTUIFileBrowserButton::setCurDir(const char *dir)
{
    //Save current path to Data object
    std::string sdir = std::string(dir);
    if ((sdir.compare("") || sdir.compare(".")) && (this->mLocation == this->mLocalIP))
    {
        sdir = this->mLocalData->resolveToAbsolute(std::string(dir));
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

void coTUIFileBrowserButton::sendList(TokenBuffer & /*tb*/)
{
}

//deprecated: Use global vrbc variable directly in VRBData
void coTUIFileBrowserButton::setVRBC(VRBClient * /* *client */)
{
    // Commenting out to see if, removal of code has side-effects
    /*VRBData *locData = static_cast<VRBData*>(this->mDataObj);
   if(client)
   {
      mVRBCId = client->getID();
      locData->setVRBC(client);
   }*/
}

std::string coTUIFileBrowserButton::getFilename(const std::string url)
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

void *coTUIFileBrowserButton::getFileHandle(bool sync)
{
    if (mData)
    {
        return this->mData->getTmpFileHandle(sync);
    }
    return NULL;
}

void coTUIFileBrowserButton::setMode(DialogMode mode)
{
    mMode = mode;
    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_MODE;
    rt << ID;
    rt << (int)mMode;

    tui()->send(rt);
}

// Method which is called from external of coTabletUI to allow
// OpenCOVER to initially set the range of available filter extensions
// used in the file dialog in the TUI.
void coTUIFileBrowserButton::setFilterList(std::string filterList)
{
    mFilterList = filterList;
    TokenBuffer rt;
    rt << TABLET_SET_VALUE;
    rt << TABLET_SET_FILTERLIST;
    rt << ID;
    rt << filterList.c_str();

    tui()->send(rt);
}

std::string coTUIFileBrowserButton::getSelectedPath()
{
    return this->mData->getSelectedPath();
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIColorTriangle::coTUIColorTriangle(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_COLOR_TRIANGLE)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
}

coTUIColorTriangle::coTUIColorTriangle(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_COLOR_TRIANGLE)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
}

coTUIColorTriangle::~coTUIColorTriangle()
{
}

void coTUIColorTriangle::parseMessage(TokenBuffer &tb)
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
            emit tabletReleaseEvent();
            if (listener)
                listener->tabletReleaseEvent(this);
        }
        if (j == TABLET_PRESSED)
        {
            emit tabletEvent();
            if (listener)
                listener->tabletEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIColorTriangle::setColor(float r, float g, float b)
{
    red = r;
    green = g;
    blue = b;
    setVal(TABLET_RED, r);
    setVal(TABLET_GREEN, g);
    setVal(TABLET_BLUE, b);
}

void coTUIColorTriangle::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(TABLET_RED, red);
    setVal(TABLET_GREEN, green);
    setVal(TABLET_BLUE, blue);
}

coTUIColorButton::coTUIColorButton(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_COLOR_BUTTON)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

coTUIColorButton::coTUIColorButton(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_COLOR_TRIANGLE)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

coTUIColorButton::~coTUIColorButton()
{
}

void coTUIColorButton::parseMessage(TokenBuffer &tb)
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
            emit tabletEvent();
            if (listener)
                listener->tabletReleaseEvent(this);
        }
        if (j == TABLET_PRESSED)
        {
            emit tabletReleaseEvent();
            if (listener)
                listener->tabletEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIColorButton::setColor(float r, float g, float b, float a)
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

void coTUIColorButton::resend(bool create)
{
    coTUIElement::resend(create);

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

//----------------------------------------------------------
//----------------------------------------------------------

coTUIColorTab::coTUIColorTab(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_COLOR_TAB)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

coTUIColorTab::coTUIColorTab(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_COLOR_TAB)
{
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    alpha = 1.0;
}

coTUIColorTab::~coTUIColorTab()
{
}

void coTUIColorTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_RGBA)
    {
        tb >> red;
        tb >> green;
        tb >> blue;
        tb >> alpha;

        emit tabletEvent();
        if (listener)
            listener->tabletEvent(this);
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIColorTab::setColor(float r, float g, float b, float a)
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

void coTUIColorTab::resend(bool create)
{
    coTUIElement::resend(create);
    setColor(red, green, blue, alpha);
}

//----------------------------------------------------------
//---------------------------------------------------------

coTUINav::coTUINav(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_NAV_ELEMENT)
{
    down = false;
    x = 0;
    y = 0;
}

coTUINav::~coTUINav()
{
}

void coTUINav::parseMessage(TokenBuffer &tb)
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
        cerr << "unknown event " << i << endl;
    }
}

void coTUINav::resend(bool create)
{
    coTUIElement::resend(create);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIBitmapButton::coTUIBitmapButton(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_BITMAP_BUTTON)
{
}

coTUIBitmapButton::coTUIBitmapButton(coTabletUI *tui, const std::string &n, int pID)
: coTUIElement(tui, n, pID, TABLET_BITMAP_BUTTON)
{
}

coTUIBitmapButton::coTUIBitmapButton(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_BITMAP_BUTTON)
{
}

coTUIBitmapButton::~coTUIBitmapButton()
{
}

void coTUIBitmapButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_PRESSED)
    {
        emit tabletEvent();
        emit tabletPressEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletPressEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
        emit tabletEvent();
        emit tabletReleaseEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIBitmapButton::resend(bool create)
{
    coTUIElement::resend(create);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUILabel::coTUILabel(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_TEXT_FIELD)
{
    color = Qt::black;
}

coTUILabel::coTUILabel(coTabletUI *tui, const std::string &n, int pID)
: coTUIElement(tui, n, pID, TABLET_TEXT_FIELD)
{
    color = Qt::black;
}

coTUILabel::coTUILabel(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_TEXT_FIELD)
{
    color = Qt::black;
}

coTUILabel::~coTUILabel()
{
}

void coTUILabel::resend(bool create)
{
    coTUIElement::resend(create);
    setColor(color);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUITabFolder::coTUITabFolder(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_TAB_FOLDER)
{
}

coTUITabFolder::coTUITabFolder(coTabletUI *tui, const std::string &n, int pID)
: coTUIElement(tui, n, pID, TABLET_TAB_FOLDER)
{
}

coTUITabFolder::coTUITabFolder(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_TAB_FOLDER)
{
}

coTUITabFolder::~coTUITabFolder()
{
}

void coTUITabFolder::parseMessage(TokenBuffer &tb)
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
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUITabFolder::resend(bool create)
{
    coTUIElement::resend(create);
}



//----------------------------------------------------------
//----------------------------------------------------------

coTUIUITab::coTUIUITab(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_UI_TAB)
{
}

coTUIUITab::coTUIUITab(coTabletUI *tui, const std::string &n, int pID)
: coTUIElement(tui, n, pID, TABLET_UI_TAB)
{

}

coTUIUITab::coTUIUITab(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_UI_TAB)
{
}

coTUIUITab::~coTUIUITab()
{
}

void coTUIUITab::parseMessage(TokenBuffer &tb)
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
        cerr << "coTUIUITab::parseMessage err: unknown event " << i << endl;
    }
}

void coTUIUITab::sendEvent(const QString &source, const QString &event)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_UI_COMMAND;
    tb << ID;
    tb << source.toStdString();
    tb << (event.size() + 1) * 2;
    tb.addBinary((const char *)event.utf16(), (event.size() + 1) * 2);

    tui()->send(tb);
}

void coTUIUITab::resend(bool create)
{
    coTUIElement::resend(create);
}

bool coTUIUITab::loadUIFile(const std::string &filename)
{
    QFile uiFile(QString::fromStdString(filename));

    if (!uiFile.exists())
    {
        std::cerr << "coTUIUITab::loadFile err: file " << filename << " does not exist" << std::endl;
        return false;
    }

    if (!uiFile.open(QIODevice::ReadOnly))
    {
        std::cerr << "coTUIUITab::loadFile err: cannot open file " << filename << std::endl;
        return false;
    }

    QTextStream inStream(&uiFile);

    this->uiDescription = inStream.readAll();

    QString jsFileName(QString::fromStdString(filename) + ".js");
    QFile jsFile(jsFileName);
    if (jsFile.exists() && jsFile.open(QIODevice::ReadOnly))
    {
        opencover::coVRFileManager::instance()->loadFile(jsFileName.toLocal8Bit().data());
    }

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_UI_USE_DESCRIPTION;
    tb << ID;
    tb << (this->uiDescription.size() + 1) * 2;
    tb.addBinary((const char *)this->uiDescription.utf16(), (this->uiDescription.size() + 1) * 2);

    tui()->send(tb);

    return true;
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUITab::coTUITab(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_TAB)
{
}

coTUITab::coTUITab(coTabletUI *tui, const std::string &n, int pID)
    : coTUIElement(tui, n, pID, TABLET_TAB)
{

}

coTUITab::coTUITab(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_TAB)
{
}

coTUITab::~coTUITab()
{
}

void coTUITab::parseMessage(TokenBuffer &tb)
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
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUITab::resend(bool create)
{
    coTUIElement::resend(create);
}

//----------------------------------------------------------
//----------------------------------------------------------

void TextureThread::msleep(int msec)
{
    usleep(msec * 1000);
}

void TextureThread::run()
{
    while (running)
    {
        while (!tab->getConnection())
        {
            this->tab->tryConnect();
            msleep(250);
        }
        //if(type == TABLET_TEX_UPDATE)
        {
            int count = 0;
            while (!tab->queueIsEmpty())
            {
                tab->sendTexture();
                sendedTextures++;

                //cout << " sended textures : " << sendedTextures << "\n";
                if (finishedTraversing && textureListCount == sendedTextures)
                {
                    //cout << " finished sending with: " << textureListCount << "  textures \n";
                    tab->sendTraversedTextures();
                    //type = THREAD_NOTHING_TO_DO;
                    finishedTraversing = false;
                    textureListCount = 0;
                    sendedTextures = 0;
                }
                if (count++ == 5)
                    break;
            }
            //msleep(100);
        }
        if (tab->getTexturesToChange()) //still Textures in queue
        {
            int count = 0;
            Message m;
            // waiting for incoming data
            while (count < 200)
            {
                count++;

                if (tab->getConnection())
                {
                    // message arrived
                    if (tab->getConnection()->check_for_input())
                    {
                        tab->getConnection()->recv_msg(&m);
                        TokenBuffer tokenbuffer(&m);
                        if (m.type == COVISE_MESSAGE_TABLET_UI)
                        {
                            int ID;
                            tokenbuffer >> ID;
                            tab->parseTextureMessage(tokenbuffer);
                            tab->decTexturesToChange();
                            //type = THREAD_NOTHING_TO_DO;
                            break;
                        }
                    }
                }
                msleep(50);
            }
        }
        else
            msleep(50);
    }
}

coTUITextureTab::coTUITextureTab(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_TEXTURE_TAB)
{
    conn = NULL;
    currentNode = 0;
    changedNode = 0;
    texturesToChange = 0;
    texturePort = coCoviseConfig::getInt("port", "COVER.TabletPC.Server", 31802);
    texturePort++;
    thread = new TextureThread(this);
    thread->setType(THREAD_NOTHING_TO_DO);
    thread->start();
}

coTUITextureTab::~coTUITextureTab()
{
    thread->terminateTextureThread();

    while (thread->isRunning())
    {
        sleep(1);
    }
    delete thread;
    delete conn;
    conn = NULL;
}

void coTUITextureTab::finishedTraversing()
{
    thread->traversingFinished();
}

void coTUITextureTab::incTextureListCount()
{
    thread->incTextureListCount();
}

void coTUITextureTab::sendTraversedTextures()
{
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_TRAVERSED_TEXTURES;
    t << ID;
    this->send(t);
}

void coTUITextureTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_TEX_UPDATE)
    {
        thread->setType(TABLET_TEX_UPDATE);
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    if (i == TABLET_TEX_PORT)
    {
        cerr << "newPort " << texturePort << endl;
        tb >> texturePort;
    }
    else if (i == TABLET_TEX_CHANGE)
    {
        //cout << " currentNode : " << currentNode << "\n";
        if (currentNode)
        {
            // let the tabletui know that it can send texture data now
            TokenBuffer t;
            int buttonNumber;
            tb >> buttonNumber;
            t << TABLET_SET_VALUE;
            t << TABLET_TEX_CHANGE;
            t << ID;
            t << buttonNumber;
            t << (uint64_t)(uintptr_t)currentNode;
            tui()->send(t);
            //thread->setType(TABLET_TEX_CHANGE);
            texturesToChange++;
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUITextureTab::parseTextureMessage(TokenBuffer &tb)
{
    int type;
    tb >> type;
    if (type == TABLET_TEX_CHANGE)
    {
        uint64_t node;
        tb >> textureNumber;
        tb >> textureMode;
        tb >> textureTexGenMode;
        tb >> alpha;
        tb >> height;
        tb >> width;
        tb >> depth;
        tb >> dataLength;
        tb >> node;

        changedNode = (osg::Node *)(uintptr_t)node;
        if (changedNode)
        {
            data = new char[dataLength];

            for (int k = 0; k < dataLength; k++)
            {
                if ((k % 4) == 3)
                    tb >> data[k];
                else if ((k % 4) == 2)
                    tb >> data[k - 2];
                else if ((k % 4) == 1)
                    tb >> data[k];
                else if ((k % 4) == 0)
                    tb >> data[k + 2];
            }
            if (listener)
            {
                listener->tabletEvent(this);
                listener->tabletPressEvent(this);
            }
        }
    }
}

void coTUITextureTab::sendTexture()
{
    mutex.lock();
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TEX;
    tb << ID;
    tb << heightList.front();
    tb << widthList.front();
    tb << depthList.front();
    tb << lengthList.front();

    int length = heightList.front() * widthList.front() * depthList.front() / 8;
    tb.addBinary(dataList.front(), length);
    this->send(tb);
    heightList.pop();
    widthList.pop();
    depthList.pop();
    lengthList.pop();
    dataList.pop();
    mutex.unlock();
}

void coTUITextureTab::setTexture(int height, int width, int depth, int dataLength, const char *data)
{
    mutex.lock();
    heightList.push(height);
    widthList.push(width);
    depthList.push(depth);
    lengthList.push(dataLength);
    dataList.push(data);
    thread->incTextureListCount();
    //cout << " added texture : \n";
    mutex.unlock();
}

void coTUITextureTab::setTexture(int texNumber, int mode, int texGenMode)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TEX_MODE;
    tb << ID;
    tb << texNumber;
    tb << mode;
    tb << texGenMode;

    tui()->send(tb);
}

void coTUITextureTab::resend(bool create)
{
    coTUIElement::resend(create);
}

void coTUITextureTab::tryConnect()
{
    serverHost = NULL;
    localHost = new Host("localhost");

    timeout = coCoviseConfig::getFloat("COVER.TabletPC.Timeout", 0.0);
    serverHost = tui()->getServerHost();
    conn = new ClientConnection(serverHost, texturePort, 0, (sender_type)0, 0);
    if (!conn->is_connected()) // could not open server port
    {
#ifndef _WIN32
        if (errno != ECONNREFUSED)
        {
            fprintf(stderr, "Could not connect to TabletPC %s; port %d: %s\n",
                    localHost->getName(), texturePort, strerror(errno));
        }
#else
        fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", localHost->getName(), texturePort);
#endif
        delete conn;
        conn = NULL;

        conn = new ClientConnection(localHost, texturePort, 0, (sender_type)0, 0);
        if (!conn->is_connected()) // could not open server port
        {
#ifndef _WIN32
            if (errno != ECONNREFUSED)
            {
                fprintf(stderr, "Could not connect to TabletPC %s; port %d: %s\n",
                        localHost->getName(), texturePort, strerror(errno));
            }
#else
            fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", localHost->getName(), texturePort);
#endif
            delete conn;
            conn = NULL;
        }
    }
}

void coTUITextureTab::send(TokenBuffer &tb)
{
    if (conn == NULL)
        return;
    Message m(tb);
    m.type = COVISE_MESSAGE_TABLET_UI;
    conn->send_msg(&m);
}

//----------------------------------------------------------
//----------------------------------------------------------

void SGTextureThread::msleep(int msec)
{
    usleep(msec * 1000);
}

void SGTextureThread::run()
{
    while (running)
    {
        //if(type == TABLET_TEX_UPDATE)
        {
            int count = 0;
            while (!tab->queueIsEmpty())
            {
                tab->sendTexture();
                sendedTextures++;

                //cout << " sended textures : " << sendedTextures << "\n";
                if (finishedTraversing && textureListCount == sendedTextures)
                {
                    //cout << " finished sending with: " << textureListCount << "  textures \n";
                    tab->sendTraversedTextures();
                    //type = THREAD_NOTHING_TO_DO;
                    finishedTraversing = false;
                    textureListCount = 0;
                    sendedTextures = 0;
                }
                if (finishedNode && textureListCount == sendedTextures)
                {
                    //cout << " finished sending with: " << textureListCount << "  textures \n";
                    tab->sendNodeTextures();

                    finishedNode = false;
                    textureListCount = 0;
                    sendedTextures = 0;
                }
                if (count++ == 5)
                    break;
            }
        }
        /*if(noTextures)
      {
         tab->sendNoTextures();
         noTextures = false;
      }*/
        if (tab->getTexturesToChange()) //still Textures in queue
        {
            int count = 0;
            Message m;
            // waiting for incoming data
            while (count < 200)
            {
                count++;

                if (tab->getConnection())
                {
                    // message arrived
                    if (tab->getConnection()->check_for_input())
                    {
                        tab->getConnection()->recv_msg(&m);
                        TokenBuffer tokenbuffer(&m);
                        if (m.type == COVISE_MESSAGE_TABLET_UI)
                        {
                            int ID;
                            tokenbuffer >> ID;
                            tab->parseTextureMessage(tokenbuffer);
                            tab->decTexturesToChange();
                            //type = THREAD_NOTHING_TO_DO;
                            break;
                        }
                    }
                }
                msleep(50);
            }
        }
        else
            msleep(50);
    }
}

//----------------------------------------------------------

coTUISGBrowserTab::coTUISGBrowserTab(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_BROWSER_TAB)
{
    thread = NULL;

    conn = NULL;
    currentNode = 0;
    changedNode = 0;
    texturesToChange = 0;

    texturePort = 0;
    // next port is for Texture communication

    currentPath = "";
    loadFile = false; //gottlieb
}

int coTUISGBrowserTab::openServer()
{
    conn = NULL;

    ServerConnection *sConn = new ServerConnection(&texturePort, 0, (sender_type)0);
    sConn->listen();
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TEX_PORT;
    tb << ID;
    tb << texturePort;

    tui()->send(tb);

    if (sConn->acceptOne(60) < 0)
    {
        fprintf(stderr, "Could not open server port %d\n", texturePort);
        delete sConn;
        sConn = NULL;
        return (-1);
    }
    if (!sConn->getSocket())
    {
        fprintf(stderr, "Could not open server port %d\n", texturePort);
        delete sConn;
        sConn = NULL;
        return (-1);
    }

    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    setsockopt(sConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

    if (!sConn->is_connected()) // could not open server port
    {
        fprintf(stderr, "Could not open server port %d\n", texturePort);
        delete sConn;
        sConn = NULL;
        return (-1);
    }
    conn = sConn;
    return 0;
}

coTUISGBrowserTab::~coTUISGBrowserTab()
{
    if (thread)
    {
        thread->terminateTextureThread();

        while (thread->isRunning())
        {
            sleep(1);
        }
        delete thread;
    }
    delete conn;
    conn = NULL;
}

void coTUISGBrowserTab::noTexture()
{
    thread->noTexturesFound(true);
}

void coTUISGBrowserTab::finishedNode()
{
    thread->nodeFinished(true);
}

void coTUISGBrowserTab::sendNodeTextures()
{
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_NODE_TEXTURES;
    t << ID;
    this->send(t);
}
//gottlieb<
void coTUISGBrowserTab::loadFilesFlag(bool state)
{
    loadFile = state;
    if (loadFile)
        setVal(TABLET_LOADFILE_SATE, 1);
    else
        setVal(TABLET_LOADFILE_SATE, 0);
}
void coTUISGBrowserTab::hideSimNode(bool state, char *nodePath, char *simPath)
{
    if (state)
    {
        setVal(TABLET_SIM_SHOW_HIDE, 0, nodePath, simPath);
    }
    else
    {
        setVal(TABLET_SIM_SHOW_HIDE, 1, nodePath, simPath);
    }
}
void coTUISGBrowserTab::setSimPair(char *nodePath, char *simPath, char *simName)
{
    setVal(TABLET_SIM_SETSIMPAIR, nodePath, simPath, simName);
}
//>gottlieb
void coTUISGBrowserTab::sendNoTextures()
{
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_NO_TEXTURES;
    t << ID;
    this->send(t);
}

void coTUISGBrowserTab::finishedTraversing()
{
    thread->traversingFinished(true);
}

void coTUISGBrowserTab::incTextureListCount()
{
    thread->incTextureListCount();
}

void coTUISGBrowserTab::sendTraversedTextures()
{
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_TRAVERSED_TEXTURES;
    t << ID;
    this->send(t);
}

void coTUISGBrowserTab::send(TokenBuffer &tb)
{
    if (conn == NULL)
        return;
    Message m(tb);
    m.type = COVISE_MESSAGE_TABLET_UI;
    conn->send_msg(&m);
}

void coTUISGBrowserTab::tryConnect()
{
    ClientConnection *cConn = NULL;
    tui()->lock();
    if (tui()->connectedHost)
    {
        //timeout = coCoviseConfig::getFloat("COVER.TabletPC.Timeout", 0.0);

        cConn = new ClientConnection(tui()->connectedHost, texturePort, 0, (sender_type)0, 10, 2.0);
        if (!cConn->is_connected()) // could not open server port
        {
#ifndef _WIN32
            if (errno != ECONNREFUSED)
            {
                fprintf(stderr, "Could not connect to TabletPC %s; port %d: %s\n",
                        tui()->connectedHost->getName(), texturePort, strerror(errno));
            }
#else
            fprintf(stderr, "Could not connect to TextureThread %s; port %d\n", tui()->connectedHost->getName(), texturePort);
#endif
            delete cConn;
            cConn = NULL;
        }
    }
    conn = cConn;
    tui()->unlock();
}

void coTUISGBrowserTab::parseTextureMessage(TokenBuffer &tb)
{
    int type;
    tb >> type;
    if (type == TABLET_TEX_CHANGE)
    {
        char *path;
        tb >> textureNumber;
        tb >> textureMode;
        tb >> textureTexGenMode;
        tb >> alpha;
        tb >> height;
        tb >> width;
        tb >> depth;
        tb >> dataLength;
        tb >> path;
        tb >> index;

        //changedNode = (osg::Node *)(uintptr_t)node;
        changedPath = std::string(path);
        if (currentNode)
        {
            changedNode = currentNode;
            data = new char[dataLength];

            for (int k = 0; k < dataLength; k++)
            {
                if ((k % 4) == 3)
                    tb >> data[k];
                else if ((k % 4) == 2)
                    tb >> data[k - 2];
                else if ((k % 4) == 1)
                    tb >> data[k];
                else if ((k % 4) == 0)
                    tb >> data[k + 2];
            }
            if (listener)
            {
                listener->tabletEvent(this);
            }
        }
    }
}

void coTUISGBrowserTab::sendTexture()
{
    mutex.lock();
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TEX;
    tb << ID;
    tb << _heightList.front();
    tb << _widthList.front();
    tb << _depthList.front();
    tb << _indexList.front();
    tb << _lengthList.front();

    int length = _heightList.front() * _widthList.front() * _depthList.front() / 8;
    tb.addBinary(_dataList.front(), length);
    this->send(tb);
    _heightList.pop();
    _widthList.pop();
    _depthList.pop();
    _indexList.pop();
    _lengthList.pop();
    _dataList.pop();
    mutex.unlock();
}

void coTUISGBrowserTab::setTexture(int height, int width, int depth, int texIndex, int dataLength, const char *data)
{
    mutex.lock();
    _heightList.push(height);
    _widthList.push(width);
    _depthList.push(depth);
    _indexList.push(texIndex);
    _lengthList.push(dataLength);
    _dataList.push(data);
    thread->incTextureListCount();
    //cout << " added texture : \n";
    mutex.unlock();
}

void coTUISGBrowserTab::setTexture(int texNumber, int mode, int texGenMode, int texIndex)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TEX_MODE;
    tb << ID;
    tb << texNumber;
    tb << mode;
    tb << texGenMode;
    tb << texIndex;

    tui()->send(tb);
}

void coTUISGBrowserTab::sendType(int type, const char *nodeType, const char *name, const char *path, const char *pPath, int mode, int numChildren)
{

    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_NODE;
    tb << ID;
    tb << type;
    tb << name;
    tb << nodeType;
    tb << mode;
    tb << path;
    tb << pPath;
    tb << numChildren;

    tui()->send(tb);
}

void coTUISGBrowserTab::sendEnd()
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_END;
    tb << ID;
    tui()->send(tb);
}

void coTUISGBrowserTab::sendProperties(std::string path, std::string pPath, int mode, int transparent)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    int i;
    const char *currentPath = path.c_str();
    const char *currentpPath = pPath.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << ID;
    tb << currentPath;
    tb << currentpPath;
    tb << mode;
    tb << transparent;
    for (i = 0; i < 4; i++)
        tb << diffuse[i];
    for (i = 0; i < 4; i++)
        tb << specular[i];
    for (i = 0; i < 4; i++)
        tb << ambient[i];
    for (i = 0; i < 4; i++)
        tb << emissive[i];

    tui()->send(tb);
}

void coTUISGBrowserTab::sendProperties(std::string path, std::string pPath, int mode, int transparent, float mat[])
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    int i;
    const char *currentPath = path.c_str();
    const char *currentpPath = pPath.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << ID;
    tb << currentPath;
    tb << currentpPath;
    tb << mode;
    tb << transparent;
    for (i = 0; i < 4; i++)
        tb << diffuse[i];
    for (i = 0; i < 4; i++)
        tb << specular[i];
    for (i = 0; i < 4; i++)
        tb << ambient[i];
    for (i = 0; i < 4; i++)
        tb << emissive[i];

    for (i = 0; i < 16; ++i)
    {
        tb << mat[i];
    }

    tui()->send(tb);
}

void coTUISGBrowserTab::sendCurrentNode(osg::Node *node, std::string path)
{
    currentNode = node;
    visitorMode = CURRENT_NODE;

    if (listener)
        listener->tabletCurrentEvent(this);

    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    const char *currentPath = path.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_CURRENT_NODE;
    tb << ID;
    tb << currentPath;

    tui()->send(tb);
}

void coTUISGBrowserTab::sendRemoveNode(std::string path, std::string parentPath)
{

    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    const char *removePath = path.c_str();
    const char *removePPath = parentPath.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_REMOVE_NODE;
    tb << ID;
    tb << removePath;
    tb << removePPath;

    tui()->send(tb);
}

void coTUISGBrowserTab::sendShader(std::string name)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    const char *shaderName = name.c_str();

    tb << TABLET_SET_VALUE;
    tb << GET_SHADER;
    tb << ID;
    tb << shaderName;

    tui()->send(tb);
}

void coTUISGBrowserTab::sendUniform(std::string name, std::string type, std::string value, std::string min, std::string max, std::string textureFile)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << GET_UNIFORMS;
    tb << ID;
    tb << name.c_str();
    tb << type.c_str();
    tb << value.c_str();
    tb << min.c_str();
    tb << max.c_str();
    tb << textureFile.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::sendShaderSource(std::string vertex, std::string fragment, std::string geometry, std::string tessControl, std::string tessEval)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << GET_SOURCE;
    tb << ID;
    tb << vertex.c_str();
    tb << fragment.c_str();
    tb << geometry.c_str();
    tb << tessControl.c_str();
    tb << tessEval.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::updateUniform(std::string shader, std::string name, std::string value, std::string textureFile)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_UNIFORM;
    tb << ID;
    tb << shader.c_str();
    tb << name.c_str();
    tb << value.c_str();
    tb << textureFile.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::updateShaderSourceV(std::string shader, std::string vertex)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_VERTEX;
    tb << ID;
    tb << shader.c_str();
    tb << vertex.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::updateShaderSourceTC(std::string shader, std::string tessControl)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_TESSCONTROL;
    tb << ID;
    tb << shader.c_str();
    tb << tessControl.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::updateShaderSourceTE(std::string shader, std::string tessEval)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_TESSEVAL;
    tb << ID;
    tb << shader.c_str();
    tb << tessEval.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::updateShaderSourceF(std::string shader, std::string fragment)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_FRAGMENT;
    tb << ID;
    tb << shader.c_str();
    tb << fragment.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::updateShaderSourceG(std::string shader, std::string geometry)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_GEOMETRY;
    tb << ID;
    tb << shader.c_str();
    tb << geometry.c_str();

    tui()->send(tb);
}

void coTUISGBrowserTab::updateShaderNumVertices(std::string shader, int value)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << SET_NUM_VERT;
    tb << ID;
    tb << shader.c_str();
    tb << value;

    tui()->send(tb);
}

void coTUISGBrowserTab::updateShaderOutputType(std::string shader, int value)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << SET_OUTPUT_TYPE;
    tb << ID;
    tb << shader.c_str();
    tb << value;

    tui()->send(tb);
}
void coTUISGBrowserTab::updateShaderInputType(std::string shader, int value)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << SET_INPUT_TYPE;
    tb << ID;
    tb << shader.c_str();
    tb << value;

    tui()->send(tb);
}

void coTUISGBrowserTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    switch (i)
    {

    case TABLET_INIT_SECOND_CONNECTION:
    {

        if (tui()->serverMode)
        {
            openServer();
        }
        if (thread)
        {
            thread->terminateTextureThread();

            while (thread->isRunning())
            {
                sleep(1);
            }
            delete thread;
        }

        thread = new SGTextureThread(this);
        thread->setType(THREAD_NOTHING_TO_DO);
        thread->traversingFinished(false);
        thread->nodeFinished(false);
        thread->noTexturesFound(false);
        thread->start();
        break;
    }

    case TABLET_BROWSER_PROPERTIES:
    {
        if (listener)
            listener->tabletDataEvent(this, tb);
        break;
    }
    case TABLET_BROWSER_UPDATE:
    {
        visitorMode = UPDATE_NODES;
        if (listener)
            listener->tabletPressEvent(this);
        break;
    }
    case TABLET_BROWSER_EXPAND_UPDATE:
    {
        visitorMode = UPDATE_EXPAND;
        char *path;
        tb >> path;
        expandPath = std::string(path);

        if (listener)
            listener->tabletPressEvent(this);
        break;
    }
    case TABLET_BROWSER_SELECTED_NODE:
    {
        visitorMode = SET_SELECTION;
        char *path, *pPath;
        tb >> path;
        selectPath = std::string(path);
        tb >> pPath;
        selectParentPath = std::string(pPath);
        if (listener)
            listener->tabletSelectEvent(this);
        break;
    }
    case TABLET_BROWSER_CLEAR_SELECTION:
    {
        visitorMode = CLEAR_SELECTION;
        if (listener)
            listener->tabletSelectEvent(this);

        break;
    }
    case TABLET_BROWSER_SHOW_NODE:
    {
        visitorMode = SHOW_NODE;
        char *path, *pPath;
        tb >> path;
        showhidePath = std::string(path);
        tb >> pPath;
        showhideParentPath = std::string(pPath);

        if (listener)
            listener->tabletSelectEvent(this);
        break;
    }
    case TABLET_BROWSER_HIDE_NODE:
    {
        visitorMode = HIDE_NODE;
        char *path, *pPath;
        tb >> path;
        showhidePath = std::string(path);
        tb >> pPath;
        showhideParentPath = std::string(pPath);

        if (listener)
            listener->tabletSelectEvent(this);
        break;
    }
    case TABLET_BROWSER_COLOR:
    {
        visitorMode = UPDATE_COLOR;
        tb >> ColorR;
        tb >> ColorG;
        tb >> ColorB;
        if (listener)
            listener->tabletChangeModeEvent(this);
        break;
    }
    case TABLET_BROWSER_WIRE:
    {
        visitorMode = UPDATE_WIRE;
        tb >> polyMode;
        if (listener)
            listener->tabletChangeModeEvent(this);
        break;
    }
    case TABLET_BROWSER_SEL_ONOFF:
    {
        visitorMode = UPDATE_SEL;
        tb >> selOnOff;
        if (listener)
            listener->tabletChangeModeEvent(this);
        break;
    }
    case TABLET_BROWSER_FIND:
    {
        char *fname;
        visitorMode = FIND_NODE;
        tb >> fname;
        findName = std::string(fname);
        if (listener)
            listener->tabletFindEvent(this);
        break;
    }
    case TABLET_BROWSER_LOAD_FILES:
    {
        char *fname;
        tb >> fname;

        if (listener)
            listener->tabletLoadFilesEvent(fname);
        break;
    }
    break;
    case TABLET_TEX_UPDATE:
    {
        thread->setType(TABLET_TEX_UPDATE);
        sendImageMode = SEND_IMAGES;
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
        break;
    }
    case TABLET_TEX_PORT:
    {
        tb >> texturePort;
        tryConnect();
        break;
    }
    case TABLET_TEX_CHANGE:
    {
        //cout << " currentNode : " << currentNode << "\n";
        if (currentNode)
        {
            // let the tabletui know that it can send texture data now
            TokenBuffer t;
            int buttonNumber;
            const char *path = currentPath.c_str();
            tb >> buttonNumber;
            t << TABLET_SET_VALUE;
            t << TABLET_TEX_CHANGE;
            t << ID;
            t << buttonNumber;
            t << path;
            tui()->send(t);
            //thread->setType(TABLET_TEX_CHANGE);
            texturesToChange++;
        }
        break;
    }
    default:
    {
        cerr << "unknown event " << i << endl;
    }
    }
}

void coTUISGBrowserTab::resend(bool create)
{
    if (thread)
    {
        thread->terminateTextureThread();

        while (thread->isRunning())
        {
            sleep(1);
        }
        delete thread;
        delete conn;
        conn = NULL;
    }

    thread = new SGTextureThread(this);
    thread->setType(THREAD_NOTHING_TO_DO);
    thread->traversingFinished(false);
    thread->nodeFinished(false);
    thread->noTexturesFound(false);
    thread->start();

    coTUIElement::resend(create);

    sendImageMode = SEND_LIST;
    if (listener)
    {
        listener->tabletEvent(this);
        listener->tabletReleaseEvent(this);
    }
    //gottlieb<
    if (loadFile)
        setVal(TABLET_LOADFILE_SATE, 1);
    else
        setVal(TABLET_LOADFILE_SATE, 0); //>gottlieb
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIAnnotationTab::coTUIAnnotationTab(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_ANNOTATION_TAB)
{
}

coTUIAnnotationTab::~coTUIAnnotationTab()
{
}

void coTUIAnnotationTab::resend(bool create)
{
    //std::cout << "coTUIAnnotationTab::resend()" << std::endl;
    coTUIElement::resend(create);
}

void coTUIAnnotationTab::parseMessage(TokenBuffer &tb)
{
    listener->tabletDataEvent(this, tb);
}

void coTUIAnnotationTab::setNewButtonState(bool state)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ANNOTATION_CHANGE_NEW_BUTTON_STATE;
    tb << ID;
    tb << (char)state;

    tui()->send(tb);
}

void coTUIAnnotationTab::addAnnotation(int id)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ANNOTATION_NEW;
    tb << ID;
    tb << id;

    tui()->send(tb);
}

void coTUIAnnotationTab::deleteAnnotation(int mode, int id)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ANNOTATION_DELETE;
    tb << ID;
    tb << mode;
    tb << id;

    tui()->send(tb);
}

void coTUIAnnotationTab::setSelectedAnnotation(int id)
{
    if (tui()->conn == NULL)
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

coTUIFunctionEditorTab::coTUIFunctionEditorTab(const char *tabName, int pID)
    : coTUIElement(tabName, pID, TABLET_FUNCEDIT_TAB)
{
    tfDim = 1;
    histogramData = NULL;
}

coTUIFunctionEditorTab::~coTUIFunctionEditorTab()
{
    if (histogramData)
        delete[] histogramData;
}

int coTUIFunctionEditorTab::getDimension() const
{
    return tfDim;
    ;
}

void coTUIFunctionEditorTab::setDimension(int dim)
{
    tfDim = dim;
}

void coTUIFunctionEditorTab::resend(bool create)
{
    coTUIElement::resend(create);

    //resend the transfer function information
    if (tui()->conn == NULL)
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

void coTUIFunctionEditorTab::sendHistogramData()
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

void coTUIFunctionEditorTab::parseMessage(TokenBuffer &tb)
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
coTUISplitter::coTUISplitter(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_SPLITTER)
{
    shape = coTUIFrame::StyledPanel;
    style = coTUIFrame::Sunken;
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}

coTUISplitter::coTUISplitter(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_SPLITTER)
{
    shape = coTUIFrame::StyledPanel;
    style = coTUIFrame::Sunken;
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}

coTUISplitter::~coTUISplitter()
{
}

void coTUISplitter::parseMessage(TokenBuffer &tb)
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
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUISplitter::resend(bool create)
{
    coTUIElement::resend(create);
    setShape(shape);
    setStyle(style);
    setOrientation(orientation);
}

void coTUISplitter::setShape(int s)
{
    TokenBuffer tb;
    shape = s;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SHAPE;
    tb << ID;
    tb << shape;
    tui()->send(tb);
}

void coTUISplitter::setStyle(int t)
{
    TokenBuffer tb;
    style = t;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STYLE;
    tb << ID;
    tb << (style | shape);
    tui()->send(tb);
}

void coTUISplitter::setOrientation(int orient)
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

coTUIFrame::coTUIFrame(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_FRAME)
{
    style = Sunken;
    shape = StyledPanel;
    setShape(shape);
    setStyle(style);
}

coTUIFrame::coTUIFrame(coTabletUI *tui, const std::string &n, int pID)
    : coTUIElement(tui, n, pID, TABLET_FRAME)
{
    style = Sunken;
    shape = StyledPanel;
    setShape(shape);
    setStyle(style);
}

coTUIFrame::coTUIFrame(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_FRAME)
{
    style = Sunken;
    shape = StyledPanel;
    setShape(shape);
    setStyle(style);
}

coTUIFrame::~coTUIFrame()
{
}

void coTUIFrame::parseMessage(TokenBuffer &tb)
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
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIFrame::setShape(int s)
{
    TokenBuffer tb;
    shape = s;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SHAPE;
    tb << ID;
    tb << shape;
    tui()->send(tb);
}

void coTUIFrame::setStyle(int t)
{
    TokenBuffer tb;
    style = t;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STYLE;
    tb << ID;
    tb << (style | shape);
    tui()->send(tb);
}

void coTUIFrame::resend(bool create)
{
    coTUIElement::resend(create);
    setShape(shape);
    setStyle(style);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIToggleButton::coTUIToggleButton(const std::string &n, int pID, bool s)
    : coTUIElement(n, pID, TABLET_TOGGLE_BUTTON)
{
    state = s;
    setVal(state);
}

coTUIToggleButton::coTUIToggleButton(coTabletUI *tui, const std::string &n, int pID, bool s)
    : coTUIElement(tui, n, pID, TABLET_TOGGLE_BUTTON)
{
    state = s;
    setVal(state);
}

coTUIToggleButton::coTUIToggleButton(QObject *parent, const std::string &n, int pID, bool s)
    : coTUIElement(parent, n, pID, TABLET_TOGGLE_BUTTON)
{
    state = s;
    setVal(state);
}

coTUIToggleButton::~coTUIToggleButton()
{
}

void coTUIToggleButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        state = true;
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
        state = false;
        emit tabletEvent();
        emit tabletReleaseEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIToggleButton::setState(bool s)
{
    if (s != state) // don't send unnecessary state changes
    {
        state = s;
        setVal(state);
    }
}

bool coTUIToggleButton::getState() const
{
    return state;
}

void coTUIToggleButton::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(state);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIToggleBitmapButton::coTUIToggleBitmapButton(const std::string &n, const std::string &down, int pID, bool state)
    : coTUIElement(n, pID, TABLET_BITMAP_TOGGLE_BUTTON)
{
    bmpUp = n;
    bmpDown = down;

    setVal(bmpDown);
    setVal(state);
}

coTUIToggleBitmapButton::coTUIToggleBitmapButton(QObject *parent, const std::string &n, const std::string &down, int pID, bool state)
    : coTUIElement(parent, n, pID, TABLET_BITMAP_TOGGLE_BUTTON)
{
    bmpUp = n;
    bmpDown = down;

    setVal(bmpDown);
    setVal(state);
}

coTUIToggleBitmapButton::~coTUIToggleBitmapButton()
{
}

void coTUIToggleBitmapButton::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    if (i == TABLET_ACTIVATED)
    {
        state = true;
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
        state = false;
        emit tabletEvent();
        emit tabletReleaseEvent();
        if (listener)
        {
            listener->tabletEvent(this);
            listener->tabletReleaseEvent(this);
        }
    }
    else
    {
        cerr << "unknown event " << i << endl;
    }
}

void coTUIToggleBitmapButton::setState(bool s)
{
    if (s != state) // don't send unnecessary state changes
    {
        state = s;
        setVal(state);
    }
}

bool coTUIToggleBitmapButton::getState() const
{
    return state;
}

void coTUIToggleBitmapButton::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(bmpDown);
    setVal(state);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIMessageBox::coTUIMessageBox(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_MESSAGE_BOX)
{
}

coTUIMessageBox::coTUIMessageBox(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_MESSAGE_BOX)
{
}

coTUIMessageBox::~coTUIMessageBox()
{
}

void coTUIMessageBox::resend(bool create)
{
    coTUIElement::resend(create);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIEditField::coTUIEditField(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_EDIT_FIELD)
{
    this->text = name;
    immediate = false;
}

coTUIEditField::coTUIEditField(coTabletUI *tui, const std::string &n, int pID)
    : coTUIElement(tui, n, pID, TABLET_EDIT_FIELD)
{
    this->text = name;
    immediate = false;
}

coTUIEditField::coTUIEditField(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_EDIT_FIELD)
{
    this->text = name;
    immediate = false;
}

coTUIEditField::~coTUIEditField()
{
}

void coTUIEditField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIEditField::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    text = m;
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUIEditField::setPasswordMode(bool b)
{
    setVal(TABLET_ECHOMODE, (int)b);
}

void coTUIEditField::setIPAddressMode(bool b)
{
    setVal(TABLET_IPADDRESS, (int)b);
}

void coTUIEditField::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

const std::string &coTUIEditField::getText() const
{
    return text;
}

void coTUIEditField::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(text);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------
//##########################################################

coTUIEditTextField::coTUIEditTextField(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_TEXT_EDIT_FIELD)
{
    text = name;
    immediate = false;
}

coTUIEditTextField::coTUIEditTextField(coTabletUI *tui, const std::string &n, int pID)
    : coTUIElement(tui, n, pID, TABLET_TEXT_EDIT_FIELD)
{
    text = name;
    immediate = false;
}

coTUIEditTextField::coTUIEditTextField(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_TEXT_EDIT_FIELD)
{
    text = name;
    immediate = false;
}

coTUIEditTextField::~coTUIEditTextField()
{
}

void coTUIEditTextField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIEditTextField::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    text = m;
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUIEditTextField::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

const std::string &coTUIEditTextField::getText() const
{
    return text;
}

void coTUIEditTextField::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(text);
    setVal(immediate);
}

//##########################################################
//----------------------------------------------------------
//----------------------------------------------------------

coTUIEditIntField::coTUIEditIntField(const std::string &n, int pID, int def)
    : coTUIElement(n, pID, TABLET_INT_EDIT_FIELD)
{
    value = def;
    immediate = 0;
    setVal(value);
}

coTUIEditIntField::coTUIEditIntField(coTabletUI *tui, const std::string &n, int pID, int def)
    : coTUIElement(tui, n, pID, TABLET_INT_EDIT_FIELD)
{
    value = def;
    immediate = 0;
    setVal(value);
}

coTUIEditIntField::coTUIEditIntField(QObject *parent, const std::string &n, int pID, int def)
    : coTUIElement(parent, n, pID, TABLET_INT_EDIT_FIELD)
{
    value = def;
    immediate = 0;
    setVal(value);
}

coTUIEditIntField::~coTUIEditIntField()
{
}

void coTUIEditIntField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIEditIntField::parseMessage(TokenBuffer &tb)
{
    tb >> value;
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

std::string coTUIEditIntField::getText() const
{
    return "";
}

void coTUIEditIntField::setMin(int min)
{
    //cerr << "coTUIEditIntField::setMin " << min << endl;
    this->min = min;
    setVal(TABLET_MIN, min);
}

void coTUIEditIntField::setMax(int max)
{
    //cerr << "coTUIEditIntField::setMax " << max << endl;
    this->max = max;
    setVal(TABLET_MAX, max);
}

void coTUIEditIntField::setValue(int val)
{
    if (value != val)
    {
        value = val;
        setVal(value);
    }
}

void coTUIEditIntField::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(TABLET_MIN, min);
    setVal(TABLET_MAX, max);
    setVal(value);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIEditFloatField::coTUIEditFloatField(const std::string &n, int pID, float def)
    : coTUIElement(n, pID, TABLET_FLOAT_EDIT_FIELD)
{
    value = def;
    setVal(value);
    immediate = 0;
}

coTUIEditFloatField::coTUIEditFloatField(coTabletUI *tui, const std::string &n, int pID, float def)
    : coTUIElement(tui, n, pID, TABLET_FLOAT_EDIT_FIELD)
{
    value = def;
    setVal(value);
    immediate = 0;
}

coTUIEditFloatField::coTUIEditFloatField(QObject *parent, const std::string &n, int pID, float def)
    : coTUIElement(parent, n, pID, TABLET_FLOAT_EDIT_FIELD)
{
    value = def;
    setVal(value);
    immediate = 0;
}

coTUIEditFloatField::~coTUIEditFloatField()
{
}

void coTUIEditFloatField::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIEditFloatField::parseMessage(TokenBuffer &tb)
{
    tb >> value;
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUIEditFloatField::setValue(float val)
{
    if (value != val)
    {
        value = val;
        setVal(value);
    }
}

void coTUIEditFloatField::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(value);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUISpinEditfield::coTUISpinEditfield(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_SPIN_EDIT_FIELD)
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

coTUISpinEditfield::coTUISpinEditfield(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_SPIN_EDIT_FIELD)
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

coTUISpinEditfield::~coTUISpinEditfield()
{
}

void coTUISpinEditfield::parseMessage(TokenBuffer &tb)
{
    tb >> actValue;
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUISpinEditfield::setPosition(int newV)
{

    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUISpinEditfield::setStep(int newV)
{
    step = newV;
    setVal(TABLET_STEP, step);
}

void coTUISpinEditfield::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void coTUISpinEditfield::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUISpinEditfield::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
    setVal(actValue);
}

//----------------------------------------------------------
//----------------------------------------------------------
coTUITextSpinEditField::coTUITextSpinEditField(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_TEXT_SPIN_EDIT_FIELD)
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

coTUITextSpinEditField::coTUITextSpinEditField(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_TEXT_SPIN_EDIT_FIELD)
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

coTUITextSpinEditField::~coTUITextSpinEditField()
{
}

void coTUITextSpinEditField::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    text = m;
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUITextSpinEditField::setStep(int newV)
{
    step = newV;
    setVal(TABLET_STEP, step);
}

void coTUITextSpinEditField::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void coTUITextSpinEditField::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUITextSpinEditField::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_STEP, step);
    setVal(text);
}

void coTUITextSpinEditField::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIProgressBar::coTUIProgressBar(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_PROGRESS_BAR)
{
    actValue = 0;
    maxValue = 100;
}

coTUIProgressBar::coTUIProgressBar(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_PROGRESS_BAR)
{
    actValue = 0;
    maxValue = 100;
}

coTUIProgressBar::~coTUIProgressBar()
{
}

void coTUIProgressBar::setValue(int newV)
{
    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUIProgressBar::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUIProgressBar::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(TABLET_MAX, maxValue);
    setVal(actValue);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIFloatSlider::coTUIFloatSlider(const std::string &n, int pID, bool s)
    : coTUIElement(n, pID, TABLET_FLOAT_SLIDER)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    ticks = 10;

    orientation = s;
    setVal(orientation);
}

coTUIFloatSlider::coTUIFloatSlider(coTabletUI *tui, const std::string &n, int pID, bool s)
: coTUIElement(tui, n, pID, TABLET_FLOAT_SLIDER)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    ticks = 10;

    orientation = s;
    setVal(orientation);
}

coTUIFloatSlider::coTUIFloatSlider(QObject *parent, const std::string &n, int pID, bool s)
    : coTUIElement(parent, n, pID, TABLET_FLOAT_SLIDER)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    ticks = 10;

    orientation = s;
    setVal(orientation);
}

coTUIFloatSlider::~coTUIFloatSlider()
{
}

void coTUIFloatSlider::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    tb >> actValue;
    if (i == TABLET_PRESSED)
    {
        emit tabletPressEvent();
        emit tabletEvent();
        if (listener)
        {
            listener->tabletPressEvent(this);
            listener->tabletEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
        emit tabletReleaseEvent();
        emit tabletEvent();
        if (listener)
        {
            listener->tabletReleaseEvent(this);
            listener->tabletEvent(this);
        }
    }
    else
    {
        emit tabletEvent();
        if (listener)
            listener->tabletEvent(this);
    }
}

void coTUIFloatSlider::setValue(float newV)
{
    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUIFloatSlider::setTicks(int newV)
{
    if (ticks != newV)
    {
        ticks = newV;
        setVal(TABLET_NUM_TICKS, ticks);
    }
}

void coTUIFloatSlider::setMin(float minV)
{
    if (minValue != minV)
    {
        minValue = minV;
        setVal(TABLET_MIN, minValue);
    }
}

void coTUIFloatSlider::setMax(float maxV)
{
    if (maxValue != maxV)
    {
        maxValue = maxV;
        setVal(TABLET_MAX, maxValue);
    }
}

void coTUIFloatSlider::setRange(float minV, float maxV)
{
    minValue = minV;
    maxValue = maxV;
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
}

void coTUIFloatSlider::setOrientation(bool o)
{
    orientation = o;
    setVal(orientation);
}

void coTUIFloatSlider::setLogarithmic(bool val)
{
    logarithmic = val;
    setVal(TABLET_SLIDER_SCALE, logarithmic ? TABLET_SLIDER_LOGARITHMIC : TABLET_SLIDER_LINEAR);
}

void coTUIFloatSlider::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_NUM_TICKS, ticks);
    setVal(TABLET_SLIDER_SCALE, logarithmic ? TABLET_SLIDER_LOGARITHMIC : TABLET_SLIDER_LINEAR);
    setVal(actValue);
    setVal(orientation);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUISlider::coTUISlider(const std::string &n, int pID, bool s)
    : coTUIElement(n, pID, TABLET_SLIDER)
    , actValue(0)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    orientation = s;
    setVal(orientation);
}

coTUISlider::coTUISlider(coTabletUI *tui, const std::string &n, int pID, bool s)
    : coTUIElement(tui, n, pID, TABLET_SLIDER)
    , actValue(0)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    orientation = s;
    setVal(orientation);
}

coTUISlider::coTUISlider(QObject *parent, const std::string &n, int pID, bool s)
    : coTUIElement(parent, n, pID, TABLET_SLIDER)
    , actValue(0)
{
    label = "";
    actValue = 0;
    minValue = 0;
    maxValue = 0;
    orientation = s;
    setVal(orientation);
}

coTUISlider::~coTUISlider()
{
}

void coTUISlider::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;
    tb >> actValue;
    if (i == TABLET_PRESSED)
    {
        emit tabletPressEvent();
        emit tabletEvent();
        if (listener)
        {
            listener->tabletPressEvent(this);
            listener->tabletEvent(this);
        }
    }
    else if (i == TABLET_RELEASED)
    {
        emit tabletReleaseEvent();
        emit tabletEvent();
        if (listener)
        {
            listener->tabletReleaseEvent(this);
            listener->tabletEvent(this);
        }
    }
    else
    {
        emit tabletEvent();
        if (listener)
            listener->tabletEvent(this);
    }
}

void coTUISlider::setValue(int newV)
{

    if (actValue != newV)
    {
        actValue = newV;
        setVal(actValue);
    }
}

void coTUISlider::setTicks(int newV)
{
    if (ticks != newV)
    {
        ticks = newV;
        setVal(TABLET_NUM_TICKS, ticks);
    }
}

void coTUISlider::setMin(int minV)
{
    minValue = minV;
    setVal(TABLET_MIN, minValue);
}

void coTUISlider::setMax(int maxV)
{
    maxValue = maxV;
    setVal(TABLET_MAX, maxValue);
}

void coTUISlider::setRange(int minV, int maxV)
{
    minValue = minV;
    maxValue = maxV;
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
}

void coTUISlider::setOrientation(bool o)
{
    orientation = o;
    setVal(orientation);
}

void coTUISlider::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(TABLET_MIN, minValue);
    setVal(TABLET_MAX, maxValue);
    setVal(TABLET_NUM_TICKS, ticks);
    setVal(actValue);
    setVal(orientation);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTUIComboBox::coTUIComboBox(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_COMBOBOX)
{
    label = "";
    text = "";
    selection = -1;
}

coTUIComboBox::coTUIComboBox(coTabletUI *tui, const std::string &n, int pID)
    : coTUIElement(tui, n, pID, TABLET_COMBOBOX)
{
    label = "";
    text = "";
    selection = -1;
}

coTUIComboBox::coTUIComboBox(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_COMBOBOX)
{
    label = "";
    text = "";
    selection = -1;
}

coTUIComboBox::~coTUIComboBox()
{
}

void coTUIComboBox::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    text = m;
    iter = elements.first();
    int i = 0;
    selection = -1;
    while (iter)
    {
        if (*iter == text)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUIComboBox::addEntry(const std::string &t)
{
    elements.append(t);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
}

void coTUIComboBox::delEntry(const std::string &t)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
    iter = elements.first();
    while (iter)
    {
        if (*iter == t)
        {
            iter.remove();
            break;
        }
        iter++;
    }
}

int coTUIComboBox::getNumEntries()
{
    return elements.num();
}

void coTUIComboBox::clear()
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ALL;
    tb << ID;
    tui()->send(tb);
    elements.clean();
}

void coTUIComboBox::setSelectedText(const std::string &t)
{
    text = t;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
    int i = 0;
    iter = elements.first();
    while (iter)
    {
        if (*iter == text)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
}

const std::string &coTUIComboBox::getSelectedText() const
{
    return text;
}

int coTUIComboBox::getSelectedEntry() const
{
    return selection;
}

void coTUIComboBox::setSelectedEntry(int e)
{
    selection = e;
    if (e >= elements.num())
        selection = elements.num() - 1;
    if (selection < 0)
        return;
    std::string selectedEntry = elements.item(selection);
    text = selectedEntry;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
}

void coTUIComboBox::resend(bool create)
{
    coTUIElement::resend(create);
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_REMOVE_ALL;
        tb << ID;
        tui()->send(tb);
        iter = elements.first();
    }
    while (iter)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_ADD_ENTRY;
        tb << ID;
        tb << (*iter).c_str();
        tui()->send(tb);
        iter++;
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

coTUIListBox::coTUIListBox(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_LISTBOX)
{
    text = "";
    selection = -1;
}

coTUIListBox::coTUIListBox(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_LISTBOX)
{
    text = "";
    selection = -1;
}

coTUIListBox::~coTUIListBox()
{
}

void coTUIListBox::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    text = m;
    iter = elements.first();
    int i = 0;
    selection = -1;
    while (iter)
    {
        if (*iter == text)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUIListBox::addEntry(const std::string &t)
{
    elements.append(t);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_ADD_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
}

void coTUIListBox::delEntry(const std::string &t)
{
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_REMOVE_ENTRY;
    tb << ID;
    tb << t.c_str();
    tui()->send(tb);
    iter = elements.first();
    while (iter)
    {
        if (*iter == text)
        {
            iter.remove();
            break;
        }
        iter++;
    }
}

void coTUIListBox::setSelectedText(const std::string &t)
{
    text = t;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
    int i = 0;
    iter = elements.first();
    while (iter)
    {
        if (*iter == text)
        {
            selection = i;
            break;
        }
        iter++;
        i++;
    }
}

const std::string &coTUIListBox::getSelectedText() const
{
    return text;
}

int coTUIListBox::getSelectedEntry() const
{
    return selection;
}

void coTUIListBox::setSelectedEntry(int e)
{
    selection = e;
    if (e >= elements.num())
        selection = elements.num() - 1;
    if (selection < 0)
        return;
    text = elements.item(selection);
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_SELECT_ENTRY;
    tb << ID;
    tb << text.c_str();
    tui()->send(tb);
}

void coTUIListBox::resend(bool create)
{
    coTUIElement::resend(create);

    iter = elements.first();
    while (iter)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_ADD_ENTRY;
        tb << ID;
        tb << (*iter).c_str();
        tui()->send(tb);
        iter++;
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

coTUIMap::coTUIMap(const char *n, int pID)
    : coTUIElement(n, pID, TABLET_MAP)
{
}

coTUIMap::~coTUIMap()
{
    iter = maps.first();
    while (iter)
    {
        delete[] * iter;
        iter++;
    }
}

void coTUIMap::parseMessage(TokenBuffer &tb)
{
    tb >> mapNum;
    tb >> xPos;
    tb >> yPos;
    tb >> height;

    if (listener)
        listener->tabletEvent(this);
}

void coTUIMap::addMap(const char *name, float ox, float oy, float xSize, float ySize, float height)
{
    MapData *md = new MapData(name, ox, oy, xSize, ySize, height);
    maps.append(md);
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

void coTUIMap::resend(bool create)
{
    coTUIElement::resend(create);

    iter = maps.first();
    while (iter)
    {
        TokenBuffer tb;
        tb << TABLET_SET_VALUE;
        tb << TABLET_ADD_MAP;
        tb << ID;
        tb << iter->name;
        tb << iter->ox;
        tb << iter->oy;
        tb << iter->xSize;
        tb << iter->ySize;
        tb << iter->height;
        tui()->send(tb);
        iter++;
    }
}

//----------------------------------------------------------
//----------------------------------------------------------
//##########################################################

coTUIPopUp::coTUIPopUp(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_POPUP)
{
    text = "";
    immediate = false;
}

coTUIPopUp::coTUIPopUp(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_POPUP)
{
    text = "";
    immediate = false;
}

coTUIPopUp::~coTUIPopUp()
{
}

void coTUIPopUp::setImmediate(bool i)
{
    immediate = i;
    setVal(immediate);
}

void coTUIPopUp::parseMessage(TokenBuffer &tb)
{
    char *m;
    tb >> m;
    text = m;
    emit tabletEvent();
    if (listener)
        listener->tabletEvent(this);
}

void coTUIPopUp::setText(const std::string &t)
{
    text = t;
    setVal(text);
}

void coTUIPopUp::resend(bool create)
{
    coTUIElement::resend(create);
    setVal(text);
    setVal(immediate);
}

//----------------------------------------------------------
//----------------------------------------------------------

coTabletUI *coTabletUI::tUI = NULL;

coTabletUI *coTabletUI::instance()
{
    if (tUI == NULL)
        tUI = new coTabletUI();
    return tUI;
}
//----------------------------------------------------------
//----------------------------------------------------------

coTUIElement::coTUIElement(const std::string &n, int pID, int type)
: QObject(0)
, type(type)
, m_tui(coTabletUI::instance())
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
    hidden = false;
    tui()->addElement(this);
    createSimple(type);
    if(tui()->debugTUI())
    {
        coVRMSController::instance()->syncStringStop(name);
        coVRMSController::instance()->syncInt(ID);
    }
}

coTUIElement::coTUIElement(coTabletUI *tabletUI, const std::string &n, int pID, int type)
: QObject(0)
, type(type)
, m_tui(tabletUI)
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
    hidden = false;
    tui()->addElement(this);
    createSimple(type);
    if(tui()->debugTUI())
    {
        coVRMSController::instance()->syncStringStop(name);
        coVRMSController::instance()->syncInt(ID);
    }
}

#if 0
coTUIElement::coTUIElement(QObject *parent, const std::string &n, int pID)
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
    hidden = false;
    if(tui()->debugTUI())
    {
        coVRMSController::instance()->syncStringStop(name);
        coVRMSController::instance()->syncInt(ID);
    }
}
#endif

coTUIElement::coTUIElement(QObject *parent, const std::string &n, int pID, int type)
: QObject(parent)
, type(type)
, m_tui(coTabletUI::instance())
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
    hidden = false;
    tui()->addElement(this);
    if(tui()->debugTUI())
    {
        coVRMSController::instance()->syncStringStop(name);
        coVRMSController::instance()->syncInt(ID);
    }
}

coTUIElement::~coTUIElement()
{
    TokenBuffer tb;
    tb << TABLET_REMOVE;
    tb << ID;
    tui()->send(tb);
    tui()->removeElement(this);
}

void coTUIElement::createSimple(int type)
{
    TokenBuffer tb;
    tb << TABLET_CREATE;
    tb << ID;
    tb << type;
    tb << parentID;
    tb << name.c_str();
    tui()->send(tb);
}

coTabletUI *coTUIElement::tui() const
{
    if (m_tui)
        return m_tui;
    else
        return coTabletUI::instance();
}

void coTUIElement::setLabel(const char *l)
{
    if (l)
        setLabel(std::string(l));
    else
        setLabel(std::string(""));
}

void coTUIElement::setLabel(const std::string &l)
{
    label = l;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_LABEL;
    tb << ID;
    tb << label.c_str();
    tui()->send(tb);
}

int coTUIElement::getID() const
{
    return ID;
}

void coTUIElement::setEventListener(coTUIListener *l)
{
    listener = l;
}

void coTUIElement::parseMessage(TokenBuffer &)
{
}

coTUIListener *coTUIElement::getMenuListener()
{
    return listener;
}

void coTUIElement::setVal(float value)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_FLOAT;
    tb << ID;
    tb << value;
    tui()->send(tb);
}

void coTUIElement::setVal(bool value)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BOOL;
    tb << ID;
    tb << (char)value;
    tui()->send(tb);
}

void coTUIElement::setVal(int value)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_INT;
    tb << ID;
    tb << value;
    tui()->send(tb);
}

void coTUIElement::setVal(const std::string &value)
{
    if (tui()->conn == NULL)
        return;

    //cerr << "coTUIElement::setVal info: " << (value ? value : "*NULL*") << endl;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_STRING;
    tb << ID;
    tb << value.c_str();
    tui()->send(tb);
}

void coTUIElement::setVal(int type, int value)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tui()->send(tb);
}

void coTUIElement::setVal(int type, float value)
{
    if (tui()->conn == NULL)
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tui()->send(tb);
}
void coTUIElement::setVal(int type, int value, const std::string &nodePath)
{
    if (tui()->conn == NULL)
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << type;
    tb << ID;
    tb << value;
    tb << nodePath.c_str();
    tui()->send(tb);
}
void coTUIElement::setVal(int type, const std::string &nodePath, const std::string &simPath, const std::string &simName)
{
    if (tui()->conn == NULL)
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

void coTUIElement::setVal(int type, int value, const std::string &nodePath, const std::string &parentPath)
{
    if (tui()->conn == NULL)
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
void coTUIElement::resend(bool create)
{
    if (create)
        createSimple(type);

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

    if (hidden)
    {
        tb.reset();
        tb << TABLET_SET_VALUE;
        tb << TABLET_SET_HIDDEN;
        tb << ID;
        tb << hidden;
        tui()->send(tb);
    }

    if (!enabled)
    {
        tb.reset();
        tb << TABLET_SET_VALUE;
        tb << TABLET_SET_ENABLED;
        tb << ID;
        tb << enabled;
        tui()->send(tb);
    }

    tb.reset();
    tb << TABLET_SET_VALUE;
    tb << TABLET_LABEL;
    tb << ID;
    tb << label.c_str();
    tui()->send(tb);
}

void coTUIElement::setPos(int x, int y)
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

void coTUIElement::setHidden(bool newState)
{
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

void coTUIElement::setEnabled(bool newState)
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

void coTUIElement::setColor(Qt::GlobalColor color)
{

    this->color = color;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_COLOR;
    tb << ID;
    tb << this->color;
    tui()->send(tb);
}

void coTUIElement::setSize(int x, int y)
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

coTabletUI::coTabletUI()
{
    assert(!tUI);
    tUI = this;

    init();

    std::string line;
    if (getenv("COVER_TABLETPC"))
    {
        std::string env(getenv("COVER_TABLETPC"));
        std::string::size_type p = env.find(':');
        if (p != std::string::npos)
        {
            port = atoi(env.substr(p + 1).c_str());
            env = env.substr(0, p);
        }
        line = env;
        std::cerr << "getting TabletPC configuration from $COVER_TABLETPC: " << line << ":" << port << std::endl;
        serverMode = false;
    }
    else
    {
        port = coCoviseConfig::getInt("port", "COVER.TabletPC.Server", port);
        line = coCoviseConfig::getEntry("COVER.TabletPC.Server");
        serverMode = coCoviseConfig::isOn("COVER.TabletPC.ServerMode", false);
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

    tryConnect();
}

coTabletUI::coTabletUI(const std::string &host, int port)
: port(port)
{
    init();

    serverMode = false;
    serverHost = new Host(host.c_str());

    tryConnect();
}

void coTabletUI::init()
{
    debugTUIState = coCoviseConfig::isOn("COVER.DebugTUI", debugTUIState);
    elements.setNoDelete();

    timeout = coCoviseConfig::getFloat("COVER.TabletPC.Timeout", timeout);
}

void coTabletUI::close()
{
    delete conn;
    conn = NULL;

    delete serverConn;
    serverConn = NULL;

    tryConnect();
}

bool coTabletUI::debugTUI()
{
    return debugTUIState;
}

void coTabletUI::tryConnect()
{
    delete serverConn;
    serverConn = NULL;
}

coTabletUI::~coTabletUI()
{
    if (tUI == this)
        tUI = nullptr;

    delete serverConn;
    delete conn;
    delete serverHost;
    delete localHost;
}

int coTabletUI::getID()
{
    return ID++;
}

void coTabletUI::send(TokenBuffer &tb)
{
    if (conn == NULL)
        return;
    Message m(tb);
    m.type = COVISE_MESSAGE_TABLET_UI;
    conn->send_msg(&m);
}

bool coTabletUI::update()
{
    if (coVRMSController::instance() == NULL)
        return false;

    if (conn)
    {
    }
    else if (coVRMSController::instance()->isMaster() && serverMode)
    {
        if (serverConn == NULL)
        {
            serverConn = new ServerConnection(port, 0, (sender_type)0);
            serverConn->listen();
        }
    }
    else if ((coVRMSController::instance()->isMaster()) && (serverHost != NULL || localHost != NULL))
    {
        lock();
        connectedHost = NULL;
        unlock();
        // try to connect to server every 2 secnods
        if ((cover->frameRealTime() - oldTime) > 2)
        {
            if (serverHost)
            {
                if ((firstConnection && cover->debugLevel(1)) || cover->debugLevel(3))
                    std::cerr << "Trying tablet UI connection to " << serverHost->getName() << ":" << port << "... " << std::flush;
                conn = new ClientConnection(serverHost, port, 0, (sender_type)0, 0, timeout);
                if ((firstConnection && cover->debugLevel(1)) || cover->debugLevel(3))
                    std::cerr << (conn->is_connected()?"success":"failed") << "." << std::endl;
                firstConnection = false;
            }
            if (conn && !conn->is_connected()) // could not open server port
            {
#ifndef _WIN32
                if (errno != ECONNREFUSED)
                {
                    fprintf(stderr, "Could not connect to TabletPC %s; port %d: %s\n",
                            serverHost->getName(), port, strerror(errno));
                    delete serverHost;
                    serverHost = NULL;
                }
#else
                fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", serverHost->getName(), port);
                delete serverHost;
                serverHost = NULL;
#endif
                delete conn;
                conn = NULL;
            }
            else if (conn)
            {
                lock();
                connectedHost = serverHost;
                unlock();
            }

            if (!conn && localHost)
            {
                if ((firstConnection && cover->debugLevel(1)) || cover->debugLevel(3))
                    std::cerr << "Trying tablet UI connection to " << localHost->getName() << ":" << port << "... " << std::flush;
                conn = new ClientConnection(localHost, port, 0, (sender_type)0, 0);
                if ((firstConnection && cover->debugLevel(1)) || cover->debugLevel(3))
                    std::cerr << (conn->is_connected()?"success":"failed") << "." << std::endl;
                firstConnection = false;
                if (!conn->is_connected()) // could not open server port
                {
#ifndef _WIN32
                    if (errno != ECONNREFUSED)
                    {
                        fprintf(stderr, "Could not connect to TabletPC %s; port %d: %s\n",
                                localHost->getName(), port, strerror(errno));
                    }
#else
                    fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", localHost->getName(), port);
#endif
                    delete conn;
                    conn = NULL;
                }
                else
                {
                    lock();
                    connectedHost = localHost;
                    unlock();
                }
            }

            if (conn && conn->is_connected())
            {
                // create Texture and SGBrowser Connections
                Message *msg = new covise::Message();
                conn->recv_msg(msg);
                if (msg->type == COVISE_MESSAGE_SOCKET_CLOSED)
                {
                    delete conn;
                    conn = NULL;
                }
                else
                {
                    TokenBuffer tb(msg);
                    int tPort;
                    tb >> tPort;

                    ClientConnection *cconn = new ClientConnection(connectedHost, tPort, 0, (sender_type)0, 2, 1);
                    if (!cconn->is_connected()) // could not open server port
                    {
#ifndef _WIN32
                        if (errno != ECONNREFUSED)
                        {
                            fprintf(stderr, "Could not connect to TabletPC TexturePort %s; port %d: %s\n",
                                    connectedHost->getName(), tPort, strerror(errno));
                        }
#else
                        fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", connectedHost->getName(), tPort);
#endif
                        delete cconn;
                        cconn = NULL;
                    }
                    textureConn = cconn;

                    conn->recv_msg(msg);
                    TokenBuffer stb(msg);

                    stb >> tPort;

                    cconn = new ClientConnection(connectedHost, tPort, 0, (sender_type)0, 2, 1);
                    if (!cconn->is_connected()) // could not open server port
                    {
#ifndef _WIN32
                        if (errno != ECONNREFUSED)
                        {
                            fprintf(stderr, "Could not connect to TabletPC TexturePort %s; port %d: %s\n",
                                    connectedHost->getName(), tPort, strerror(errno));
                        }
#else
                        fprintf(stderr, "Could not connect to TabletPC %s; port %d\n", connectedHost->getName(), tPort);
#endif
                        delete cconn;
                        cconn = NULL;
                    }

                    sgConn = cconn;
                    // resend all ui Elements to the TabletPC
                    coDLListIter<coTUIElement *> iter;
                    iter = elements.first();
                    while (iter)
                    {
                        iter->resend(true);
                        iter++;
                    }
                }
            }
            else
            {
                Message msg(Message::UI, "WANT_TABLETUI");
                coVRPluginList::instance()->sendVisMessage(&msg);
            }
            oldTime = cover->frameRealTime();
        }
    }
    if (serverConn && serverConn->check_for_input())
    {
        conn = serverConn->spawn_connection();
        if (conn && conn->is_connected())
        {
            Message m;
            conn->recv_msg(&m);
            TokenBuffer tb(&m);
            char *hostName;
            tb >> hostName;
            serverHost = new Host(hostName);
            // resend all ui Elements to the TabletPC
            coDLListIter<coTUIElement *> iter;
            iter = elements.first();
            while (iter)
            {
                iter->resend(true);
                iter++;
            }
        }
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
        if (coVRMSController::instance()->isMaster())
        {
            if (conn)
            {
                if (conn->check_for_input())
                {
                    conn->recv_msg(&m);
                    gotMessage = true;
                }
            }
            coVRMSController::instance()->sendSlaves((char *)&gotMessage, sizeof(bool));
            if (gotMessage)
            {
                coVRMSController::instance()->sendSlaves(&m);
            }
        }
        else
        {
            if (coVRMSController::instance()->readMaster((char *)&gotMessage, sizeof(bool)) < 0)
            {
                cerr << "bcould not read message from Master" << endl;
                exit(0);
            }
            if (gotMessage)
            {
                if (coVRMSController::instance()->readMaster(&m) < 0)
                {
                    cerr << "ccould not read message from Master" << endl;
                    //cerr << "sync_exit13 " << myID << endl;
                    exit(0);
                }
            }
        }
        if (gotMessage)
        {
            changed = true;

            TokenBuffer tb(&m);
            switch (m.type)
            {
            case COVISE_MESSAGE_SOCKET_CLOSED:
            case COVISE_MESSAGE_CLOSE_SOCKET:
            {
                delete conn;
                conn = NULL;
            }
            break;
            case COVISE_MESSAGE_TABLET_UI:
            {

                int ID;
                tb >> ID;
                if (ID >= 0)
                {
                    //coDLListSafeIter<coTUIElement*> iter;

                    coDLListIter<coTUIElement *> iter;
                    iter = elements.first();
                    while (iter)
                    {
                        if (*iter)
                        {

                            if (iter->getID() == ID)
                            {
                                iter->parseMessage(tb);
                                break;
                            }
                        }
                        iter++;
                    }
                }
            }
            break;
            default:
            {
                cerr << "unknown Message type" << endl;
            }
            break;
            }
        }
    } while (gotMessage);

    return changed;
}

void coTabletUI::addElement(coTUIElement *e)
{
    elements.append(e);
    newElements.push_back(e);
}

void coTabletUI::removeElement(coTUIElement *e)
{
    auto it = std::find(newElements.begin(), newElements.end(), e);
    if (it != newElements.end())
        newElements.erase(it);
    coDLListIter<coTUIElement *> iter;
    iter = elements.findElem(e);
    if (iter)
        iter.remove();
}


coTUIGroupBox::coTUIGroupBox(const std::string &n, int pID)
    : coTUIElement(n, pID, TABLET_GROUPBOX)
{

}

coTUIGroupBox::coTUIGroupBox(coTabletUI *tui, const std::string &n, int pID)
    : coTUIElement(tui, n, pID, TABLET_GROUPBOX)
{

}

coTUIGroupBox::coTUIGroupBox(QObject *parent, const std::string &n, int pID)
    : coTUIElement(parent, n, pID, TABLET_GROUPBOX)
{

}

coTUIGroupBox::~coTUIGroupBox()
{

}

void coTUIGroupBox::resend(bool create)
{
    coTUIElement::resend(create);
}

void coTUIGroupBox::parseMessage(TokenBuffer &tb)
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
    else
    {
        cerr << "unknown event " << i << endl;
    }
}
