/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Office Plugin (connection to Microsoft Office)              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "OfficePlugin.h"
#include <util/unixcompat.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <vrml97/vrml/VrmlNamespace.h>

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>
#include "VrmlNodeOffice.h"
#include <sys/stat.h>
#ifndef WIN32
#include <sys/socket.h>
#endif
#include <cassert>

#include <cover/ui/Manager.h>
#include <cover/ui/Menu.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Label.h>

#include <osg/Matrix>
#include <osg/MatrixTransform>

using covise::TokenBuffer;
using covise::coCoviseConfig;

OfficeConnection::OfficeConnection(ServerConnection *to)
: ui::Owner("Server"+std::to_string((intptr_t)to), OfficePlugin::instance())
{
    if (coVRMSController::instance()->isMaster())
        toOffice = to;
    ServerOc = to;
    
    auto menu = OfficePlugin::instance()->menu;
    myFrame = new ui::Group("Product", this);
    menu->add(myFrame);

    commandLine = new ui::EditField(myFrame, "CommandLine");
    commandLine->setText("Command line");
    commandLine->setCallback([this](const std::string &cmd){
        if(cmd.length()>0)
        {
            TokenBuffer stb;
            stb << cmd;

            Message message(stb);
            message.type = (int)OfficePlugin::MSG_String;
            sendMessage(message);
            commandLine->setValue("");
        }
    });
    auto group = new ui::Group(menu, "LastMessageGroup");
    group->setText("");
    group->setPriority(ui::Element::Low);
    auto label = new ui::Label(group, "LastMessageLabel");
    label->setText("Last message: ");
    lastMessage = new ui::Label(group, "LastMessage");
    lastMessage->setText("");
}

OfficeConnection::~OfficeConnection()
{
    if (toOffice && toOffice->getSocket())
        cover->unwatchFileDescriptor(toOffice->getSocket()->get_id());
    delete toOffice;
}

void OfficeConnection::sendMessage(Message &m)
{
    if(toOffice) // false on slaves
    {
        toOffice->sendMessage(&m);
    }
}

void OfficeConnection::handleMessage(Message *m)
{
    enum OfficePlugin::MessageTypes type = (enum OfficePlugin::MessageTypes)m->type;
    TokenBuffer tb(m);

    switch (type)
    {
    case OfficePlugin::MSG_String:
        {
            const char *line;
            tb >> line;
            lastMessage->setText(line);
            if(strncmp(line,"setViewpoint",12)==0)
            {
                float scale=1.0;
                coCoord coord;
                sscanf(line+13,"scale=%f,position=%f;%f;%f,orientation=%f;%f;%f",&scale,&coord.xyz[0],&coord.xyz[1],&coord.xyz[2],&coord.hpr[0],&coord.hpr[1],&coord.hpr[2]);
                osg::Matrix m;
                coord.makeMat(m);
                cover->setXformMat(m);
                cover->setScale(scale);
            }
            for(std::list<VrmlNodeOffice *>::iterator it = VrmlNodeOffice::allOffice.begin();it != VrmlNodeOffice::allOffice.end();it++)
            {
                if((*it)->getApplicationType() == applicationType)
                {
                    (*it)->setMessage(line);
                }
            }
        }
        break;
    case OfficePlugin::MSG_ApplicationType:
        {
            const char *at=NULL;
            tb >> at;
            if(at)
            {
             applicationType = at;
            }
            const char *pn;
            tb >> pn;
            productName = pn;
            fprintf(stderr,"applicationType: %s  product: %s\n",at,pn);
            myFrame->setText(pn);
        }
        break;
    default:
        std::cerr << "Unknown message [" << m->type << "] " << std::endl;
        break;
    }
}

void
OfficePlugin::handleMessage(OfficeConnection *oc,Message *m)
{
    enum Message::Type type = (enum Message::Type)m->type;
    if(oc)
    {
        switch (type)
        {
        case Message::SOCKET_CLOSED:
        case Message::CLOSE_SOCKET:
            officeConnections.destroy(oc);
            break;
        default:
            oc->handleMessage(m);
        }
    }
}

void OfficePlugin::createMenu()
{
    menu = new ui::Menu("Office", this);
    menu->setVisible(false);
    menu->setVisible(true, ui::View::Tablet);

  /*  updateCameraTUIButton = new coTUIButton("Update Camera", officeTab->getID());
    updateCameraTUIButton->setEventListener(this);
    updateCameraTUIButton->setPos(0, 0);*/
}

void OfficePlugin::destroyMenu()
{
}

OfficePlugin::OfficePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("OfficePlugin", cover->ui)
{
    assert(!plugin);
    fprintf(stderr, "OfficePlugin::OfficePlugin\n");
    plugin = this;
}

bool OfficePlugin::init()
{
    bool connected = false;
    if (coVRMSController::instance()->isMaster())
    {
        int port = coCoviseConfig::getInt("port", "COVER.Plugin.Office.Server", 31315);
        serverConn = new ServerConnection(port, 1234, Message::UNDEFINED);
        if (!serverConn->getSocket())
        {
            std::cout << "tried to open server Port " << port << std::endl;
            std::cout << "Creation of server failed!" << std::endl;
            std::cout << "Port-Binding failed! Port already bound?" << std::endl;
            delete serverConn;
            serverConn = NULL;
        }
        else
        {
            cover->watchFileDescriptor(serverConn->getSocket()->get_id());
        }

        struct linger linger;
        linger.l_onoff = 0;
        linger.l_linger = 0;
        std::cout << "Set socket options..." << std::endl;
        if (serverConn)
        {
            setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

            std::cout << "Set server to listen mode..." << std::endl;
            serverConn->listen();
            if (!serverConn->is_connected()) // could not open server port
            {
                fprintf(stderr, "Could not open server port %d\n", port);
                cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
                delete serverConn;
                serverConn = NULL;
            }
        }

        connected = true;
    }

    connected = coVRMSController::instance()->syncBool(connected);

    if (connected)
    {
        VrmlNamespace::addBuiltIn(VrmlNodeOffice::defineType());
        createMenu();
    }
    return connected;
}

OfficePlugin *OfficePlugin::instance()
{
    return plugin;
}

// this is called if the plugin is removed at runtime
OfficePlugin::~OfficePlugin()
{
    destroyMenu();
    if (serverConn && serverConn->getSocket())
        cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
    delete serverConn;
    serverConn = NULL;
    plugin = nullptr;
}

void OfficePlugin::message(int toWhom, int type, int len, const void *buf)
{
    TokenBuffer tb{ covise::DataHandle{(char*)buf, len, false} };
    if (type == PluginMessageTypes::PBufferDoneSnapshot)
    {
        std::string fileName;
        tb >> fileName;
#ifdef WIN32
        int fileDesc = open(fileName.c_str(), O_RDONLY|O_BINARY);
#else
        int fileDesc = open(fileName.c_str(), O_RDONLY);
#endif
        if (fileDesc >= 0)
        {
            int fileSize = 0;
#ifndef WIN32
            struct stat statbuf;
            fstat(fileDesc, &statbuf);
#else
            struct _stat statbuf;
            _fstat(fileDesc, &statbuf);
#endif
            fileSize = statbuf.st_size;
            char *buf = new char[fileSize];
            read(fileDesc,buf,fileSize);
            TokenBuffer stb;
            stb << fileSize;
            stb.addBinary(buf,fileSize);

            std::string transform;
            osg::Matrix m = cover->getObjectsXform()->getMatrix();
            coCoord coord(m);
            char *tmps = new char[200];
            snprintf(tmps,200,"scale=%f,position=%f;%f;%f,orientation=%f;%f;%f",cover->getScale(),coord.xyz[0],coord.xyz[1],coord.xyz[2],coord.hpr[0],coord.hpr[1],coord.hpr[2]);
            transform = tmps;
            delete[] tmps;
            stb << transform;
            Message message(stb);
            message.type = (int)OfficePlugin::MSG_PNGSnapshot;
            sendMessage(message);
            delete[] buf;
        }
        else
        {
            fprintf(stderr, "Office Plugin:: failed to open %s\n", fileName.c_str());
        }
    }
}

OfficePlugin *OfficePlugin::plugin = NULL;


void OfficePlugin::sendMessage(Message &m)
{
    officeConnections.sendMessage("Word",m);
    officeConnections.sendMessage("PowerPoint",m);
}

void OfficePlugin::preFrame()
{
    std::unique_ptr<ServerConnection> toOffice;
    if (coVRMSController::instance()->isMaster())
    {
        if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
        {
            //   std::cout << "Trying serverConn..." << std::endl;
            auto toOffice = serverConn->spawn_connection();
            if (toOffice && toOffice->is_connected())
            {
                fprintf(stderr, "Connected to Office system\n");
                cover->watchFileDescriptor(toOffice->getSocket()->get_id());
            }
        }
        coVRMSController::instance()->sendSlaves(&toOffice, sizeof(toOffice));
    }
    else
    {
        coVRMSController::instance()->readMaster(&toOffice, sizeof(toOffice));
    }

    if (toOffice)
    {
        officeConnections.push_back(new OfficeConnection(&*toOffice));
    }

    officeConnections.checkAndHandleMessages();
}

void officeList::sendMessage(std::string application, Message &m)
{
    for(iterator it=begin();it!=end();it++)
    {
        if(application == (*it)->applicationType)
        {
            (*it)->sendMessage(m);
        }
    }
}

officeList::officeList()
{
    msg = new Message;
}

officeList::~officeList()
{
    delete msg;
}

void officeList::destroy(OfficeConnection *oc)
{
    remove(oc);
    delete oc;
    deletedConnection=true;
}

void officeList::checkAndHandleMessages()
{
    const ServerConnection *sc=NULL;
    deletedConnection=false;
    if (coVRMSController::instance()->isMaster())
    {
        for(iterator it=begin();(it!=end() && deletedConnection==false);it++)
        {
            while ((*it) && (*it)->toOffice!=NULL &&  (*it)->toOffice->check_for_input())
            {
                (*it)->toOffice->recv_msg(msg);
                if (msg)
                {
                    auto oc=(*it);
                    sc = oc->ServerOc;
                    coVRMSController::instance()->sendSlaves(&sc, sizeof(sc));
                    coVRMSController::instance()->sendSlaves(msg);

                    cover->sendMessage(OfficePlugin::instance(), coVRPluginSupport::TO_SAME_OTHERS,PluginMessageTypes::HLRS_Office_Message+msg->type-OfficePlugin::MSG_String,msg->data.length(), msg->data.data());
                    OfficePlugin::instance()->handleMessage(oc,msg);
                    if(deletedConnection)
                        break;
                }
                else
                {
                    sc = NULL;
                    std::cerr << "could not read message" << std::endl;
                    break;
                }
            }
        }
        sc = NULL;
        coVRMSController::instance()->sendSlaves(&sc, sizeof(sc));
    }
    else
    {
        do
        {
            coVRMSController::instance()->readMaster(&sc, sizeof(sc));
            if (sc != NULL)
            {
                // find the local oc for this master connection
                // 
                OfficeConnection *localOc=NULL;
                for(iterator it=begin();it!=end();it++)
                {
                    if((*it)->ServerOc == sc)
                    {
                        localOc = (*it);
                    }
                }
                Message msg;
                coVRMSController::instance()->readMaster(&msg);
                cover->sendMessage(OfficePlugin::instance(), coVRPluginSupport::TO_SAME_OTHERS,PluginMessageTypes::HLRS_Office_Message+msg.type-OfficePlugin::MSG_String,msg.data.length(), msg.data.data());
                if(localOc)
                {
                    OfficePlugin::instance()->handleMessage(localOc,&msg);
                }
                else
                {
                    std::cerr << "Office: did not find connection on slave" << std::endl;
                }
            }
        } while (sc != NULL);
    }
}


COVERPLUGIN(OfficePlugin)
