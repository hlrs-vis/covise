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
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/coVRSelectionManager.h>
#include "cover/coVRTui.h"
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <vrml97/vrml/VrmlNamespace.h>


#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Array>
#include <osg/CullFace>
#include <osg/MatrixTransform>

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>
#include "VrmlNodeOffice.h"
#include <sys/stat.h>

using covise::TokenBuffer;
using covise::coCoviseConfig;

OfficeConnection::OfficeConnection(ServerConnection *to)
{
    toOffice = to;
    
    productLabel = new coTUILabel("product",OfficePlugin::instance()->officeTab->getID());
    productLabel->setPos(OfficePlugin::instance()->officeConnections.size(),0);
    myFrame = new coTUIFrame("connectionFrame",OfficePlugin::instance()->officeTab->getID());
    myFrame->setPos(OfficePlugin::instance()->officeConnections.size(),1);
    commandLine = new coTUIEditField("commandline",myFrame->getID());
    commandLine->setPos(0,0);
    commandLine->setEventListener(this);
    lastMessage = new coTUILabel("none",myFrame->getID());
    lastMessage->setPos(0,1);
}
OfficeConnection::~OfficeConnection()
{
    delete lastMessage;
    delete commandLine;
    delete myFrame;
    delete productLabel;
}

void OfficeConnection::tabletEvent(coTUIElement *tUIItem)
{
    if(tUIItem == commandLine)
    {
        std::string cmd = commandLine->getText();
        if(cmd.length()>0)
        {
            TokenBuffer stb;
            stb << cmd;

            Message message(stb);
            message.type = (int)OfficePlugin::MSG_String;
            sendMessage(message);
            commandLine->setText("");
        }
    }
}
void OfficeConnection::tabletPressEvent(coTUIElement *)
{
    
}
void OfficeConnection::sendMessage(Message &m)
{
    if(toOffice) // false on slaves
    {
        toOffice->send_msg(&m);
    }
}
void
OfficeConnection::handleMessage(Message *m)
{
    enum OfficePlugin::MessageTypes type = (enum OfficePlugin::MessageTypes)m->type;
    TokenBuffer tb(m);

    switch (type)
    {
    case OfficePlugin::MSG_String:
        {
            char *line;
            tb >> line;
            lastMessage->setLabel(line);
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
            char *at;
            tb >> at;
            applicationType = at;
            char *pn;
            tb >> pn;
            productName = pn;
            fprintf(stderr,"applicationType: %s  product: %s\n",at,pn);
            productLabel->setLabel(pn);
        }
        break;
    default:
        cerr << "Unknown message [" << m->type << "] " << endl;
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
    OfficeButton = new coSubMenuItem("Office");
//    OfficeButton->setMenu(viewpointMenu);
    
    cover->getMenu()->add(OfficeButton);

    officeTab = new coTUITab("Office", coVRTui::instance()->mainFolder->getID());
    officeTab->setPos(0, 0);

  /*  updateCameraTUIButton = new coTUIButton("Update Camera", officeTab->getID());
    updateCameraTUIButton->setEventListener(this);
    updateCameraTUIButton->setPos(0, 0);*/
}

void OfficePlugin::destroyMenu()
{
    delete OfficeButton;
    delete officeTab;
}

OfficePlugin::OfficePlugin()
{
    fprintf(stderr, "OfficePlugin::OfficePlugin\n");
    plugin = this;
    int port = coCoviseConfig::getInt("port", "COVER.Plugin.Office.Server", 31315);
    serverConn = new ServerConnection(port, 1234, Message::UNDEFINED);
    if (!serverConn->getSocket())
    {
        cout << "tried to open server Port " << port << endl;
        cout << "Creation of server failed!" << endl;
        cout << "Port-Binding failed! Port already bound?" << endl;
        delete serverConn;
        serverConn = NULL;
    }

    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    cout << "Set socket options..." << endl;
    if (serverConn)
    {
        setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

        cout << "Set server to listen mode..." << endl;
        serverConn->listen();
        if (!serverConn->is_connected()) // could not open server port
        {
            fprintf(stderr, "Could not open server port %d\n", port);
            delete serverConn;
            serverConn = NULL;
        }
    }
    msg = new Message;
    
    VrmlNamespace::addBuiltIn(VrmlNodeOffice::defineType());
}

bool OfficePlugin::init()
{
    createMenu();
    return true;
}
// this is called if the plugin is removed at runtime
OfficePlugin::~OfficePlugin()
{
    destroyMenu();
    delete serverConn;
    serverConn = NULL;
}

void OfficePlugin::menuEvent(coMenuItem *aButton)
{
 //   if (aButton == updateCameraButton)
    {
    }
}
void OfficePlugin::tabletPressEvent(coTUIElement *tUIItem)
{
   // if (tUIItem == updateCameraTUIButton)
    {
       
    }
}

void OfficePlugin::tabletEvent(coTUIElement *tUIItem)
{
   // if (tUIItem == addCameraTUIButton)
    {
    }
}


void OfficePlugin::message(int type, int len, const void *buf)
{
    TokenBuffer tb((const char *)buf,len);
    if (type == PluginMessageTypes::PBufferDoneSnapshot)
    {
            std::string fileName;
            tb >> fileName;
            int fileDesc = open(fileName.c_str(), O_RDONLY|O_BINARY);
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
    }
void
OfficePlugin::preFrame()
{
    covise::ServerConnection *toOffice=NULL;
    char gotMsg = '\0';
    if (coVRMSController::instance()->isMaster())
    {
        if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
        {
            //   std::cout << "Trying serverConn..." << std::endl;
            toOffice = serverConn->spawn_connection();
            if (toOffice && toOffice->is_connected())
            {
                fprintf(stderr, "Connected to Office system\n");
            }

        }
        coVRMSController::instance()->sendSlaves(&toOffice, sizeof(toOffice));
    }
    else
    {
        coVRMSController::instance()->readMaster(&toOffice, sizeof(toOffice));
    }
    if(toOffice!=NULL)
    {
        officeConnections.push_back(new OfficeConnection(toOffice));
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
    OfficeConnection *oc=NULL;
    deletedConnection=false;
    if (coVRMSController::instance()->isMaster())
    {
        for(iterator it=begin();(it!=end() && deletedConnection==false);it++)
        {
            while ((*it) && (*it)->toOffice->check_for_input())
            {
                (*it)->toOffice->recv_msg(msg);
                if (msg)
                {
                    oc=(*it);
                    coVRMSController::instance()->sendSlaves(&oc, sizeof(oc));
                    coVRMSController::instance()->sendSlaves(msg);

                    cover->sendMessage(OfficePlugin::instance(), coVRPluginSupport::TO_SAME_OTHERS,PluginMessageTypes::HLRS_Office_Message+msg->type-OfficePlugin::MSG_String,msg->length, msg->data);
                    OfficePlugin::instance()->handleMessage(oc,msg);
                    if(deletedConnection)
                        break;
                }
                else
                {
                    oc = NULL;
                    cerr << "could not read message" << endl;
                    break;
                }
            }
        }
        oc = NULL;
        coVRMSController::instance()->sendSlaves(&oc, sizeof(oc));
    }
    else
    {
        do
        {
            coVRMSController::instance()->readMaster(&oc, sizeof(oc));
            if (oc != NULL)
            {
                Message msg;
                coVRMSController::instance()->readMaster(&msg);
                OfficePlugin::instance()->handleMessage(oc,&msg);
            }
        } while (oc != NULL);
    }
}


COVERPLUGIN(OfficePlugin)
