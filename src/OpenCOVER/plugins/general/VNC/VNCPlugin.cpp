/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2009 HLRS  **
 **                                                                        **
 ** Description: VNC Plugin for OpenCOVER                                  **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** Created on: Nov 17, 2008                                               **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

#include "VNCPlugin.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <net/tokenbuffer.h>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coRowMenu.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coVRMSController.h>
#include <sys/types.h>
#ifndef WIN32
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

#include <PluginUtil/PluginMessageTypes.h>

#include "VNCWindow.h"
#include "VNCConnection.h"
#include "VNCWindowActor.h"

using covise::TokenBuffer;

VNCPlugin *VNCPlugin::plugin = NULL;

void VNCPlugin::key(int type, int keysym, int mod)
{
#if 0
   std::cerr << "VNCPlugin::keyEvent(): Called with type:" << type
         << ", keysym: " << keysym << ", mod: " << mod << std::endl;
#endif
    if (focusedActor)
    {
        if (type == 32)
            focusedActor->keyDown(keysym, mod);
        else if (type == 64)
            focusedActor->keyUp(keysym, mod);
    }
}

//
//   this needs to be changed, so we can connect to arbitrary servers!
//
VNCPlugin::VNCPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin = this;

    pinboardEntry = 0;
    VNCPluginMenu = 0;
    desktopEntry = 0;

    tuiVNCPluginTab = 0;
    tuiPortEdit = 0;
    tuiIPaddrLabel = 0;
    tuiIPaddr = 0;
    tuiPasswdLabel = 0;
    tuiPasswd = 0;
    tuiConnectButton = 0;
    tuiDisconnectButton = 0;

    focusedActor = 0;
}

bool VNCPlugin::init()
{
    initUI();
    return true;
}

// this is called if the plugin is removed at runtime
VNCPlugin::~VNCPlugin()
{
    std::list<VNCConnection *>::iterator iter = connections.begin();
    for (; iter != connections.end(); ++iter)
    {
        delete *iter;
    }

    std::cerr << "VNCPlugin::~VNCPlugin" << std::endl;
}

void VNCPlugin::initUI()
{
    // opencover ui
    pinboardEntry = new coSubMenuItem("VNCPlugin");
    cover->getMenu()->add(pinboardEntry);
    VNCPluginMenu = new coRowMenu("Servers");
    pinboardEntry->setMenu(VNCPluginMenu);

    // tablet ui
    tuiVNCPluginTab = new coTUITab("VNC",
                                   coVRTui::instance()->mainFolder->getID());
    tuiVNCPluginTab->setPos(0, 0);

    tuiNewConnectionFrame = new coTUIFrame("New Connections", tuiVNCPluginTab->getID());
    tuiNewConnectionFrame->setPos(0, 0);

    tuiIPaddrLabel = new coTUILabel("Host", tuiNewConnectionFrame->getID());
    tuiIPaddrLabel->setPos(0, 0);

    tuiIPaddr = new coTUIEditField("Host", tuiNewConnectionFrame->getID());
    tuiIPaddr->setEventListener(this);
    tuiIPaddr->setText(""); // 141.58.8.13
    tuiIPaddr->setPos(1, 0);

    (new coTUILabel("Port", tuiNewConnectionFrame->getID()))->setPos(0, 1);

    tuiPortEdit = new coTUIEditIntField("Port", tuiNewConnectionFrame->getID());
    tuiPortEdit->setValue(5901);
    tuiPortEdit->setEventListener(this);
    tuiPortEdit->setImmediate(true);
    tuiPortEdit->setMin(0);
    tuiPortEdit->setMax(65535);
    tuiPortEdit->setPos(1, 1);

    tuiPasswdLabel = new coTUILabel("Password", tuiNewConnectionFrame->getID());
    tuiPasswdLabel->setPos(0, 2);

    tuiPasswd = new coTUIEditField("Password", tuiNewConnectionFrame->getID());
    tuiPasswd->setEventListener(this);
    tuiPasswd->setPos(1, 2);
    tuiPasswd->setText("");
    tuiPasswd->setPasswordMode(true);

    tuiConnectButton = new coTUIButton("Connect", tuiNewConnectionFrame->getID());
    tuiConnectButton->setEventListener(this);
    tuiConnectButton->setPos(0, 3);

    tuiOpenConnectionsFrame = new coTUIFrame("Open connections", tuiVNCPluginTab->getID());
    tuiOpenConnectionsFrame->setPos(0, 1);

    (new coTUILabel("Open Connections", tuiOpenConnectionsFrame->getID()))->setPos(0, 0);

    tuiConnectionList = new coTUIListBox("Open connections", tuiOpenConnectionsFrame->getID());
    tuiConnectionList->setEventListener(this);
    tuiConnectionList->setPos(0, 1);

    tuiDisconnectButton
        = new coTUIButton("Disconnect", tuiOpenConnectionsFrame->getID());
    tuiDisconnectButton->setEventListener(this);
    tuiDisconnectButton->setPos(0, 2);

    tuiConnectionOperationFrame = new coTUIFrame("Connection operations", tuiVNCPluginTab->getID());
    tuiConnectionOperationFrame->setPos(0, 2);

    tuiCommandlineLabel = new coTUILabel("Command Line",
                                         tuiConnectionOperationFrame->getID());
    tuiCommandlineLabel->setPos(0, 0);

    tuiCommandline
        = new coTUIEditField("Command Line", tuiConnectionOperationFrame->getID());
    tuiCommandline->setEventListener(this);
    tuiCommandline->setText("");
    tuiCommandline->setPos(1, 0);

    tuiCommandlineSendAndReturn = new coTUIButton("Send Text and RETURN",
                                                  tuiConnectionOperationFrame->getID());
    tuiCommandlineSendAndReturn->setEventListener(this);
    tuiCommandlineSendAndReturn->setPos(0, 1);

    tuiCommandlineSend = new coTUIButton("Send Text", tuiConnectionOperationFrame->getID());
    tuiCommandlineSend->setEventListener(this);
    tuiCommandlineSend->setPos(1, 1);

    tuiCommandlineReturn = new coTUIButton("RETURN", tuiConnectionOperationFrame->getID());
    tuiCommandlineReturn->setEventListener(this);
    tuiCommandlineReturn->setPos(2, 1);

    tuiCommandlineCTRL_C = new coTUIButton("CTRL+C", tuiConnectionOperationFrame->getID());
    tuiCommandlineCTRL_C->setEventListener(this);
    tuiCommandlineCTRL_C->setPos(0, 2);

    tuiCommandlineCTRL_D = new coTUIButton("CTRL+D", tuiConnectionOperationFrame->getID());
    tuiCommandlineCTRL_D->setEventListener(this);
    tuiCommandlineCTRL_D->setPos(1, 2);

    tuiCommandlineCTRL_Z = new coTUIButton("CTRL+Z", tuiConnectionOperationFrame->getID());
    tuiCommandlineCTRL_Z->setEventListener(this);
    tuiCommandlineCTRL_Z->setPos(2, 2);

    tuiCommandlineESCAPE = new coTUIButton("ESCAPE", tuiConnectionOperationFrame->getID());
    tuiCommandlineESCAPE->setEventListener(this);
    tuiCommandlineESCAPE->setPos(3, 2);
} // initUI

void VNCPlugin::tabletEvent(coTUIElement * /*tUIItem*/)
{
} // tabletEvent

void VNCPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiConnectButton)
    {
        connectToServer(tuiIPaddr->getText().c_str(), (unsigned)tuiPortEdit->getValue(),
                        tuiPasswd->getText().c_str());
    }
    else if (tUIItem == tuiDisconnectButton)
    {
        terminateSelectedConnection();
    }
    else if (tUIItem == tuiCommandlineSend)
    {
        std::cerr << "tuiCommandlineSend" << std::endl;
        if (focusedActor)
        {
            focusedActor->getWindow()->sendText(tuiCommandline->getText().c_str());
        }
    }
    else if (tUIItem == tuiCommandlineSendAndReturn)
    {
        std::cerr << "tuiCommandlineSendAndReturn" << std::endl;
        if (focusedActor)
        {
            focusedActor->getWindow()->sendText(tuiCommandline->getText().c_str());
            focusedActor->getWindow()->sendReturn();
        }
    }
    else if (tUIItem == tuiCommandlineReturn)
    {
        std::cerr << "tuiCommandlineReturn" << std::endl;
        if (focusedActor)
        {
            focusedActor->getWindow()->sendReturn();
        }
    }
    else if (tUIItem == tuiCommandlineCTRL_C)
    {
        std::cerr << "tuiCommandlineCTRL_C" << std::endl;
        if (focusedActor)
        {
            focusedActor->getWindow()->sendCTRL_C();
        }
    }
    else if (tUIItem == tuiCommandlineCTRL_D)
    {
        std::cerr << "tuiCommandlineCTRL_D" << std::endl;
        if (focusedActor)
        {
            focusedActor->getWindow()->sendCTRL_D();
        }
    }
    else if (tUIItem == tuiCommandlineCTRL_Z)
    {
        std::cerr << "tuiCommandlineCTRL_Z" << std::endl;
        if (focusedActor)
        {
            focusedActor->getWindow()->sendCTRL_Z();
        }
    }
    else if (tUIItem == tuiCommandlineESCAPE)
    {
        std::cerr << "tuiCommandlineESCAPE" << std::endl;
        if (focusedActor)
        {
            focusedActor->getWindow()->sendESCAPE();
        }
    }
} // tabletPressEvent

void VNCPlugin::tabletReleaseEvent(coTUIElement * /*tUIItem*/)
{
    std::cerr << "VNCPlugin::tabletReleaseEvent()" << std::endl;
} // tabletReleaseEvent

void VNCPlugin::connectToServer(const char *server, unsigned port,
                                const char *password)
{
    std::cerr << "VNCPlugin::connectToServer(): Server: " << server
              << ", port: " << port << std::endl;

    VNCConnection *con = new VNCConnection(server, port, password);
    if (con->isConnected())
    {
        connections.push_back(con);

        std::string input;
        std::ostringstream os(input);
        os << server << ":" << port;

        tuiConnectionList->addEntry(os.str().c_str());
        desktopEntry = new coCheckboxMenuItem(os.str().c_str(), false);
        VNCPluginMenu->add(desktopEntry);
        desktopEntry->setMenuListener(this);
        VNCPluginMenu->show();
    }
    else
    {
        delete con;
        con = 0;

        std::cerr
            << "VNCPlugin::connectToServer() err: Could not connect to "
            << server << ":" << port << std::endl;
    }
} // connectToServer()

void VNCPlugin::terminateSelectedConnection()
{
    int sel = tuiConnectionList->getSelectedEntry();
    if (sel >= 0)
    {
        std::string serverId(tuiConnectionList->getSelectedText());
        std::cerr << "VNCPlugin::terminateSelectedConnection() info: Terminating "
                  << serverId << std::endl;
    }
}

void VNCPlugin::terminateConnection(const char *hostname, unsigned int port)
{
    (void)hostname;
    (void)port;

    std::cerr
        << "VNCPlugin::terminateConnection() err: Disconnect from "
        << hostname << ":" << port << std::endl;
} // terminateConnection

void VNCPlugin::show(const char *hostname, unsigned int port)
{
    setVisible(true, hostname, port);
}

void VNCPlugin::hide(const char *hostname, unsigned int port)
{
    setVisible(false, hostname, port);
}

void VNCPlugin::setVisible(bool on, const char *hostname, unsigned int port)
{
    (void)on;
    (void)hostname;
    (void)port;

    std::cerr << "VNCPlugin::setVisible() err: Set visibility of "
              << hostname << ":" << port << " to " << on << std::endl;
}

void VNCPlugin::preFrame()
{
    // for every VNC connection, update it
    // notify window(s)
    for (std::list<VNCConnection *>::iterator iter = connections.begin(); iter != connections.end(); ++iter)
    {
        (*iter)->update();
    }
}

void VNCPlugin::menuEvent(coMenuItem *menuItem)
{
    (void)menuItem;
}

void VNCPlugin::message(int toWhom, int type, int len, const void *buf)
{
    switch (type)
    {
    case PluginMessageTypes::RemoteDTConnectToHost:
    {
        TokenBuffer tb((const char *)buf, len);
        const char *c_hostname, *c_passwd;
        unsigned int port;

        tb >> c_hostname;
        tb >> port;
        tb >> c_passwd;

        std::cerr << "VNCPlugin::message() info: got message ConnectToHost "
                  << c_hostname << ":" << port << std::endl;

        connectToServer(c_hostname, port, c_passwd);
        break;
    }

    case PluginMessageTypes::RemoteDTDisconnect:
    {
        TokenBuffer tb((const char *)buf, len);
        const char *c_hostname;
        unsigned int port;

        tb >> c_hostname;
        tb >> port;

        std::cerr << "VNCPlugin::message() info: got message Disconnect"
                  << c_hostname << ":" << port << std::endl;

        terminateConnection(c_hostname, port);
        break;
    }

    case PluginMessageTypes::RemoteDTShowDesktop:
    {
        TokenBuffer tb((const char *)buf, len);
        const char *c_hostname;
        unsigned int port;

        tb >> c_hostname;
        tb >> port;

        std::cerr << "VNCPlugin::message() info: got message Show"
                  << c_hostname << ":" << port << std::endl;
        show(c_hostname, port);
        break;
    }

    case PluginMessageTypes::RemoteDTHideDesktop:
    {
        TokenBuffer tb((const char *)buf, len);
        const char *c_hostname;
        unsigned int port;

        tb >> c_hostname;
        tb >> port;

        std::cerr << "VNCPlugin::message() info: got message Hide"
                  << c_hostname << ":" << port << std::endl;
        hide(c_hostname, port);
        break;
    }

    default:
        std::cerr << "VNCPlugin::message() err: unknown message" << std::endl;
    }
}

void VNCPlugin::setNewWindowActor(VNCWindowActor *actor)
{
    focusedActor = actor;
}

COVERPLUGIN(VNCPlugin)
