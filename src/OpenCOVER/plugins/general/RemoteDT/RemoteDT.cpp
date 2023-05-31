/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2007 HLRS  **
 **                                                                        **
 ** Description: Remote Desktop plugin (VNC, RDP)                          **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** based on Vic.cpp by U. Woessner                                        **
 **                                                                        **
 ** History:                                                               **
 ** Oct 12, 2007    v1                                                     **
 **                                                                        **
\****************************************************************************/

#include "RemoteDT.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>

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
#include <net/tokenbuffer.h>
#ifndef WIN32
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include "Interface.h"
#include "RemoteDTActor.h"
#include "vnc/window.h"

#include <PluginUtil/PluginMessageTypes.h>

using covise::TokenBuffer;

RemoteDT *RemoteDT::plugin = NULL;

void RemoteDT::key(int type, int keysym, int mod)
{
    if (desktop)
    {
        if (type == 6)
            desktop->getActor()->keyDown(keysym, mod);
        else if (type == 7)
            desktop->getActor()->keyUp(keysym, mod);
        else
        {
            fprintf(stderr, "RemoteDT::keyEvent(): Unknown key event type %d!\n", type);
        }
    }
    else
    {
        fprintf(stderr, "RemoteDT::keyEvent(): desktop is null!\n");
    }
}

//
//   this needs to be changed, so we can connect to arbitrary servers!
//
RemoteDT::RemoteDT()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin = this;

    desktop = 0;
    eventHandler = 0;

    pinboardEntry = 0;
    remoteDTMenu = 0;
    desktopEntry = 0;

    tuiRemoteDTTab = 0;
    tuiPortEdit = 0;
    tuiIPaddrLabel = 0;
    tuiIPaddr = 0;
    tuiPasswdLabel = 0;
    tuiPasswd = 0;
    tuiToggleVNC = 0;
    tuiToggleRDP = 0;
    tuiConnectButton = 0;
    tuiDisconnectButton = 0;

    protocol = PROTO_VNC;
}

bool RemoteDT::init()
{
    initUI();
    return true;
}

// this is called if the plugin is removed at runtime
RemoteDT::~RemoteDT()
{
    fprintf(stderr, "RemoteDT::~RemoteDT\n");
    if (desktop)
    {
        delete desktop;
    }
}

void RemoteDT::initUI()
{
    // opencover ui
    pinboardEntry = new coSubMenuItem("RemoteDT");
    cover->getMenu()->add(pinboardEntry);
    remoteDTMenu = new coRowMenu("Servers");
    pinboardEntry->setMenu(remoteDTMenu);

    // tablet ui
    tuiRemoteDTTab = new coTUITab("RemoteDesktop", coVRTui::instance()->mainFolder->getID());
    tuiRemoteDTTab->setPos(0, 0);

    tuiIPaddrLabel = new coTUILabel("IP address", tuiRemoteDTTab->getID());
    tuiIPaddrLabel->setPos(0, 0);

    tuiIPaddr = new coTUIEditField("IP address", tuiRemoteDTTab->getID());
    tuiIPaddr->setEventListener(this);
    tuiIPaddr->setText(""); // 141.58.8.13
    tuiIPaddr->setPos(1, 0);
    // tuiIPaddr->setIPAddressMode(true);

    (new coTUILabel("Port", tuiRemoteDTTab->getID()))->setPos(0, 1);

    tuiPortEdit = new coTUIEditIntField("Port", tuiRemoteDTTab->getID());
    tuiPortEdit->setValue(5901);
    tuiPortEdit->setEventListener(this);
    tuiPortEdit->setImmediate(true);
    tuiPortEdit->setMin(0);
    tuiPortEdit->setMax(65536);
    tuiPortEdit->setPos(1, 1);

    tuiPasswdLabel = new coTUILabel("Password", tuiRemoteDTTab->getID());
    tuiPasswdLabel->setPos(0, 2);

    tuiPasswd = new coTUIEditField("Password", tuiRemoteDTTab->getID());
    tuiPasswd->setEventListener(this);
    tuiPasswd->setText("");
    tuiPasswd->setPos(1, 2);
    tuiPasswd->setPasswordMode(true);

    tuiToggleVNC = new coTUIToggleButton("VNC", tuiRemoteDTTab->getID());
    tuiToggleVNC->setEventListener(this);
    tuiToggleVNC->setPos(0, 3);
    tuiToggleVNC->setState(true);

    tuiToggleRDP = new coTUIToggleButton("RDP", tuiRemoteDTTab->getID());
    tuiToggleRDP->setEventListener(this);
    tuiToggleRDP->setPos(1, 3);

    tuiConnectButton = new coTUIButton("Connect", tuiRemoteDTTab->getID());
    tuiConnectButton->setEventListener(this);
    tuiConnectButton->setPos(0, 4);

    tuiDisconnectButton = new coTUIButton("Disconnect", tuiRemoteDTTab->getID());
    tuiDisconnectButton->setEventListener(this);
    tuiDisconnectButton->setPos(1, 4);

    tuiCommandlineLabel = new coTUILabel("Command Line", tuiRemoteDTTab->getID());
    tuiCommandlineLabel->setPos(0, 6);

    tuiCommandline = new coTUIEditField("Command Line", tuiRemoteDTTab->getID());
    tuiCommandline->setEventListener(this);
    tuiCommandline->setText("");
    tuiCommandline->setPos(1, 6);

    tuiCommandlineSendAndReturn = new coTUIButton("Send Text and RETURN", tuiRemoteDTTab->getID());
    tuiCommandlineSendAndReturn->setEventListener(this);
    tuiCommandlineSendAndReturn->setPos(2, 6);

    tuiCommandlineSend = new coTUIButton("Send Text", tuiRemoteDTTab->getID());
    tuiCommandlineSend->setEventListener(this);
    tuiCommandlineSend->setPos(3, 6);

    tuiCommandlineReturn = new coTUIButton("RETURN", tuiRemoteDTTab->getID());
    tuiCommandlineReturn->setEventListener(this);
    tuiCommandlineReturn->setPos(4, 6);

    tuiCommandlineCTRL_C = new coTUIButton("CTRL+C", tuiRemoteDTTab->getID());
    tuiCommandlineCTRL_C->setEventListener(this);
    tuiCommandlineCTRL_C->setPos(0, 7);

    tuiCommandlineCTRL_D = new coTUIButton("CTRL+D", tuiRemoteDTTab->getID());
    tuiCommandlineCTRL_D->setEventListener(this);
    tuiCommandlineCTRL_D->setPos(1, 7);

    tuiCommandlineCTRL_Z = new coTUIButton("CTRL+Z", tuiRemoteDTTab->getID());
    tuiCommandlineCTRL_Z->setEventListener(this);
    tuiCommandlineCTRL_Z->setPos(2, 7);

    tuiCommandlineESCAPE = new coTUIButton("ESCAPE", tuiRemoteDTTab->getID());
    tuiCommandlineESCAPE->setEventListener(this);
    tuiCommandlineESCAPE->setPos(3, 7);
} // initUI

void RemoteDT::tabletEvent(coTUIElement * /*tUIItem*/)
{
} // tabletEvent

void RemoteDT::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiConnectButton)
    {
        if (!desktop)
        {
            connectToServer(
                tuiIPaddr->getText().c_str(),
                (unsigned)tuiPortEdit->getValue(),
                tuiPasswd->getText().c_str());
        }
    }
    else if (tUIItem == tuiDisconnectButton)
    {
        if (desktop)
        {
            terminateConnection();
        }
    }
    else if (tUIItem == tuiToggleVNC)
    {
        protocol = PROTO_VNC;
        tuiToggleRDP->setState(false);
    }
    else if (tUIItem == tuiToggleRDP)
    {
        protocol = PROTO_RDP;
        tuiToggleVNC->setState(false);
    }
    else if (tUIItem == tuiCommandlineSend)
    {
        if (desktop)
        {
            desktop->sendText(tuiCommandline->getText().c_str());
        }
    }
    else if (tUIItem == tuiCommandlineSendAndReturn)
    {
        if (desktop)
        {
            desktop->sendText(tuiCommandline->getText().c_str());
            desktop->sendReturn();
        }
    }
    else if (tUIItem == tuiCommandlineReturn)
    {
        if (desktop)
        {
            desktop->sendReturn();
        }
    }
    else if (tUIItem == tuiCommandlineCTRL_C)
    {
        if (desktop)
        {
            desktop->sendCTRL_C();
        }
    }
    else if (tUIItem == tuiCommandlineCTRL_D)
    {
        if (desktop)
        {
            desktop->sendCTRL_D();
        }
    }
    else if (tUIItem == tuiCommandlineCTRL_Z)
    {
        if (desktop)
        {
            desktop->sendCTRL_Z();
        }
    }
    else if (tUIItem == tuiCommandlineESCAPE)
    {
        if (desktop)
        {
            desktop->sendESCAPE();
        }
    }
} // tabletPressEvent

void RemoteDT::tabletReleaseEvent(coTUIElement * /*tUIItem*/)
{
} // tabletReleaseEvent

void RemoteDT::connectToServer(const char *server, unsigned port, const char *password)
{
    if (strlen(server) <= 980)
    {
        switch (protocol)
        {
        case PROTO_VNC:
            desktop = new VNCWindow();
            if (desktop)
            {
                if (desktop->connectToServer(server, port, password))
                {
                    char buf[1000];
                    sprintf(buf, "%s:%d", server, port);
                    desktopEntry = new coCheckboxMenuItem(buf, false);
                    remoteDTMenu->add(desktopEntry);
                    desktopEntry->setMenuListener(this);
                    remoteDTMenu->show();
                }
                else
                {
                    fprintf(stderr, "RemoteDT::connectToServer() err: Could not connect to %s:%d!\n", server, port);
                    delete desktop;
                    desktop = 0;
                }
            }
            else
            {
                fprintf(stderr, "RemoteDT::connectToServer() err: Could not create a VNCWindow!\n");
            }
            break;

        case PROTO_RDP:
            fprintf(stderr, "RemoteDT::connectToServer() err: RDP not yet supported!\n");
            break;

        default:
            fprintf(stderr, "RemoteDT::connectToServer() err: Unknown protocol '0x%x'\n", protocol);
            break;
        } // switch protocol
    }
    else
    {
        fprintf(stderr, "RemoteDT::connectToServer() err: Servername is too long (%d), max. 980!\n", (int)strlen(server));
    }
} // connectToServer()

void RemoteDT::terminateConnection()
{
    if (desktop)
    {
        delete desktop;
        desktop = 0;
    }

    if (desktopEntry)
    {
        remoteDTMenu->remove(desktopEntry);
        delete desktopEntry;
        desktopEntry = 0;

        remoteDTMenu->hide();
    }
} // terminateConnection

void RemoteDT::show()
{
    setVisible(true);
}

void RemoteDT::hide()
{
    setVisible(false);
}

void RemoteDT::setVisible(bool on)
{
    if (this->desktopEntry)
    {
        // Generate the corresponding event via the menu
        desktopEntry->setState(on, true);
    }
}

void RemoteDT::preFrame()
{
    if (desktop)
        desktop->update();
}

void RemoteDT::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == desktopEntry)
    {
        if (desktopEntry && desktop)
        {
            if (desktopEntry->getState())
            {
                desktop->show();
            }
            else
            {
                desktop->hide();
            }
        }
    }
}

void RemoteDT::message(int toWhom, int type, int len, const void *buf)
{
    switch (type)
    {
    case PluginMessageTypes::RemoteDTConnectToHost:
    {
        TokenBuffer tb((const char *)buf, len);
        const char *c_ip, *c_passwd;
        unsigned int port;

        tb >> c_ip;
        tb >> port;
        tb >> c_passwd;

        fprintf(stderr, "RemoteDT::message() info: got message ConnectToHost %s, %d\n", c_ip, port);

        connectToServer(c_ip, port, c_passwd);
        break;
    }
    case PluginMessageTypes::RemoteDTDisconnect:
    {
        fprintf(stderr, "RemoteDT::message() info: got message Disconnect\n");
        terminateConnection();
        break;
    }
    case PluginMessageTypes::RemoteDTShowDesktop:
    {
        fprintf(stderr, "RemoteDT::message() info: got message ShowDesktop\n");
        show();
        break;
    }
    case PluginMessageTypes::RemoteDTHideDesktop:
    {
        fprintf(stderr, "RemoteDT::message() info: got message HideDesktop\n");
        hide();
        break;
    }
    default:
        fprintf(stderr, "RemoteDT::message() err: unknown message\n");
    }
}

COVERPLUGIN(RemoteDT)
