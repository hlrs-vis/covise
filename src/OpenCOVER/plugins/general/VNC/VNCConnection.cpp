/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2009 HLRS  **
 **                                                                        **
 ** Description: VNC Connection class                                      **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** Created on: Nov 26, 2008                                               **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

#include "VNCConnection.h"

#include "VNCWindow.h"
#include "vnc/vncclient.h"
#include "vnc/WindowBuffer.h"

#include <cover/coVRMSController.h>

#include <iostream>
#include <sstream>

VNCConnection::VNCConnection(const char *_server, unsigned short _port, const char *_password)
    : serverName(_server)
    , port(_port)
    , vncClient(0)
    , vncWindow(0)
    , vncWindowBuffer(0)
    , connectSuccess(false)
{
    connectToServer(_server, _port, _password);
}

VNCConnection::~VNCConnection()
{
    std::cerr << "VNCConnection::~VNCConnection() info: called" << std::endl;
    disconnect();

    delete vncClient;
    delete vncWindow;
    delete vncWindowBuffer;
}

bool VNCConnection::connectToServer(const char *server, int port,
                                    const char *password)
{
    bool check = false;
    connectSuccess = false;

    if (opencover::coVRMSController::instance()->isMaster())
    {
        vncClient = new VNCClient(server, port, password);
        vncWindowBuffer = new VNCWindowBuffer(800, 600, 0);
        vncClient->setWindowBuffer(vncWindowBuffer);

        if (!vncClient->VNCInit())
        {
            delete vncClient;
            vncClient = 0;
            delete vncWindowBuffer;
            vncWindowBuffer = 0;

            static const char *reason = "unknown reason";
            std::cerr
                << "VNCConnection::connectToServer() err: Could not connect to "
                << server << ":" << port << " because '" << reason << "'"
                << std::endl;

            opencover::coVRMSController::instance()->sendSlaves((void *)&check, sizeof(check));
        }
        else
        {

            std::cerr
                << "VNCConnection::connectToServer() info: Connection successful, sending fbUpdate request!"
                << std::endl;

            vncClient->sendFramebufferUpdateRequest(0, 0, vncClient->getFBWidth(),
                                                    vncClient->getFBHeight(), false);
            check = true;
            opencover::coVRMSController::instance()->sendSlaves((void *)&check, sizeof(check));

            std::string serverId;
            std::ostringstream os(serverId);
            os << server << ":" << port;

            vncWindow = new VNCWindow(os.str(), vncClient, vncWindowBuffer);
            connectSuccess = true;
            return true;
        }
    }
    else
    {
        // slaves need to check whether master has created a connection
        // then create a vncwindow that synchronizes with the master
        // in the init method
        opencover::coVRMSController::instance()->readMaster((void *)&check, sizeof(check));
        if (check)
        {
            std::cerr
                << "VNCConnection::connectToServer() info: Connection successfull, synchronizing with master"
                << std::endl;
            vncWindow = new VNCWindow;
            connectSuccess = true;
            return true;
        }
        else
        {
            std::cerr
                << "VNCConnection::connectToServer() info: Master failed connecting."
                << std::endl;
        }
    }

    return false;
}

void VNCConnection::disconnect()
{
    if (vncClient)
    {
        vncClient->VNCClose();
    }
}

void VNCConnection::update()
{
    bool updatesAvailable = false;
    if (opencover::coVRMSController::instance()->isMaster())
    {
        updatesAvailable = pollServer();
        opencover::coVRMSController::instance()->sendSlaves((char *)&updatesAvailable, sizeof(bool));
    }
    else
    {
        opencover::coVRMSController::instance()->readMaster((char *)&updatesAvailable, sizeof(bool));
    }

    if (updatesAvailable)
    {
        vncWindow->update();
    }
}

bool VNCConnection::pollServer()
{
    return vncClient ? vncClient->pollServer() : false;
} // VNCWindow::pollServer()
