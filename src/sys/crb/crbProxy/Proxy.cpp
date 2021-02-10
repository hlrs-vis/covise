/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <direct.h>
#include <stdlib.h>
#else
#include <sys/wait.h>
#include <signal.h>
#endif
#include "Proxy.h"
#include <covise/Covise_Util.h>
#ifdef HASVRB
#include <vrb/lib/VRBClient.h>
#endif

#include "CRBConnection.h"
#include <comsg/CRB_EXEC.h>
using namespace covise;

Proxy::Proxy(const CRB_EXEC& messageData, CRBConnection *crbC)
{
    crbConn = crbC;
    moduleConn = NULL;
    ctrlConn = NULL;
    auto args = getCmdArgs(messageData);

    int serverPort;
    auto newModuleConn = std::unique_ptr<ServerConnection>(new ServerConnection(&serverPort, messageData.moduleCount, APPLICATIONMODULE));
    if (!moduleConn->is_connected())
        return;
    newModuleConn->listen();
    CRB_EXEC newMessage{messageData.flag,
                        messageData.name,
                        serverPort,
                        crbConn->myHost->getAddress(),
                        messageData.moduleCount,
                        messageData.moduleId,
                        messageData.moduleIp,
                        messageData.moduleHostName,
                        messageData.category,
                        messageData.vrbClientIdOfController,
                        messageData.vrbCredentials,
                        messageData.params};

    //cerr << " new Message:" << newMessage << endl;
    sendCoviseMessage(newMessage, *crbConn->toCrb);

    if (newModuleConn->acceptOne(60) < 0)
    {
        cerr << "* timelimit in accept for module exceeded!!" << endl;
        return;
    }
    Host h {messageData.controllerIp};
    ctrlConn = new ClientConnection(&h, messageData.controllerPort, messageData.moduleCount, APPLICATIONMODULE);
    moduleConn = dynamic_cast<const ServerConnection*>(crbConn->listOfConnections->add(std::move(newModuleConn)));
    ctrlConn = dynamic_cast<const ClientConnection*>(crbConn->listOfConnections->add(std::unique_ptr<ClientConnection>(new ClientConnection(&h, messageData.controllerPort, messageData.moduleCount, APPLICATIONMODULE))));
}

Proxy::~Proxy()
{
    crbConn->listOfConnections->remove(moduleConn);
    crbConn->listOfConnections->remove(ctrlConn);
}
