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

using namespace covise;

Proxy::Proxy(char *messageData, CRBConnection *crbC)
{
    crbConn = crbC;
    moduleConn = NULL;
    ctrlConn = NULL;
    char *args[1000];
    char *newMessage = new char[strlen(messageData) + 100];
    newMessage[0] = '\0';
    char tmpBuf[1000];
    int i = 1, n = 0;
    int controllerPort;
    char *controllerHost;
    int id;
    int offset = 0;
    //cerr << " old Message:" << messageData << endl;
    args[n] = messageData;
    n++;
    while (messageData[i])
    {
        if (messageData[i] == ' ')
        {
            messageData[i] = '\0';
            args[n] = messageData + i + 1;
            //fprintf(stderr,"args %d:%s\n",n,args[n]);
            n++;
        }
        i++;
    }
    args[n] = NULL;
    if (messageData[0] != '\001')
    {
        offset = 2;
    }
    else
    {
        offset = 3;
    }
    int retval;
    retval = sscanf(args[offset], "%d", &controllerPort);
    if (retval != 1)
    {
        std::cerr << "Proxy::Proxy: sscanf failed" << std::endl;
        return;
    }
    controllerHost = args[1 + offset];
    retval = sscanf(args[2 + offset], "%d", &id);
    if (retval != 1)
    {
        std::cerr << "Proxy::Proxy: sscanf failed" << std::endl;
        return;
    }
    int serverPort;
    moduleConn = new ServerConnection(&serverPort, id, APPLICATIONMODULE);
    if (!moduleConn->is_connected())
        return;
    moduleConn->listen();
    for (i = 0; i < n; i++)
    {
        if (i == offset)
        {
            sprintf(tmpBuf, "%d", serverPort);
            strcat(newMessage, tmpBuf);
        }
        else if (i == 1 + offset)
        {
            strcat(newMessage, crbConn->myHost->getAddress());
        }
        else
        {
            strcat(newMessage, args[i]);
        }
        strcat(newMessage, " ");
    }
    //cerr << " new Message:" << newMessage << endl;
    Message msg{ COVISE_MESSAGE_CRB_EXEC, DataHandle(newMessage, strlen(newMessage) + 1) };

    crbConn->toCrb->send_msg(&msg);

    if (moduleConn->acceptOne(60) < 0)
    {
        delete moduleConn;
        moduleConn = NULL;
        cerr << "* timelimit in accept for module exceeded!!" << endl;
        return;
    }
    Host *h = new Host(controllerHost);
    ctrlConn = new ClientConnection(h, controllerPort, id, APPLICATIONMODULE);
    delete h;
    crbConn->listOfConnections->add(moduleConn);
    crbConn->listOfConnections->add(ctrlConn);
}

Proxy::~Proxy()
{
    crbConn->listOfConnections->remove(moduleConn);
    crbConn->listOfConnections->remove(ctrlConn);
    delete moduleConn;
    delete ctrlConn;
}
