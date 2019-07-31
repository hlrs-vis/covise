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
#define MAXHOSTNAMELEN 255
#else
#include <sys/wait.h>
#include <signal.h>
#endif

#include "CRBConnection.h"
#include <covise/Covise_Util.h>
#ifdef HASVRB
#include <vrb/lib/VRBClient.h>
#endif

#include <config/CoviseConfig.h>

using namespace covise;

CRBConnection::CRBConnection(int p, char *h, int ID)
{
    host = new char[strlen(h) + 1];
    strcpy(host, h);
    port = p;
    id = ID;
    listOfConnections = new ConnectionList;

    myHost = new Host();
}

CRBConnection::~CRBConnection()
{
    delete myHost;
    delete host;
}

void CRBConnection::contactController()
{
    Host *h = new Host(host);
    toController = new ClientConnection(h, port, id, DATAMANAGER);
    listOfConnections->add(toController);
    delete h;
}

void CRBConnection::forwardMessage(Message *msg, Connection *conn)
{
    if (conn == toCrb)
    {
        toController->send_msg(msg);
        return;
    }
    else if (conn == toController)
    {
        toCrb->send_msg(msg);
        return;
    }
    ProxyIter iter = modules.first();
    while (iter)
    {
        if (iter->moduleConn == conn) // message coming from module
        {
            iter->ctrlConn->send_msg(msg);
            break;
        }
        else if (iter->ctrlConn == conn) // message to module
        {
            iter->moduleConn->send_msg(msg);
            break;
        }
        iter++;
    }
}

void CRBConnection::processMessages()
{
    Connection *conn;
    conn = listOfConnections->check_for_input(0);
    if (conn)
    {
        Message *msg = new Message;
        conn->recv_msg(msg);
        switch (msg->type)
        {
        case COVISE_MESSAGE_CLOSE_SOCKET:
        case COVISE_MESSAGE_EMPTY:
        case COVISE_MESSAGE_SOCKET_CLOSED:
        {
            forwardMessage(msg, conn);
            ProxyIter iter = modules.first();
            while (iter)
            {
                // message coming from module
                if ((iter->moduleConn == conn) || (iter->ctrlConn == conn))
                {
                    iter.remove();
                    break;
                }
                iter++;
            }
            if ((conn == toCrb) || (conn == toController))
            {
                listOfConnections->remove(toController);
                listOfConnections->remove(toCrb);
                delete toCrb;
                delete toController;
            }
        }
        //break;
        break;
        case COVISE_MESSAGE_CRB_EXEC:
            modules.append(new Proxy(msg->data.accessData(), this));
            break;
        default:
            forwardMessage(msg, conn);
            break;
        }
        delete msg;
    }
}

int CRBConnection::execCRB(char *instance)
{
    (void)instance;

    int serverPort;

    toCrb = new ServerConnection(&serverPort, id, CONTROLLER);
    if (!toCrb->is_connected())
        return 0;
    toCrb->listen();
    std::string command = coCoviseConfig::getEntry("command", "System.CRB.Proxy");
    if (!command.empty())
    {
        cerr << "CRB.ProxyCommand not found in covise.config file" << endl;
        command = "crb";
    }
    char co[2000];
    sprintf(co, "%s %d %s %d&", command.c_str(), serverPort, myHost->getAddress(), id);
    cerr << "starting: " << co << endl;
    int retval;
    retval = system(co);
    if (retval == -1)
    {
        std::cerr << "CRBConnection::execCRB: system failed" << std::endl;
        return 0;
    }
    /*
   sprintf(chport,"%d", serverPort);
   sprintf(chid,"%d", id);
   #ifdef _WIN32
   spawnlp(P_NOWAIT,"crb", "crb", chport, myHost->getAddress(), chid,  0L);
   #else
   if(fork() == 0)
   {
      execlp("crb", "crb", chport, myHost->getAddress(), chid,  0L);
   }
   else
   #endif
   */
    {
        if (toCrb->acceptOne(60) < 0)
        {
            delete toCrb;
            cerr << "* timelimit in accept for crb exceeded!!" << endl;
            return -1;
        }
        listOfConnections->add(toCrb);
    }
    contactController();
    return 1;
}
