/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef YAC
#include "covise/covise.h"
#endif
#include <memory>
namespace covise
{
class ServerConnection;
class Connection;
class ConnectionList;
class Message;
}

class VRBPClient;
class VRBProxy;

class VRBPClient
{
public:
    VRBPClient(std::unique_ptr<covise::Connection> &&cc, VRBProxy *prox);
    ~VRBPClient();
    void sendMessage(const covise::Message *msg);

    const covise::Connection *toClient;
    const covise::Connection *toVRB;
    VRBProxy *prox;

private:
};

class VRBPClientList : public std::list<VRBPClient *>
{

public:
    VRBPClient *get(const covise::Connection *c);
    void deleteAll();
    void sendMessage(const covise::Message *msg);
};

extern VRBPClientList clients;

class VRBProxy
{
public:
    VRBProxy(int argc, char **argv);
    ~VRBProxy();
    void handleMessages();
    int openServer();
    void closeServer();
    covise::ConnectionList *connections;

private:
    const covise::ServerConnection *sConn;
    covise::Message *msg;
    int port;
};
