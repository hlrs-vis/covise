/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_H
#define _VRB_CLIENT_H
#include <stdio.h>
#include <future>
#include <mutex>
#include <util/DLinkList.h>
#include <util/coTypes.h>
namespace covise
{

class Host;
class CoviseConfig;
class ClientConnection;
class Message;
class TokenBuffer;

//
//
//

class VRBEXPORT VRBClient
{

public:
    VRBClient(const char *name, const char *collaborativeConfigurationFile = NULL, bool isSlave = false);
    VRBClient(const char *name, const char *host, int pPort, bool isSlave = false);
    ~VRBClient();
    int connectToServer(); // returns -1, if Connection to Server fails
    void connectToCOVISE(int argc, const char **argv);
    int isCOVERRunning();
    int isConnected();
    int poll(Message *m);
    int wait(Message *m);
    int wait(Message *m, int messageType);
    int setUserInfo(const char *userInfo);
    int sendMessage(const Message *m);
    void sendMessage(TokenBuffer &tb, int type);
    int getID();
    void setID(int ID);
    DLinkList<Message *> messageQueue;
    float getSendDelay();

private:
    ClientConnection *sConn; // connection to Server
    char *name;
    int port;
    int ID;
    Host *serverHost;
    bool isSlave; // it true, we are a slave in a multiPC config, so do not actually connect to server
    float sendDelay; // low-pass filtered time for sending one packet of 1000 bytes
    std::mutex connMutex;
    std::future<ClientConnection *> connFuture;
    bool firstVrbConnection = true;
};
}
#endif
