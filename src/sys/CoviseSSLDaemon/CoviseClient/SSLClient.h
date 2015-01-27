/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SSLCLIENT_H
#define SSLCLIENT_H

#include <string>
#include <net/covise_connect.h>

class SSLClient
{
public:
    SSLClient(void);
    SSLClient(std::string server, int port);
    ~SSLClient(void);

    void run(std::string command);
    static int sslPasswdCallback(char *buf, int size, int rwflag, void *userData);

private:
    covise::SSLClientConnection *mConn;
    bool bIsRunning;
    bool bCanRun;
};

#endif
