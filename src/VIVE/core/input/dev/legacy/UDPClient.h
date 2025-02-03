/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UDPCLIENT_H
#define UDPCLIENT_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#ifdef _WIN32
#include <winsock.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <unistd.h>
#endif

const int BUF_SIZ = 4096;

enum SendCommandEnum
{
    createDebugOutput
};
enum RequestCommandEnum
{
    MarkerPos_filtered,
    MarkerPos_unfiltered,
    Target_filtered,
    Target_unfiltered,
    All
};
class UDPClient
{
public:
    bool setup(std::string _remote, std::string _port);

    /* sends message over UDP-protocol to remote server */
    bool sendData(SendCommandEnum command, bool includePacketNumber);
    /* request data over UDP-protocol from remote server */
    bool requestData(RequestCommandEnum command, bool includePacketNumber);
    /* receive data from remote server over UDP-protocol */
    bool receiveData(bool &areOld, long timeout_us = 10000);
    void close();

    std::string getData() const
    {
        return m_sRecvData;
    }

private:
    /* sends message over UDP-protocol to remote server */
    bool sendMessage(const std::string &message, bool includePacketNumber);
#ifdef _WIN32
    WSADATA wsaData; /* Structure for WinSock
 setup communication */
#endif

#ifdef _WIN32
    SOCKET m_iSocket;
#else
    int m_iSocket;
#endif
    struct sockaddr_in m_sRemote;

    static unsigned int m_iCurrentPacketNumber;
    unsigned int m_iWaitingForPacket;
    std::string m_sRecvData; // contains received data
};

#endif
