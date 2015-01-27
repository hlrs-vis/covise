/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MESSAGE_RECEIVER_H
#define _MESSAGE_RECEIVER_H

// winsock2.h must be included before windows.h
#ifdef _WIN32
#include <winsock2.h>
#endif

#include "Mutex.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>

#include <net/covise_socket.h>
#include <net/covise_host.h>

using namespace covise;

class MessageReceiver
{
public:
    MessageReceiver(int port = 5555, int timeout = 3600);
    virtual ~MessageReceiver();

    std::vector<std::string> popMessageQueue();
    int send(std::string s);
    int write(const void *buf, unsigned nbyte);

private:
    void _mainLoop();
    int _receiveData();
    int _sendData();
    void _waitForConnection();
    void _disconnect();

    std::vector<std::string> _messageQueue;
    std::vector<std::string> _sendingQueue;

#ifdef _WIN32
    static void receiveLoop(void *messageReceiver);
#else
    static void *receiveLoop(void *messageReceiver);
#endif

    Mutex _mutex;
    Mutex _mutexSendingQueue;

    int _port;
    Socket *_serverSocket;
    int _timeout;
};

#endif
