/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MessageReceiver.h"

#include <stdio.h>
#include <iostream>
#include <string>

#ifndef WIN32
#include <pthread.h>
#endif

#include <util/common.h>
#include <signal.h>
#ifndef _WIN32
#include <sys/socket.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#ifndef __APPLE__
#include <sys/prctl.h>
#endif
#ifdef __linux__
#define sigset signal
#endif
#ifndef _WIN32
#define closesocket close
#endif

#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <netinet/in.h>
#include <arpa/inet.h>
//For closing a socket you need
//closesocket under windows
//but close under linux
#else
typedef unsigned long in_addr_t;
#include <windows.h>
#define ioctl ioctlsocket
#endif

#ifdef _WIN32
void MessageReceiver::receiveLoop(void *messageReceiver)
#else
void *MessageReceiver::receiveLoop(void *messageReceiver)
#endif
{

    MessageReceiver *mr = (MessageReceiver *)messageReceiver;

    while (1)
    {
        mr->_waitForConnection();
        while (1) //while (mr->_receiveData() != -1)
        {
            mr->_receiveData();
            mr->_sendData();
        }
        mr->_disconnect();
    }

#ifndef _WIN32
    return NULL;
#endif
}

void MessageReceiver::_waitForConnection()
{
    _serverSocket = new Socket(_port);

    std::cout << "Waiting for connection on port " << _port << "... ";
    int accepted = _serverSocket->accept(_timeout);
    if (accepted == -1)
    {
        std::cout << "failed!" << std::endl;
    }
    else
    {
        std::cout << "done!" << std::endl;
    }
}

void MessageReceiver::_disconnect()
{
    if (_serverSocket)
    {
        delete _serverSocket;
        _serverSocket = 0;
    }
}

std::vector<std::string> MessageReceiver::popMessageQueue()
{
    std::vector<std::string> queue;

    _mutex.lock();
    queue = _messageQueue;
    _messageQueue.clear();
    _mutex.unlock();

    return queue;
}

int MessageReceiver::send(std::string s)
{
    _mutexSendingQueue.lock();
    _sendingQueue.push_back(s);
    _mutexSendingQueue.unlock();

    return 0;
}

int MessageReceiver::write(const void *buf, unsigned nbyte)
{
    return _serverSocket->write(buf, nbyte);
}

int MessageReceiver::_sendData()
{
    if (_sendingQueue.size() == 0)
    {
        return 0;
    }

    _mutexSendingQueue.lock();
    for (size_t i = 0; i < _sendingQueue.size(); i++)
    {
        _serverSocket->write((const void *)(_sendingQueue[i].c_str()), (unsigned int)_sendingQueue[i].length());
    }
    _sendingQueue.clear();
    _mutexSendingQueue.unlock();

    return 0;
}

int MessageReceiver::_receiveData()
{
    char buffer[1024];
    buffer[1023] = buffer[0] = 0;

    fd_set sready;
    struct timeval nowait;

    FD_ZERO(&sready);
    FD_SET(_serverSocket->get_id(), &sready);
    memset((char *)&nowait, 0, sizeof(nowait));

    if (select(_serverSocket->get_id() + 1, &sready, NULL, NULL, &nowait) == 0)
    {
        // nothing to see here, move along
    }
    else
    {
        // read package

        int numRead = 0;
        numRead = _serverSocket->Read(buffer, 1023);
        if (numRead != -1)
        {
            buffer[numRead] = 0;

            // push buffer on message stack
            _mutex.lock();
            // Ray: We're pushing the binary data as string to the _messageQueue
            _messageQueue.push_back(std::string(buffer));
            _mutex.unlock();

            // send a short ack to indicate we are ready
            /*
          * Ray: We don't want to send an ACK to the application!?
         char ack[1];
         ack[0] = 255;
         _serverSocket->write(ack, 1);
         */
        }
        else
        {
            std::cout << "numRead:-1 errno: " << errno << "... ";
            // error while reading. connection closed?
            return -1;
        }
    }

    return 0;
}

MessageReceiver::MessageReceiver(int port, int timeout)
{
    _port = port;
    _timeout = timeout;

    _mutex.init();
    _mutexSendingQueue.init();

    _mainLoop();
}

MessageReceiver::~MessageReceiver()
{
    // _mutex.destroy();
    // _mutexSendingQueue.destroy();
}

void MessageReceiver::_mainLoop()
{
/**/
#ifdef _WIN32
    // Not used: "uintptr_t thread = "
    _beginthread(MessageReceiver::receiveLoop, 0, this);
#else
    //std::cerr << "Treadmill: MessageReceiver starting pthread..." << std::endl;
    pthread_t thread;
    pthread_create(&thread, NULL, MessageReceiver::receiveLoop, this);
#endif
    /**/
}
