/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EventReceiver.h"

#include "VRWindow.h"
#include "coVRConfig.h"

#include <osgViewer/GraphicsWindow>

#ifndef _WIN32
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <unistd.h>
#else
#include <process.h>
#endif

namespace opencover
{

#ifdef _WIN32
void EventReceiver::_receiveLoop(void *receiver)
#else
void *EventReceiver::_receiveLoop(void *receiver)
#endif
{

    EventReceiver *er = (EventReceiver *)receiver;

    while (1)
    {
        er->_createSocket();
        while (1)
        {
            er->_readData();
        }
        er->_disconnectSocket();
    }

#ifndef _WIN32
    return NULL;
#endif
}

EventReceiver::EventReceiver(int port, int window)
    : _port(port)
    , _windowID(window)
{
    _socket = -1;
    _mainLoop();
}

EventReceiver::~EventReceiver()
{
}

void EventReceiver::_createSocket()
{
#ifdef _WIN32
    WSADATA wsaData;

    //create socket
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != NO_ERROR)
    {
        WSACleanup();
        return;
    }
#endif

    _socket = socket(AF_INET, SOCK_DGRAM, 0);
#ifdef _WIN32
    if (_socket == INVALID_SOCKET)
#else
    if (_socket < 0)
#endif
    {
        fprintf(stderr, "invalid socket\n");
    }

    // FILL SOCKET ADRESS STRUCTURE
    sockaddr_in any_adr;

    memset((char *)&any_adr, 0, sizeof(any_adr));
    any_adr.sin_family = AF_INET;
    any_adr.sin_addr.s_addr = INADDR_ANY;
    any_adr.sin_port = htons(7878);

// BIND TO A LOCAL PROTOCOL PORT
#ifdef _WIN32
    if (bind(_socket, (sockaddr *)&any_adr, sizeof(any_adr)) == SOCKET_ERROR)
#else
    if (bind(_socket, (sockaddr *)&any_adr, sizeof(any_adr)) < 0)
#endif
    {
        fprintf(stderr, "invalid socket 2\n");
    }
}

void EventReceiver::_disconnectSocket()
{
    //closeSocket(_socket);
}

void EventReceiver::_mainLoop()
{
#ifdef _WIN32
    // Not used: "uintptr_t thread = "
    _beginthread(EventReceiver::_receiveLoop, 0, this);
#else
    pthread_t thread;
    pthread_create(&thread, NULL, EventReceiver::_receiveLoop, this);
#endif
}

void EventReceiver::_readData()
{
#ifdef _WIN32
    if (_socket == INVALID_SOCKET)
#else
    if (_socket < 0)
#endif
        return;

    sockaddr_in remote_adr;
#ifdef WIN32
    int rlen;
#else
    socklen_t rlen;
#endif

    // read into a buffer first, copy only vaild parts afterwards
    const int BUFSIZE = 2048;
    char rawdata[BUFSIZE];

// check wether we already received package
#ifdef WIN32
    u_long bytes = 0;
#else //WIN32
    size_t bytes = 0;
#endif //WIN32

    // if no data: print message after 5 sec.
    if (bytes <= 0)
    {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(_socket, &readfds);
        struct timeval timeout = { 1, 0 };
        if (0 == select(
                     _socket + 1, // param nfds: specifies the range of descriptors to be tested;
                     // descriptors in [0;nfds-1] are examined
                     &readfds, NULL, NULL, &timeout))
        {
            //fprintf(stderr, "EventReceiver: no data received\n");
        }
    }

    // receive a package
    rlen = sizeof(remote_adr);
    int numbytes = recvfrom(
        _socket, // specifies the socket file descriptor
        rawdata, // container for received data
        BUFSIZE - 1, // one char reserved for trailing '\0' (see below)
        0, // flags
        (struct sockaddr *)&remote_adr, // actual sending socket address
        &rlen); // size of remote_adr
#ifdef WIN32
    WSAGetLastError();
#endif //WIN32
    if (numbytes < 0)
    {
        fprintf(stderr, "EventReceiver: error reading\n");
    } // fi

    // terminate string
    rawdata[numbytes] = '\0';

    //fprintf(stderr, "read %s\n", rawdata);

    // split data
    char msg[50];
    int param1, param2, param3;
    int numRead = sscanf(rawdata, "%s = [%d, %d, %d]", msg, &param1, &param2, &param3);

    // fprintf(stderr, "EventReceiver::read %s %d [%d, %d, %d]\n", msg, numRead, param1, param2, param3);

    // get event queue
    if (!coVRConfig::instance()->windows[_windowID].window)
        return;
    osgGA::EventQueue *queue = coVRConfig::instance()->windows[_windowID].window->getEventQueue();

    if (numRead == 4)
    {
        if (strcmp(msg, "mousePressEvent") == 0)
        {
            queue->mouseButtonPress(param1, param2, param3);
        }
        else if (strcmp(msg, "mouseReleaseEvent") == 0)
        {
            queue->mouseButtonRelease(param1, param2, param3);
        }
        else if (strcmp(msg, "mouseDoubleClickEvent") == 0)
        {
            queue->mouseDoubleButtonPress(param1, param2, param3);
        }
    }
    else if (numRead == 3)
    {
        if (strcmp(msg, "resizeEvent") == 0)
        {
            VRWindow::instance()->setOrigHSize(_windowID, param1);
            VRWindow::instance()->setOrigVSize(_windowID, param2);
            coVRConfig::instance()->windows[_windowID].window->resized(0, 0, param1, param2);
        }
        else if (strcmp(msg, "mouseMoveEvent") == 0)
        {
            queue->mouseMotion(param1, param2);
        }
        else if (strcmp(msg, "keyPressEvent") == 0)
        {
            osgGA::GUIEventAdapter *event = queue->createEvent();
            event->setEventType(osgGA::GUIEventAdapter::KEYDOWN);
            event->setTime(queue->getTime());
            event->setKey(param1);
            event->setModKeyMask(param2);
            queue->addEvent(event);
        }
        if (strcmp(msg, "keyReleaseEvent") == 0)
        {
            osgGA::GUIEventAdapter *event = queue->createEvent();
            event->setEventType(osgGA::GUIEventAdapter::KEYUP);
            event->setTime(queue->getTime());
            event->setKey(param1);
            event->setModKeyMask(param2);
            queue->addEvent(event);
        }
    }
    else if (numRead == 2)
    {
        if (strcmp(msg, "ping") == 0)
        {
            int localSocket = -1;
            localSocket = socket(AF_INET, SOCK_DGRAM, 0);
            unsigned short port = static_cast<unsigned short>(param1);
            sockaddr_in ping_address = remote_adr;
            ping_address.sin_addr.s_addr = remote_adr.sin_addr.s_addr;
            ping_address.sin_port = htons(port);

            static const char *pong = "pong";
            sendto(localSocket, pong, strlen(pong), 0, (struct sockaddr *)&ping_address, sizeof(ping_address));
#if WIN32
            closesocket(localSocket);
#else
            close(localSocket);
#endif
        }
        else if (strcmp(msg, "wheelEvent") == 0)
        { // mouse wheel moved, arg is rotation in +/-degrees, one step == 15 degrees
            if (param1 > 0)
            {
                queue->mouseScroll(osgGA::GUIEventAdapter::SCROLL_UP);
            }
            else
            {
                queue->mouseScroll(osgGA::GUIEventAdapter::SCROLL_DOWN);
            }
        }
    }
    else if (numRead == 1)
    {
        if (strcmp(msg, "dropEvent") == 0)
        {
            osgGA::GUIEventAdapter *event = queue->createEvent();
            event->setEventType(osgGA::GUIEventAdapter::USER);
            event->setModKeyMask(1);
            event->setTime(queue->getTime());
            queue->addEvent(event);
        }
    }
}
}
