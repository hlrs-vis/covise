/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EVENT_RECEIVER_H
#define _EVENT_RECEIVER_H

// winsock2.h must be included before windows.h
#ifdef _WIN32
#include <winsock2.h>
#endif

namespace opencover
{

class EventReceiver
{
public:
    EventReceiver(int port, int window);
    virtual ~EventReceiver();

private:
    void _mainLoop();
    void _readData();
#ifdef _WIN32
    static void _receiveLoop(void *);
#else
    static void *_receiveLoop(void *);
#endif

    void _createSocket();
    void _disconnectSocket();

    int _port;
    int _windowID;
#ifdef _WIN32
    SOCKET _socket;
#else
    int _socket;
#endif
};
}
#endif
