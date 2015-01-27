/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// MyTuioServer.h

#if USE_TUIO

#ifndef MY_TUIO_SERVER_H
#define MY_TUIO_SERVER_H

#include <TuioServer.h>
#include <TuioTime.h>
#include <TuioCursor.h>

#include <map>

class MyTuioServer
{
public:
    MyTuioServer();
    ~MyTuioServer();

    enum TouchPointState
    {
        TPS_DOWN = 1,
        TPS_UP = 2,
        TPS_MOVE = 4,
        TPS_STATIONARY = 8,
    };

    bool connectToHost(const char *host, int port, int size);

    void disconnectFromHost();

    void beginFrame();

    void endFrame();

    void updateCursorList(int id, float x, float y, TouchPointState state);

    bool isCursorListEmpty();

private:
    TUIO::TuioCursor *getTuioCursorById(int id);

private:
    TUIO::TuioServer *tuioServer;

    typedef std::map<int, TUIO::TuioCursor *> TuioCursorMap;

    // Qt touch points <--> TUIO touch points
    TuioCursorMap cursorMap;
};
#endif
#endif
