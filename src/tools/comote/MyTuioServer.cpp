/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// MyTuioServer.cpp

#if USE_TUIO

#include "MyTuioServer.h"
#include "Debug.h"

MyTuioServer::MyTuioServer()
    : tuioServer(0)
{
}

MyTuioServer::~MyTuioServer()
{
    ASSERT(tuioServer == 0);
}

bool MyTuioServer::connectToHost(const char *host, int port, int /*size*/)
{
    ASSERT(tuioServer == 0); // Currently connected; call Disconnect() first

    try
    {
        tuioServer = new TUIO::TuioServer(host, port);
    }
    catch (std::exception & /*e*/)
    {
        Log() << "TuioServer: connection failed!";

        tuioServer = 0;

        return false;
    }

    bool result = tuioServer->isConnected();

    Log() << "TuioServer: " << (result ? "connected." : "could not connect to host!");

    return result;
}

void MyTuioServer::disconnectFromHost()
{
    //  ASSERT( tuioServer != 0 );

    Log() << "TuioServer: disconnecting...";

    delete tuioServer;
    tuioServer = 0;

    Log() << "TuioServer: disconnected";
}

void MyTuioServer::beginFrame()
{
    if (!tuioServer)
        return;

    tuioServer->initFrame(TUIO::TuioTime::getSessionTime());
}

void MyTuioServer::endFrame()
{
    if (!tuioServer)
        return;

    tuioServer->stopUntouchedMovingCursors();
    tuioServer->commitFrame();
}

void MyTuioServer::updateCursorList(int id, float x, float y, TouchPointState state)
{
    if (!tuioServer)
        return;

    switch (state)
    {
    case TPS_DOWN:
    {
        //
        // Touch point recently pressed
        // Add to list
        //

        TUIO::TuioCursor *cur = getTuioCursorById(id);

        // A touch point with the given id should not exist yet
        ASSERT(cur == 0);

        // Add a new cursor to the server's cursor list
        cur = tuioServer->addTuioCursor(x, y);

        // Add the cursor to the map
        cursorMap[id] = cur;
    }
    break;

    case TPS_UP:
    {
        //
        // Touch point recently release
        // Remove from list
        //

        TUIO::TuioCursor *cur = getTuioCursorById(id);

        // A touch point with the given id should exist
        ASSERT(cur != 0);

        // Remove from the server's cursor list
        tuioServer->removeTuioCursor(cur);

        // Remove from the map
        cursorMap.erase(cursorMap.find(id));
    }
    break;

    case TPS_MOVE:
    {
        //
        // Touch point moved
        // Update position
        //

        TUIO::TuioCursor *cur = getTuioCursorById(id);

        // A touch point with the given id should exist
        ASSERT(cur != 0);

        // Update the server's cursor list
        tuioServer->updateTuioCursor(cur, x, y);
    }
    break;

    case TPS_STATIONARY:
    {
        //
        // Touch point did not move
        // Ignore
        //
    }
    break;
    }
}

TUIO::TuioCursor *MyTuioServer::getTuioCursorById(int id)
{
    if (!tuioServer)
        return 0;

    TuioCursorMap::iterator pos = cursorMap.find(id);

    if (pos != cursorMap.end())
    {
        return cursorMap[id];
    }

    return 0;
}

bool MyTuioServer::isCursorListEmpty()
{
    return cursorMap.empty();
}
#endif
