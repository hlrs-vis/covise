/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
	TUIO C++ Library - part of the reacTIVision project
	http://reactivision.sourceforge.net/

	Copyright (c) 2005-2008 Martin Kaltenbrunner <mkalten@iua.upf.edu>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef INCLUDED_TUIOCLIENT_H
#define INCLUDED_TUIOCLIENT_H

#ifndef WIN32
#include <pthread.h>
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include <iostream>
#include <list>
#include <algorithm>

#include "osc/OscReceivedElements.h"
#include "osc/OscPrintReceivedElements.h"

#include "ip/UdpSocket.h"
#include "ip/PacketListener.h"

#include "TuioListener.h"
#include "TuioObject.h"
#include "TuioCursor.h"

#include <OpenThreads/Thread>

#include <OpenThreads/Mutex>

using namespace osc;

class TuioClient : public PacketListener, public OpenThreads::Thread
{

public:
    TuioClient();
    TuioClient(int p);
    ~TuioClient();

    virtual void run();
    void stop();

    TuioObject *getTuioObject(long s_id);
    TuioCursor *getTuioCursor(long s_id);
    //std::list<TuioObject*> getTuioObjects();         // Change BBM
    std::list<TuioObject *> *getTuioObjects();
    // std::list<TuioCursor*>  getTuioCursors();        // Change BBM
    std::list<TuioCursor *> *getTuioCursors();

    void addTuioListener(TuioListener *listener);
    void removeTuioListener(TuioListener *listener);

    void ProcessPacket(const char *data, int size, const IpEndpointName &remoteEndpoint);

    UdpListeningReceiveSocket *socket;

    //		std::list<TuioObject*> objectList;
    //		std::list<long> aliveObjectList, objectBuffer;
    //		std::list<TuioCursor*> cursorList;
    //		std::list<long> aliveCursorList, cursorBuffer;

    OpenThreads::Mutex mutex;

protected:
    void ProcessBundle(const ReceivedBundle &b, const IpEndpointName &remoteEndpoint);
    void ProcessMessage(const ReceivedMessage &m, const IpEndpointName &remoteEndpoint);

private:
    std::list<TuioListener *> listenerList;

    std::list<TuioObject *> objectList;
    std::list<long> aliveObjectList, objectBuffer;
    std::list<TuioCursor *> cursorList;
    std::list<long> aliveCursorList, cursorBuffer;

    int32 currentFrame, lastFrame;

    long startTime;
    long lastTime;
    long getCurrentTime();

    std::list<TuioCursor *> freeCursorList, freeCursorBuffer;
    int maxFingerID;

#ifndef WIN32
    pthread_t thread;
#else
    HANDLE thread;
#endif

    bool locked;
    bool running;
};

#endif /* INCLUDED_TUIOCLIENT_H */
