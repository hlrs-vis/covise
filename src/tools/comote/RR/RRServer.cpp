/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRServer.cpp

#include "../Debug.h"

#include <vector>
#include <algorithm>
#include <memory>

#include "RRServer.h"
#include "RRXEvent.h"
#include "RRFrame.h"
#include "RRCompressedTile.h"
#include "RRDecompressor.h"

#include "tjplanar.h"

#include <QHostAddress>

#if !defined(__APPLE__) || !defined(_OPENMP)
#define DECOMP_IN_THREAD
#endif

RRServer::RRServer(const QString &hostname, unsigned short port, bool planarOutput)
    : stopped(false)
    , connected(false)
    , hostname(hostname)
    , port(port)
    , frame(0)
    , planarOutput(planarOutput)
{
    frame = new RRFrame(planarOutput);

    connect(this, SIGNAL(finished()), this, SLOT(onFinished()));
}

RRServer::~RRServer()
{
    delete frame;
}

int RRServer::recv(QTcpSocket *socket, void *data, int len)
{
    ASSERT(socket);
    ASSERT(data != 0 && len > 0);

    // need to synchronize reads
    while ((int)socket->bytesAvailable() < len)
    {
        if (!socket->waitForReadyRead(30000 /*ms*/))
        {
            Log() << "RRServer::recv: wait failed";
            return -1;
        }
    }

    qint64 received = socket->read((char *)data, len);
    if (received < 0)
    {
        Log() << "RRServer::recv: read failed";
        return -1;
    }

    return 0;
}

int RRServer::send(QTcpSocket *socket, void *data, int len)
{
    ASSERT(socket);
    ASSERT(data != 0 && len > 0);

    qint64 sent = socket->write((char *)data, len);
    if (sent < 0)
    {
        Log() << "RRServer::send: write failed";
        return -1;
    }

    return 0;
}

void RRServer::endianize(rrframeheader & /*h*/)
{
    ASSERT(*(unsigned int *)"\x1\x0\x0\x0" == 1);
}

void RRServer::endianize(rrframeheader_v1 & /*h*/)
{
    ASSERT(*(unsigned int *)"\x1\x0\x0\x0" == 1);
}

void RRServer::convertHeader(rrframeheader &h, const rrframeheader_v1 &h1)
{
    h.size = h1.size;
    h.winid = h1.winid;
    h.framew = h1.framew;
    h.frameh = h1.frameh;
    h.width = h1.width;
    h.height = h1.height;
    h.x = h1.x;
    h.y = h1.y;
    h.qual = h1.qual;
    h.subsamp = h1.subsamp;
    h.flags = h1.flags;
    h.dpynum = static_cast<unsigned short>(h1.dpynum);
}

void RRServer::sendEvent(const rrxevent &rrev)
{
    if (stopped)
        return;

    eventQueueLock.lock();
    {
        // See if the event can be merged with an existing one
        switch (rrev.type)
        {
        case RREV_RESIZE:
        case RREV_MOTION: //allowed?
            if (eventQueue.size() > 0 && eventQueue.back().type == rrev.type)
            {
                eventQueue.pop_back();
            }
            break;
        }

        // Add event to queue
        eventQueue.push_back(rrev);
    }
    eventQueueLock.unlock();
}

int RRServer::processEvents(QTcpSocket *socket)
{
    ASSERT(socket);

    int size = 0;

    // process pending events
    eventQueueLock.lock();
    {
        size = (int)eventQueue.size();

        while (!eventQueue.empty())
        {
            rrxevent &rrev = eventQueue.front();

            send(socket, &rrev, sizeof(rrxevent));

            eventQueue.pop_front();
        }

        // send null event, ie. all xevents have been processed
        rrxevent rrev;
        send(socket, &rrev, sizeof(rrev));
    }
    eventQueueLock.unlock();

    return size;
}

void RRServer::run()
{
    Log() << "RRServer::run: enter...";

    connected = false;

    // Create socket
    std::auto_ptr<QTcpSocket> socket(new QTcpSocket);

    emit signalMessage(Message_Connecting, hostname, port);

    socket->connectToHost(hostname, port);

    if (!socket->waitForConnected(30000 /*ms*/))
    {
        emit signalMessage(Message_Failed, hostname, port);

        return;
    }

    // Stop this thread if the connection is lost/canceled
    connect(socket.get(), SIGNAL(disconnected()), this, SLOT(stop()));

    connected = true;

    emit signalMessage(Message_Connected, hostname, port);

    //
    // Read server version
    //

    rrframeheader_v1 h1;

    memset(&h1, 0, sizeof_rrframeheader_v1);

    if (recv(socket.get(), &h1, sizeof_rrframeheader_v1) < 0)
    {
    }

    endianize(h1);

    if (h1.framew != 0 && h1.frameh != 0 && h1.width != 0 && h1.height != 0 && h1.winid != 0 && h1.size != 0 && h1.flags != RR_EOF)
    {
        ASSERT(0); // Version 1.0 -- not supported
    }
    else
    {
        rrversion v = {
            { 'V', 'G', 'L' }, RR_MAJOR_VERSION, RR_MINOR_VERSION
        };

        if (send(socket.get(), &v, sizeof_rrversion) < 0)
        {
        }
        if (recv(socket.get(), &v, sizeof_rrversion) < 0)
        {
        }

        /*
      if (strncmp(v.id, "VGL", 3) != 0 || v.major < 1)
      {
          // error reading server version
      }
      */

        Log() << "RRServer::run: server version is " << (int)v.major << "." << (int)v.minor;
    }

//
// Process frames
//

#ifdef DECOMP_IN_THREAD
    RRTileDecompressor decompressor(planarOutput);
#endif

    while (!stopped) // Process frames
    {
        RRTileDecompressor::TileVector *tiles = new RRTileDecompressor::TileVector;

        rrframeheader h;

        memset(&h, 0, sizeof_rrframeheader);

        do // Process tiles
        {
            // Read tile header
            if (recv(socket.get(), &h, sizeof_rrframeheader) < 0)
            {
                Log() << "RRServer::run: connection lost";
                return;
            }

            endianize(h);

            // Stereo not supported yet
            ASSERT(h.flags != RR_LEFT && h.flags != RR_RIGHT);

            frame->setSubSampling(h.subsamp,
                                  tjMCUWidth[jpegsub(h.subsamp)] / 8,
                                  tjMCUHeight[jpegsub(h.subsamp)] / 8);

            if (!frame->resize(h.framew, h.frameh))
            {
                Log() << "RRServer::run: out of memory";
                return;
            }

            if (h.flags != RR_EOF && h.size > 0)
            {
                RRCompressedTile tile;

                tile.buffer = new unsigned char[h.size];
                tile.size = h.size;
                tile.x = h.x;
                tile.y = h.y;
                tile.w = h.width;
                tile.h = h.height;

                if (recv(socket.get(), tile.buffer, tile.size) < 0)
                {
                    Log() << "RRServer::run: connection lost";
                    return;
                }

                tiles->push_back(tile);
            }
        } while (h.flags != RR_EOF);

        // Process pending events
        processEvents(socket.get());

#ifdef DECOMP_IN_THREAD
        // Decompress tiles
        // Release resources in case thread is stopped
        decompressor.run(tiles, frame, true);
#endif

        if (stopped)
        {
            break;
        }

#ifdef DECOMP_IN_THREAD
        // A new frame is available
        // The scribble widget might now request the frame and draw it
        if (!tiles->empty())
        {
            emit signalFrameComplete();
        }
        delete tiles;
#else
        if (!tiles->empty())
        {
            emit signalFrameReceived(tiles);
        }
#endif
    }

    connected = false;

    emit signalMessage(Message_Disconnected, hostname, port);

    Log() << "RRServer::run: leave...";
}

void RRServer::onFinished()
{
    Log() << "RRServer::run: finished...";

    emit signalMessage(Message_Disconnected, hostname, port);
}
