/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRServer.h

#ifndef RR_SERVER_H
#define RR_SERVER_H

#include "../Debug.h"

#include <deque>

#include <QThread>
#include <QTcpSocket>
#include <QMutex>
#include <QString>

#include "RR.h"
#include "RRXEvent.h"
#include "RRDecompressor.h"

class RRFrame;

class RRServer : public QThread
{
    Q_OBJECT

public:
    enum Message
    {
        Message_Connecting,
        Message_Connected,
        Message_Disconnected,
        Message_Failed,
        Message_InternalError,
    };

    RRServer(const QString &hostname, unsigned short port, bool planarOutput = false);

    virtual ~RRServer();

    void sendEvent(const rrxevent &rrev);

    void sendMessage(int /*Message*/ type, QString address, quint16 port);

    inline RRFrame *getFrame()
    {
        return frame;
    }

    inline const RRFrame *getFrame() const
    {
        return frame;
    }

    inline bool isConnected() const
    {
        return connected;
    }

signals:
    // New frame complete; time to update
    void signalFrameComplete();

    // New frame received; tiles have to be decompressed
    void signalFrameReceived(RRTileDecompressor::TileVector *tv);

    // Triggered if the status of the connection has changed
    void signalMessage(int /*Message*/ type, QString address, quint16 port);

public slots:
    void stop()
    {
        stopped = true;
    }

private slots:
    void onFinished();

protected:
    virtual void run();

private:
    static int recv(QTcpSocket *socket, void *data, int len);
    static int send(QTcpSocket *socket, void *data, int len);

    static void endianize(rrframeheader &h);
    static void endianize(rrframeheader_v1 &h);
    static void convertHeader(rrframeheader &h, const rrframeheader_v1 &h1);

    int processEvents(QTcpSocket *socket);

private:
    typedef std::deque<rrxevent> EventQueue;

    bool stopped; // whether the thread should terminate
    bool connected;

    QString hostname;
    unsigned short port;

    RRFrame *frame;

    EventQueue eventQueue;

    QMutex eventQueueLock;

    bool planarOutput;
};
#endif
