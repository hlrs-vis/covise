// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VSERVER_SIMPLE_SERVER_H
#define VSERVER_SIMPLE_SERVER_H

#include "server.h"

#include <virvo/private/message_queue.h>
#include <virvo/private/work_queue.h>
#include <virvo/vvrenderer.h>

#include <boost/thread/thread.hpp>

#include <memory>

class vvRemoteServer;
class vvRenderContext;
class vvVolDesc;

class vvSimpleServer : public vvServer
{
    typedef vvServer BaseType;

public:
    typedef virvo::ConnectionPointer ConnectionPointer;
    typedef virvo::MessagePointer MessagePointer;

    // Constructor.
    vvSimpleServer(ConnectionPointer conn);

    // Destructor.
    virtual ~vvSimpleServer();

    // Starts processing messages
    void start();

    // Stops processing messages
    void stop();

    // Called when a new message has successfully been read from the server.
    virtual void on_read(MessagePointer message) VV_OVERRIDE;

    // Called when a message has successfully been written to the server.
    virtual void on_write(MessagePointer message) VV_OVERRIDE;

private:
    bool createRenderContext(int w = -1/*use default*/, int h = -1/*use default*/);

    bool createRemoteServer(vvRenderer::RendererType type);

    void processSingleMessage(MessagePointer const& message);
    void processMessages();

    void processNull(MessagePointer const& message);
    void processCameraMatrix(MessagePointer const& message);
    void processCurrentFrame(MessagePointer const& message);
    void processDisconnect(MessagePointer const& message);
    void processGpuInfo(MessagePointer const& message);
    void processObjectDirection(MessagePointer const& message);
    void processParameter(MessagePointer const& message);
    void processPosition(MessagePointer const& message);
    void processRemoteServerType(MessagePointer const& message);
    void processServerInfo(MessagePointer const& message);
    void processStatistics(MessagePointer const& message);
    void processTransFunc(MessagePointer const& message);
    void processTransFuncChanged(MessagePointer const& message);
    void processViewingDirection(MessagePointer const& message);
    void processVolume(MessagePointer const& message);
    void processVolumeFile(MessagePointer const& message);
    void processWindowResize(MessagePointer const& message);

    void handleNewVolume();

private:
    // The message queue
    virvo::MessageQueue queue_;
    // The thread to process incoming messages
    boost::thread worker_;
    // Whether to stop processing messages
    bool cancel_;
    // The current volume
    std::auto_ptr<vvVolDesc> volume_;
    // The current render context
    std::auto_ptr<vvRenderContext> renderContext_;
    // The current remote server (IBR or Image)
    std::auto_ptr<vvRemoteServer> server_;
    // The current renderer
    std::auto_ptr<vvRenderer> renderer_;
    // The current renderer type
    vvRenderer::RendererType rendererType_;
    // Work queue (to compress images)
    virvo::WorkQueue workQueue_;
};

#endif
