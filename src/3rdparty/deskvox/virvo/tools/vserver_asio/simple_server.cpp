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

#include "simple_server.h"

#include <virvo/gl/util.h>
#include <virvo/math/serialization.h>
#include <virvo/private/vvgltools.h>
#include <virvo/private/vvibrimage.h>
#include <virvo/private/vvimage.h>
#include <virvo/private/vvmessages.h>
#include <virvo/private/vvtimer.h>
#include <virvo/private/work_queue.h>
#include <virvo/vvfileio.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvparam.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvrendercontext.h>
#include <virvo/vvrendererfactory.h>
#include <virvo/vvrendertarget.h>
#include <virvo/vvserialization.h>
#include <virvo/vvvoldesc.h>

#include <cassert>
#include <cstdio>

using virvo::vec3;

#define TIME_FPS 0
#define TIME_READS 0
#define TIME_WRITES 0

static const int DEFAULT_WINDOW_SIZE = 512;

vvSimpleServer::vvSimpleServer(ConnectionPointer conn)
    : BaseType(conn)
    , cancel_(true)
    , rendererType_(vvRenderer::INVALID)
{
    start();
}

vvSimpleServer::~vvSimpleServer()
{
    stop();
}

void vvSimpleServer::start()
{
    assert(cancel_ == true && "server already running");

    // Reset state
    cancel_ = false;

    // Create a new thread
    worker_ = boost::thread(&vvSimpleServer::processMessages, this);

    // Start the work queue
    workQueue_.run_in_thread();
}

void vvSimpleServer::stop()
{
    if (cancel_)
        return;

    // Tell the thread to cancel
    cancel_ = true;
    // Wake up the thread in case it's sleeping
    queue_.push_back(virvo::makeMessage());

    // Wait for the thread to finish
    worker_.join();

    // Stop the work queue
    workQueue_.stop();
}

void vvSimpleServer::on_read(MessagePointer message)
{
#if TIME_READS
    static virvo::FrameCounter counter;

    printf("IBR: reads/sec: %.2f\n", counter.registerFrame());
#endif

    switch (message->type())
    {
    case virvo::Message::CameraMatrix:
    case virvo::Message::TransFunc:
    case virvo::Message::TransFuncChanged:
    case virvo::Message::WindowResize:
        queue_.push_back_merge(message);
        break;
    default:
        queue_.push_back(message);
        break;
    }
}

void vvSimpleServer::on_write(MessagePointer /*message*/)
{
#if TIME_WRITES
    static virvo::FrameCounter counter;

    printf("IBR: writes/sec: %.2f\n", counter.registerFrame());
#endif
}

bool vvSimpleServer::createRenderContext(int w, int h)
{
    assert(renderContext_.get() == 0);

    // Destroy the old context first - if any.
    renderContext_.reset();

    if (w <= 0) w = DEFAULT_WINDOW_SIZE;
    if (h <= 0) h = DEFAULT_WINDOW_SIZE;

    vvContextOptions options;

    options.type = vvContextOptions::VV_PBUFFER;
    options.width = w;
    options.height = h;
    options.displayName = "";

    // Create the new context
    renderContext_.reset(new vvRenderContext(options));

    return renderContext_->makeCurrent();
}

bool vvSimpleServer::createRemoteServer(vvRenderer::RendererType type)
{
    assert(volume_.get());

    // Create a render context if not already done.
    if (renderContext_.get() == 0 && !createRenderContext())
        return false;

    // Create a new remote server
    switch (type)
    {
    case vvRenderer::REMOTE_IBR:
        server_.reset(new vvIbrServer);
        break;

    case vvRenderer::REMOTE_IMAGE:
        server_.reset(new vvImageServer);
        break;

    default:
        assert(0 && "unhandled renderer type");
        return false;
    }

    // Install a debug callback to catch GL errors
    virvo::gl::enableDebugCallback();

    vvRenderState state;

#if 0
    if (volume_->tf.isEmpty()) // Set default color scheme!
    {
        float min = volume_->real[0];
        float max = volume_->real[1];

        volume_->tf.setDefaultAlpha(0, min, max);
        volume_->tf.setDefaultColors(volume_->getChan() == 1 ? 0 : 2, min, max);
    }
#endif

#if 1
    //
    // TODO:
    // FIXME!
    //

    // Create a new renderer
    renderer_.reset( vvRendererFactory::create( volume_.get(),
                                                state,
                                                type == vvRenderer::REMOTE_IMAGE ? "default" : "rayrendcuda",
                                                vvRendererFactory::Options() ));

    if (type == vvRenderer::REMOTE_IMAGE)
    {
        renderer_->setRenderTarget(virvo::FramebufferObjectRT::create(virvo::PF_RGBA8, virvo::PF_DEPTH24_STENCIL8));
    }
#else
    if (type == vvRenderer::REMOTE_IMAGE)
    {
        state.setParameter(vvRenderer::VV_USE_OFFSCREEN_BUFFER, true);
        state.setParameter(vvRenderer::VV_IMAGE_PRECISION, 8);
    }

    // Create a new renderer
    renderer_.reset( vvRendererFactory::create( volume_.get(),
                                                state,
                                                type == vvRenderer::REMOTE_IMAGE ? "default" : "rayrendcuda",
                                                vvRendererFactory::Options() ));
#endif

    if (renderer_.get() == 0)
    {
        return false;
    }

    // Create/Resize the render target
    renderer_->resize(DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE);

    // Enable IBR
    renderer_->setParameter(vvRenderer::VV_USE_IBR, type == vvRenderer::REMOTE_IBR);
#if 0
    renderer_->updateTransferFunction();
#endif

    return true;
}

void vvSimpleServer::processSingleMessage(MessagePointer const& message)
{
#define X(M) \
    case virvo::Message::M: process##M(message); break;

    switch (message->type())
    {
    X(CameraMatrix)
    X(CurrentFrame)
    X(Disconnect)
    X(GpuInfo)
    X(ObjectDirection)
    X(Parameter)
    X(Position)
    X(RemoteServerType)
    X(ServerInfo)
    X(Statistics)
    X(TransFunc)
    X(TransFuncChanged)
    X(ViewingDirection)
    X(Volume)
    X(VolumeFile)
    X(WindowResize)
    default:
        break;
    }

#undef X
}

void vvSimpleServer::processMessages()
{
    try
    {
        while (!cancel_)
        {
            processSingleMessage(queue_.pop_front());
        }
    }
    catch (std::exception& e)
    {
#ifndef NDEBUG
        printf("vserver: Exception caught: %s", e.what());
#else
        static_cast<void>(e);
#endif
        throw;
    }
}

void vvSimpleServer::processNull(MessagePointer const& /*message*/)
{
}

void vvSimpleServer::processCameraMatrix(MessagePointer const& message)
{
#if TIME_FPS
    static virvo::FrameCounter counter;

    printf("IBR: FPS: %.2f\n", counter.registerFrame());
#endif

    assert(renderer_.get());

    // Extract the matrices from the message
    virvo::messages::CameraMatrix p = message->deserialize<virvo::messages::CameraMatrix>();

    // Render a new image
    server_->renderImage(conn(), message, p.proj, p.view, renderer_.get(), workQueue_);
}

void vvSimpleServer::processCurrentFrame(MessagePointer const& message)
{
    assert(renderer_.get());

    // Extract the frame number from the message
    size_t frame = message->deserialize<size_t>();

    // Set the current frame
    renderer_->setCurrentFrame(frame);
}

void vvSimpleServer::processDisconnect(MessagePointer const& /*message*/)
{
}

void vvSimpleServer::processGpuInfo(MessagePointer const& /*message*/)
{
}

void vvSimpleServer::processObjectDirection(MessagePointer const& message)
{
    assert(renderer_.get());

    // Extract the 3D-vector from the message
    // and set the new direction
    renderer_->setObjectDirection(message->deserialize<vec3>());
}

void vvSimpleServer::processParameter(MessagePointer const& message)
{
    assert(renderer_.get());

    // Extract the parameters from the message
    virvo::messages::Param p = message->deserialize<virvo::messages::Param>();

    // Set the new renderer parameter
    renderer_->setParameter(p.name, p.value);
}

void vvSimpleServer::processPosition(MessagePointer const& message)
{
    assert(renderer_.get());

    // Extract the 3D-vector from the message
    // and set the new direction
    renderer_->setPosition(message->deserialize<vec3>());
}

void vvSimpleServer::processRemoteServerType(MessagePointer const& message)
{
    assert(volume_.get());

    // Get the requested renderer type from the message
    vvRenderer::RendererType type = message->deserialize<vvRenderer::RendererType>();

    // Create the renderer!
    createRemoteServer(type);
}

void vvSimpleServer::processServerInfo(MessagePointer const& /*message*/)
{
}

void vvSimpleServer::processStatistics(MessagePointer const& /*message*/)
{
}

void vvSimpleServer::processTransFunc(MessagePointer const& message)
{
    assert(renderer_.get());

    // Extract the transfer function
    message->deserialize(renderer_->getVolDesc()->tf);
    // Update the transfer function
    renderer_->updateTransferFunction();
}

void vvSimpleServer::processTransFuncChanged(MessagePointer const& message)
{
    conn()->write(message);
}

void vvSimpleServer::processViewingDirection(MessagePointer const& message)
{
    assert(renderer_.get());

    // Extract the 3D-vector from the message
    // and set the new direction
    renderer_->setViewingDirection(message->deserialize<vec3>());
}

void vvSimpleServer::processVolume(MessagePointer const& message)
{
    // Create a new volume
    volume_.reset(new vvVolDesc("{no-volume-name}"));

    // Extract the volume from the message
    message->deserialize(*volume_);

    volume_->printInfoLine();

    // Update the volume
    handleNewVolume();
}

void vvSimpleServer::processVolumeFile(MessagePointer const& message)
{
    // Extract the filename from the message
    std::string filename = message->deserialize<std::string>();

    // Create a new volume description
    volume_.reset(new vvVolDesc(filename.c_str()));

    // Load the volume
    vvFileIO fileIO;

    if (fileIO.loadVolumeData(volume_.get()) == vvFileIO::OK)
    {
        volume_->printInfoLine();
    }
    else
    {
        volume_->printInfoLine(); // volume_.reset();
    }

    // Update the volume
    handleNewVolume();
}

void vvSimpleServer::processWindowResize(MessagePointer const& message)
{
    assert(renderer_.get());

    // Extract the window size from the message
    virvo::messages::WindowResize p;

    message->deserialize(p);

    // Set the new window size
    renderer_->resize(p.w, p.h);
}

void vvSimpleServer::handleNewVolume()
{
    if (volume_.get() == 0)
        return;

    float min = volume_->range(0)[0];
    float max = volume_->range(0)[1];

    if (volume_->tf[0].isEmpty())
    {
        volume_->tf[0].setDefaultAlpha(0, min, max);
        volume_->tf[0].setDefaultColors(volume_->getChan() == 1 ? 0 : 2, min, max);
    }

    if (volume_->bpc == 4 && min == 0.0f && max == 1.0f)
    {
        volume_->findAndSetRange();
    }

    if (renderer_.get())
    {
#if 0
        volume_->resizeEdgeMax(renderer_->getRenderTarget()->width() * 0.6f);
#endif
        renderer_->setVolDesc(volume_.get());
    }
}
