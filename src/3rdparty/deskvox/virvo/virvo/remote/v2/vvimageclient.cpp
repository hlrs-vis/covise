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

#include "vvimageclient.h"

#include "gl/util.h"
#include "private/vvimage.h"
#include "private/vvmessages.h"
#include "private/vvtimer.h"
#include "vvvoldesc.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <iostream>

using virvo::makeMessage;
using virvo::Message;

#define TIME_FPS 0
#define TIME_READS 0
#define TIME_WRITES 0
#define TIME_NEW_IMAGES 1

struct vvImageClient::Impl
{
    // The mutex to protect the members below
    boost::mutex lock;
    // The current image to render
    std::auto_ptr<virvo::Image> curr;
    // The next image
    std::auto_ptr<virvo::Image> next;

    void setNextImage(virvo::Image* image)
    {
        boost::unique_lock<boost::mutex> guard(lock);

        next.reset(image);
    }

    bool fetchNextImage()
    {
        boost::unique_lock<boost::mutex> guard(lock);

        if (next.get() == 0)
            return false;

        curr.reset(next.release());
        return true;
    }
};

vvImageClient::vvImageClient(vvVolDesc *vd, vvRenderState renderState,
        virvo::ConnectionPointer conn, std::string const& filename)
    : vvRemoteClient(REMOTE_IMAGE, vd, renderState, conn, filename)
    , impl_(new Impl)
{
    conn->set_handler(boost::bind(&vvImageClient::handler, this, _1, _2, _3));

    init();
    init_connection(conn);
}

vvImageClient::~vvImageClient()
{
}

bool vvImageClient::render()
{
#if TIME_FPS
    static virvo::FrameCounter counter;

    printf("IMAGE-client: FPS: %.2f\n", counter.registerFrame());
#endif

    // Send a new request
    if (conn_)
        conn_->write(makeMessage(Message::CameraMatrix, virvo::messages::CameraMatrix(view(), proj())));

    if (impl_->fetchNextImage())
    {
        // Decompress
        if (!impl_->curr->decompress())
        {
            throw std::runtime_error("decompression failed");
        }
    }

    if (impl_->curr.get() == 0)
        return true;

    virvo::Image const& image = *impl_->curr.get();

    // Render the current image
    virvo::PixelFormatInfo f = mapPixelFormat(image.format());
    virvo::gl::blendPixels(image.width(), image.height(), f.format, f.type, image.data().ptr());

    return true;
}

void vvImageClient::handler(virvo::Connection::Reason reason, virvo::MessagePointer message, boost::system::error_code const& e)
{
    if (e)
    {
        conn_->close();
        conn_.reset();
        return;
    }

    switch (reason)
    {
    case virvo::Connection::Read:
        on_read(message);
        break;
    case virvo::Connection::Write:
        on_write(message);
        break;
    }
}

void vvImageClient::on_read(virvo::MessagePointer message)
{
#if TIME_READS
    static virvo::FrameCounter counter;

    printf("IMAGE-client: reads/sec: %.2f\n", counter.registerFrame());
#endif

    switch (message->type())
    {
    case virvo::Message::Image:
        processImage(message);
        break;
    }
}

void vvImageClient::on_write(virvo::MessagePointer message)
{
#if TIME_WRITES
    static virvo::FrameCounter counter;

    printf("IBR-client: writes/sec: %.2f\n", counter.registerFrame());
#endif
}

void vvImageClient::processImage(virvo::MessagePointer message)
{
#if TIME_NEW_IMAGES
    static virvo::FrameCounter counter;

    printf("IMAGE-client: images/sec %.2f\n", counter.registerFrame());
#endif

    // Create a new image
    std::auto_ptr<virvo::Image> image(new virvo::Image);

    // Extract the image from the message
    //
    // FIXME:
    // Move into render() ??
    //
    if (!message->deserialize(*image))
    {
        throw std::runtime_error("deserialization failed");
    }

    // Update the next image
    impl_->setNextImage(image.release());
}
