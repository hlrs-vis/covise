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

#include "vvimageserver.h"

#include "gl/util.h"
#include "private/vvimage.h"
#include "private/vvtimer.h"
#include "private/work_queue.h"
#include "vvrenderer.h"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace gl = virvo::gl;
using virvo::mat4;

#define TIME 1
#define TIME_VERBOSE 1

inline int percent(double t, double x) {
    return static_cast<int>(100.0 * t / x + 0.5);
}

static void CompressSerializeAndSend(virvo::ConnectionPointer conn, virvo::MessagePointer message, boost::shared_ptr<virvo::Image> image)
{
    // Compress the image
    image->compress();

    // Serialize the image
    message->reset(virvo::Message::Image, *image);

    // Send the image
    conn->write(message);
}

vvImageServer::vvImageServer()
    : vvRemoteServer()
{
}

vvImageServer::~vvImageServer()
{
}

void vvImageServer::renderImage(ConnectionPointer conn, MessagePointer message,
        mat4 const& pr, mat4 const& mv, vvRenderer* renderer, virvo::WorkQueue& queue)
{
#if 1

    // Update matrices
    gl::setProjectionMatrix(pr);
    gl::setModelviewMatrix(mv);

    renderer->beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
    renderer->renderVolumeGL();
    renderer->endFrame();

    virvo::RenderTarget* rt = renderer->getRenderTarget();

    int w = rt->width();
    int h = rt->height();

    boost::shared_ptr<virvo::Image> image = boost::make_shared<virvo::Image>(w, h, rt->colorFormat());

    // Fetch rendered image
    if (!rt->downloadColorBuffer(image->data().ptr(), image->size()))
    {
        std::cout << "vvImageServer::renderImage: failed to download color buffer\n";
        return;
    }

    queue.post(boost::bind(CompressSerializeAndSend, conn, message, image));

#else

#if TIME_VERBOSE
    virvo::Timer timer;
#endif

    // Update matrices
    vvGLTools::setProjectionMatrix(pr);
    vvGLTools::setModelviewMatrix(mv);

    renderer->beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
    renderer->renderVolumeGL();
    renderer->endFrame();

    virvo::RenderTarget* rt = renderer->getRenderTarget();

    int w = rt->width();
    int h = rt->height();

    virvo::Image image(w, h, rt->colorFormat());

    // Fetch rendered image
    if (!rt->downloadColorBuffer(image.data().ptr(), image.size()))
    {
        std::cout << "vvImageServer::renderImage: failed to download color buffer\n";
        return;
    }

#if TIME_VERBOSE
    double tRender = timer.lapSeconds();
#endif

    queue.post(boost::bind(CompressSerializeAndSend, conn, message, image));

    // Compress the image
    if (!image.compress())
    {
        std::cout << "vvImageServer::renderImage: failed to compress the image\n";
        return;
    }

#if TIME_VERBOSE
    double tCompress = timer.lapSeconds();
#endif

    // Serialize the image
    message->reset(virvo::Message::Image, image);

#if TIME_VERBOSE
    double tSerialize = timer.lapSeconds();
#endif

#if TIME
    static virvo::FrameCounter counter;

#if TIME_VERBOSE
    double t = timer.elapsedSeconds();

    int pRender     = percent(tRender, t);
    int pCompress   = percent(tCompress, t);
    int pSerialize  = percent(tSerialize, t);

    printf("IMAGE-server: FPS: %.2f [%.2f ms (render: %d%%, compress: %d%%, serialize: %d%%)]\n",
        counter.registerFrame(), t * 1000.0, pRender, pCompress, pSerialize);
#else
    printf("IMAGE-server: FPS: %.2f\n", counter.registerFrame());
#endif
#endif

#endif
}
