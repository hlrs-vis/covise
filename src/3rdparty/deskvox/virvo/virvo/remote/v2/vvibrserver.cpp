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

#include "vvibrserver.h"

#include "gl/util.h"
#include "math/serialization.h"
#include "private/vvibrimage.h"
#include "private/vvtimer.h"
#include "private/work_queue.h"
#include "vvibr.h"
#include "vvrenderer.h"
#include "vvvoldesc.h"

#include <cassert>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace gl = virvo::gl;
using virvo::mat4;

#define TIME 1
#define TIME_VERBOSE 1

inline int percent(double t, double x) {
    return static_cast<int>(100.0 * t / x + 0.5);
}

static void CompressSerializeAndSend(virvo::ConnectionPointer conn, virvo::MessagePointer message, boost::shared_ptr<virvo::IbrImage> image)
{
    // Compress the image
    image->compress();

    // Serialize the image
    message->reset(virvo::Message::IbrImage, *image);

    // Send the image
    conn->write(message);
}

vvIbrServer::vvIbrServer()
    : vvRemoteServer()
{
}

vvIbrServer::~vvIbrServer()
{
}

void vvIbrServer::renderImage(ConnectionPointer conn, MessagePointer message,
    mat4 const& pr, mat4 const& mv, vvRenderer* renderer, virvo::WorkQueue& queue)
{
#if 1

#if TIME
    static virvo::FrameCounter counter;

    printf("IBR-server: FPS: %.2f\n", counter.registerFrame());
#endif

    assert(renderer->getRenderTarget()->width() > 0);
    assert(renderer->getRenderTarget()->height() > 0);

    // Update matrices
    gl::setProjectionMatrix(pr);
    gl::setModelviewMatrix(mv);

    // Render volume:
    renderer->beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
    renderer->renderVolumeGL();
    renderer->endFrame();

    // Compute depth range
    virvo::aabb box = renderer->getVolDesc()->getBoundingBox();

    float drMin = 0.0f;
    float drMax = 0.0f;

    virvo::ibr::calcDepthRange(pr, mv, box, drMin, drMax);

    renderer->setParameter(vvRenderer::VV_IBR_DEPTH_RANGE, virvo::vec2f(drMin, drMax));

    virvo::RenderTarget* rt = renderer->getRenderTarget();

    int w = rt->width();
    int h = rt->height();

    // Create a new IBR image
    boost::shared_ptr<virvo::IbrImage> image = boost::make_shared<virvo::IbrImage>(w, h, rt->colorFormat(), rt->depthFormat());

    image->setDepthMin(drMin);
    image->setDepthMax(drMax);
    image->setViewMatrix(mv);
    image->setProjMatrix(pr);
    image->setViewport(virvo::recti(0, 0, w, h));

    // Fetch rendered image
    if (!rt->downloadColorBuffer(image->colorBuffer().data().ptr(), image->colorBuffer().size()))
    {
        std::cout << "vvIbrServer: download color buffer failed" << std::endl;
        return;
    }
    if (!rt->downloadDepthBuffer(image->depthBuffer().data().ptr(), image->depthBuffer().size()))
    {
        std::cout << "vvIbrServer: download depth buffer failed" << std::endl;
        return;
    }

    queue.post(boost::bind(&CompressSerializeAndSend, conn, message, image));

#else

#if TIME_VERBOSE
    virvo::Timer timer;
#endif

    assert(renderer->getRenderTarget()->width() > 0);
    assert(renderer->getRenderTarget()->height() > 0);

    // Update matrices
    vvGLTools::setProjectionMatrix(pr);
    vvGLTools::setModelviewMatrix(mv);

    // Render volume:
    renderer->beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
    renderer->renderVolumeGL();
    renderer->endFrame();

    // Compute depth range
    aabb box = renderer->getVolDesc()->getBoundingBox();

    float drMin = 0.0f;
    float drMax = 0.0f;

    vvIbr::calcDepthRange(pr, mv, box, drMin, drMax);

    renderer->setParameter(vvRenderer::VV_IBR_DEPTH_RANGE, vvVector2(drMin, drMax));

    virvo::RenderTarget* rt = renderer->getRenderTarget();

    int w = rt->width();
    int h = rt->height();

    // Create a new IBR image
    virvo::IbrImage image(w, h, rt->colorFormat(), rt->depthFormat());

    image.setDepthMin(drMin);
    image.setDepthMax(drMax);
    image.setViewMatrix(mv);
    image.setProjMatrix(pr);
    image.setViewport(virvo::Viewport(0, 0, w, h));

    // Fetch rendered image
    if (!rt->downloadColorBuffer(image.colorBuffer().data().ptr(), image.colorBuffer().size()))
    {
        std::cout << "vvIbrServer: download color buffer failed" << std::endl;
        return;
    }
    if (!rt->downloadDepthBuffer(image.depthBuffer().data().ptr(), image.depthBuffer().size()))
    {
        std::cout << "vvIbrServer: download depth buffer failed" << std::endl;
        return;
    }

#if TIME_VERBOSE
    double tRender = timer.lapSeconds();
#endif

    // Compress the image
    if (!image.compress(/*virvo::Compress_JPEG*/))
    {
        std::cout << "vvIbrServer::renderImage: failed to compress the image.\n";
        return;
    }

#if TIME_VERBOSE
    double tCompress = timer.lapSeconds();
#endif

    // Serialize the image
    message->reset(virvo::Message::IbrImage, image);

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

    printf("IBR-server: FPS: %.2f [%.2f ms (render: %d%%, compress: %d%%, serialize: %d%%)]\n",
        counter.registerFrame(), t * 1000.0, pRender, pCompress, pSerialize);
#else
    printf("IBR-server: FPS: %.2f\n", counter.registerFrame());
#endif
#endif

#endif
}
