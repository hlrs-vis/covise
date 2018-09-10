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

#include "vvremoteclient.h"

#include "gl/util.h"
#include "math/math.h"
#include "math/serialization.h"
#include "private/vvmessages.h"
#include "vvvoldesc.h"

using virvo::makeMessage;
using virvo::Message;

namespace gl = virvo::gl;

vvRemoteClient::vvRemoteClient(vvRenderer::RendererType type, vvVolDesc *vd, vvRenderState renderState,
        virvo::ConnectionPointer conn, const std::string& /*filename*/)
    : vvRenderer(vd, renderState)
{
    rendererType = type;
}

vvRemoteClient::~vvRemoteClient()
{
    if (conn_)
        conn_->remove_handler();
}

bool vvRemoteClient::beginFrame(unsigned /*clearMask*/)
{
    return true;
}

bool vvRemoteClient::endFrame()
{
    return true;
}

void vvRemoteClient::renderVolumeGL()
{
    proj_ = gl::getProjectionMatrix();
    view_ = gl::getModelviewMatrix();

    render();
}

bool vvRemoteClient::resize(int w, int h)
{
    virvo::RenderTarget* rt = getRenderTarget();

    if (rt->width() == w && rt->height() == h)
        return true;

    if (conn_)
        conn_->write(makeMessage(Message::WindowResize, virvo::messages::WindowResize(w, h)));

    return vvRenderer::resize(w, h);
}

bool vvRemoteClient::present() const
{
    return true;
}

void vvRemoteClient::setCurrentFrame(size_t index)
{
    if (conn_)
        conn_->write(makeMessage(Message::CurrentFrame, index));

    vvRenderer::setCurrentFrame(index);
}

void vvRemoteClient::setObjectDirection(virvo::vec3f const& od)
{
    if (conn_)
        conn_->write(makeMessage(Message::ObjectDirection, od));

    vvRenderer::setObjectDirection(od);
}

void vvRemoteClient::setViewingDirection(virvo::vec3f const& vd)
{
    if (conn_)
        conn_->write(makeMessage(Message::ViewingDirection, vd));

    vvRenderer::setViewingDirection(vd);
}

void vvRemoteClient::setPosition(virvo::vec3f const& p)
{
    if (conn_)
        conn_->write(makeMessage(Message::Position, p));

    vvRenderer::setPosition(p);
}

void vvRemoteClient::updateTransferFunction()
{
#if 0
    if (conn_)
        conn_->write(makeMessage(Message::TransFuncChanged, true));
#else
    if (conn_)
        conn_->write(makeMessage(Message::TransFunc, vd->tf));
#endif

    vvRenderer::updateTransferFunction();
}

void vvRemoteClient::setParameter(ParameterType name, const vvParam& value)
{
    vvRenderer::setParameter(name, value);

    if (conn_)
        conn_->write(makeMessage(Message::Parameter, virvo::messages::Param(name, value)));
}

void vvRemoteClient::setVolDesc(vvVolDesc* voldesc)
{
    vvRenderer::setVolDesc(voldesc);

    if (conn_)
        conn_->write(makeMessage(Message::Volume, *voldesc));
}

void vvRemoteClient::init()
{
}

void vvRemoteClient::init_connection(virvo::ConnectionPointer conn)
{
    conn_ = conn;
    conn_->write(makeMessage(Message::Volume, *vd));
    conn_->write(makeMessage(Message::RemoteServerType, rendererType));

    virvo::messages::Param p(vvRenderState::VV_USE_IBR, rendererType == REMOTE_IBR);

    // Enable/Disable IBR
    conn_->write(makeMessage(Message::Parameter, p));
}
