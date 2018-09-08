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

#ifndef VV_REMOTE_CLIENT_H
#define VV_REMOTE_CLIENT_H

#include "math/forward.h"
#include "private/connection.h"
#include "private/connection_manager.h"
#include "vvcompiler.h"
#include "vvrenderer.h"

class vvRemoteClient
    : public vvRenderer
{
public:
    typedef virvo::MessagePointer MessagePtr;

public:
    // Constructor
    VVAPI vvRemoteClient(vvRenderer::RendererType type, vvVolDesc *vd, vvRenderState renderState,
        virvo::ConnectionPointer conn, const std::string &filename);

    // Destructor
    VVAPI virtual ~vvRemoteClient();

    // Returns the connection
    virvo::ConnectionPointer& conn() {
        return conn_;
    }

    // Returns the connection
    virvo::ConnectionPointer const& conn() const {
        return conn_;
    }

    // Returns the current projection matrix
    virvo::mat4 const& proj() const { return proj_; }

    // Returns the current model-view matrix
    virvo::mat4 const& view() const { return view_; }

    // vvRenderer API ----------------------------------------------------------

    VVAPI virtual bool beginFrame(unsigned clearMask) VV_OVERRIDE;
    VVAPI virtual bool endFrame() VV_OVERRIDE;
    VVAPI virtual bool present() const VV_OVERRIDE;
    VVAPI virtual bool render() = 0;
    VVAPI virtual void renderVolumeGL() VV_OVERRIDE;
    VVAPI virtual bool resize(int w, int h) VV_OVERRIDE;
    VVAPI virtual void setCurrentFrame(size_t index) VV_OVERRIDE;
    VVAPI virtual void setObjectDirection(virvo::vec3f const& od) VV_OVERRIDE;
    VVAPI virtual void setParameter(ParameterType param, const vvParam& value) VV_OVERRIDE;
    VVAPI virtual void setPosition(virvo::vec3f const& p) VV_OVERRIDE;
    VVAPI virtual void setViewingDirection(virvo::vec3f const& vd) VV_OVERRIDE;
    VVAPI virtual void setVolDesc(vvVolDesc* voldesc) VV_OVERRIDE;
    VVAPI virtual void updateTransferFunction() VV_OVERRIDE;

protected:
    void init();
    void init_connection(virvo::ConnectionPointer conn);

protected:
    // The connection to the server
    virvo::ConnectionPointer conn_;
    // Current projection matrix
    virvo::mat4 proj_;
    // Current modelview matrix
    virvo::mat4 view_;
};

#endif
