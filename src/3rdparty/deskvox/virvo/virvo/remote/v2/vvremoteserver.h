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

#ifndef VV_REMOTE_SERVER_H
#define VV_REMOTE_SERVER_H

#include "math/forward.h"
#include "private/connection.h"
#include "vvcompiler.h"

namespace virvo
{
    class WorkQueue;
}

class vvRenderer;

class vvRemoteServer
{
public:
    typedef virvo::ConnectionPointer ConnectionPointer;
    typedef virvo::MessagePointer MessagePointer;

public:
    // Constructor.
    VVAPI vvRemoteServer();

    // Destructor.
    VVAPI virtual ~vvRemoteServer();

    // Resize the frame buffer or ...
    VVAPI virtual void resize(int w, int h);

    // Sends the image to the client
    virtual void renderImage(ConnectionPointer conn, MessagePointer message,
        virvo::mat4 const& pr, virvo::mat4 const& mv, vvRenderer* renderer, virvo::WorkQueue& queue) = 0;
};

#endif
