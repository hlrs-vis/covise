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

#ifndef VV_PRIVATE_MESSAGES_H
#define VV_PRIVATE_MESSAGES_H

#include "../math/math.h"
#include "../vvparam.h"
#include "../vvrenderer.h"

#include <istream>
#include <ostream>

namespace virvo
{
namespace messages
{

struct CameraMatrix
{
    // The model-view matrix
    mat4 view;
    // The projection matrix
    mat4 proj;

    CameraMatrix()
    {
    }

    CameraMatrix(mat4 const& view, mat4 const& proj)
        : view(view)
        , proj(proj)
    {
    }

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
        a & view;
        a & proj;
    }
};

struct Param
{
    // The name of the parameter
    vvRenderState::ParameterType name;
    // The actual parameter
    vvParam value;

    Param()
        : name(static_cast<vvRenderState::ParameterType>(0))
        , value()
    {
    }

    Param(vvRenderState::ParameterType name, vvParam const& value)
        : name(name)
        , value(value)
    {
    }

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
        a & name;
        a & value;
    }
};

struct WindowResize
{
    int w;
    int h;

    WindowResize()
    {
    }

    WindowResize(int w, int h)
        : w(w)
        , h(h)
    {
    }

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
        a & w;
        a & h;
    }
};

} // namespace messages
} // namespace virvo

#endif
