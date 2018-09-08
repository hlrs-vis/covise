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

#include "gl/util.h"
#include "vvswitchrenderer.h"
#include "vvdebugmsg.h"
#include "vvvecmath.h"

template<class Orthographic, class Perspective>
vvSwitchRenderer<Orthographic, Perspective>::vvSwitchRenderer(vvVolDesc *vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
  , _ortho(NULL)
  , _persp(NULL)
{
  vvDebugMsg::msg(1, "vvSwitchRenderer::vvSwitchRenderer()");
}

template<class Orthographic, class Perspective>
vvSwitchRenderer<Orthographic, Perspective>::~vvSwitchRenderer()
{
  vvDebugMsg::msg(1, "vvSwitchRenderer::~vvSwitchRenderer()");
  delete _ortho;
  delete _persp;
}

template<class Orthographic, class Perspective>
void vvSwitchRenderer<Orthographic, Perspective>::setParameter(ParameterType param, const vvParam& newValue)
{
  vvRenderer::setParameter(param, newValue);
  if(_ortho)
    _ortho->setParameter(param, newValue);
  if(_persp)
    _persp->setParameter(param, newValue);
}

template<class Orthographic, class Perspective>
void vvSwitchRenderer<Orthographic, Perspective>::updateTransferFunction()
{
  vvRenderer::updateTransferFunction();
  if(_ortho)
    _ortho->updateTransferFunction();
  if(_persp)
    _persp->updateTransferFunction();
}

template<class Orthographic, class Perspective>
void vvSwitchRenderer<Orthographic, Perspective>::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvSwitchRenderer::renderVolumeGL()");

  vvRenderer *rend = NULL;

  vvMatrix pm = virvo::gl::getProjectionMatrix();
  if(pm.isProjOrtho())
  {
    // orthographic projection
    if(!_ortho)
      _ortho = new Orthographic(vd, *this);
    rend = _ortho;
  }
  else
  {
    // perspective projection
    if(!_persp)
      _persp = new Perspective(vd, *this);
    rend = _persp;
  }

  if(rend)
    rend->renderVolumeGL();
  else
    vvDebugMsg::msg(1, "no renderer");
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
