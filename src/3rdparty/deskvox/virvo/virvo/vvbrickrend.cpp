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

#include "math/math.h"

#include "vvbrickrend.h"
#include "vvbsptree.h"
#include "vvbsptreevisitors.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"

#include <cmath>

vvBrickRend::vvBrickRend(vvVolDesc *vd, vvRenderState renderState, size_t numBricks,
                         const std::string& type, const vvRendererFactory::Options& options)
  : vvRenderer(vd, renderState)
  , _type(type)
  , _options(options)
{
  vvDebugMsg::msg(1, "vvBrickRend::vvBrickRend()");

  _showBricksVisitor = new vvShowBricksVisitor(vd);

  if (!buildBspTree(numBricks))
  {
    vvDebugMsg::msg(0, "vvbrickrend::vvbrickrend(): error: could not create bsp tree");
  }
}

vvBrickRend::~vvBrickRend()
{
  vvDebugMsg::msg(1, "vvBrickRend::~vvBrickRend()");

  delete _showBricksVisitor;
  delete _bspTree;
}

void vvBrickRend::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvBrickRend::renderVolumeGL()");

  if (_showBricks)
  {
    _bspTree->setVisitor(_showBricksVisitor);
    // no depth ordering necessary, don't calculate eye pos
    virvo::vector< 3, ssize_t > eye(0, 0, 0);
    _bspTree->traverse(eye);
  }
  else
  {
    _bspTree->setVisitor(NULL);
  }

  vvRenderer::renderVolumeGL();
}

void vvBrickRend::setParameter(ParameterType param, const vvParam& newValue)
{
  vvDebugMsg::msg(3, "vvBrickRend::setParameter()");

  vvRenderer::setParameter(param, newValue);
}

vvParam vvBrickRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvBrickRend::setParameter()");

  return vvRenderer::getParameter(param);
}

void vvBrickRend::updateTransferFunction()
{
  vvDebugMsg::msg(3, "vvBrickRend::updateTransferFunction()");

  vvRenderer::updateTransferFunction();
}

bool vvBrickRend::buildBspTree(size_t numBricks)
{
  vvBspData data;
  data.numLeafs = numBricks;

  _bspTree = new vvBspTree(vd->vox, data);

   return (_bspTree != NULL);
}

void vvBrickRend::setVisibleRegion(vvRenderer* renderer, virvo::basic_aabb< ssize_t > const& box, size_t padding)
{
  vvDebugMsg::msg(3, "vvBrickRend::setVisibleRegion()");

  renderer->setParameter(vvRenderer::VV_VISIBLE_REGION, box);

  virvo::vector< 3, ssize_t > minval = box.min;
  virvo::vector< 3, ssize_t > maxval = box.max;
  for (size_t j = 0; j < 3; ++j)
  {
    if (minval[j] > 0)
    {
      minval[j] -= padding;
    }
    
    if (maxval[j] < renderer->getVolDesc()->vox[j])
    {
      maxval[j] += padding;
    }
  }
  renderer->setParameter(vvRenderer::VV_PADDING_REGION, virvo::basic_aabb< ssize_t >(minval, maxval));

}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
