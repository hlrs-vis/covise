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

#include "vvbsptree.h"
#include "vvbsptreevisitors.h"
#include "vvdebugmsg.h"
#include "vvserbrickrend.h"
#include "vvvoldesc.h"


vvSerBrickRend::vvSerBrickRend(vvVolDesc *vd, vvRenderState renderState, size_t numBricks,
                               const std::string& type, const vvRendererFactory::Options& options)
  : vvBrickRend(vd, renderState, numBricks, type, options)
{
  vvDebugMsg::msg(1, "vvSerBrickRend::vvSerBrickRend()");

  ErrorType err = createRenderers();
  if (err != VV_OK)
  {
    vvDebugMsg::msg(0, "vvSerBrickRend::vvSerBrickRend(): Error creating renderers");
    return;
  }

  _simpleRenderVisitor = new vvSimpleRenderVisitor(_renderers);
}

vvSerBrickRend::~vvSerBrickRend()
{
  vvDebugMsg::msg(1, "vvSerBrickRend::~vvSerBrickRend()");

  delete _simpleRenderVisitor;
}

void vvSerBrickRend::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvSerBrickRend::renderVolumeGL()");

  if (!_showBricks)
  {
    _bspTree->setVisitor(_simpleRenderVisitor);

    // find eye position:
    virvo::vec3f eye = getEyePosition();

    // bsp tree maintains boxes in voxel coordinates
    _bspTree->traverse(vd->voxelCoords(eye));
  }
  else
  {
    _bspTree->setVisitor(NULL);
  }

  vvBrickRend::renderVolumeGL();
}

void vvSerBrickRend::setParameter(ParameterType param, const vvParam& newValue)
{
  vvDebugMsg::msg(3, "vvSerBrickRend::setParameter()");

  for (std::vector<vvRenderer*>::iterator it = _renderers.begin();
       it != _renderers.end(); ++it)
  {
    (*it)->setParameter(param, newValue);
  }

  vvBrickRend::setParameter(param, newValue);
}

vvParam vvSerBrickRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvSerBrickRend::getParameter()");

  for (std::vector<vvRenderer*>::const_iterator it = _renderers.begin();
       it != _renderers.end(); ++it)
  {
    return (*it)->getParameter(param);
  }

  return vvBrickRend::getParameter(param);
}

void vvSerBrickRend::updateTransferFunction()
{
  vvDebugMsg::msg(3, "vvSerBrickRend::updateTransferFunction()");

  for (std::vector<vvRenderer*>::iterator it = _renderers.begin();
       it != _renderers.end(); ++it)
  {
    (*it)->updateTransferFunction();
  }
}

vvSerBrickRend::ErrorType vvSerBrickRend::createRenderers()
{
  vvDebugMsg::msg(3, "vvSerBrickRend::createRenderers()");

  // start out with empty rendering regions
  virvo::basic_aabb< ssize_t > emptyBox(virvo::vector< 3, ssize_t >(ssize_t(0)), virvo::vector< 3, ssize_t >(ssize_t(0)));
  setParameter(vvRenderer::VV_VISIBLE_REGION, emptyBox);
  setParameter(vvRenderer::VV_PADDING_REGION, emptyBox);

  for (std::vector<vvRenderer*>::const_iterator it = _renderers.begin();
       it != _renderers.end(); ++it)
  {
    delete *it;
  }
  _renderers.clear();

  for (uint i = 0; i < _bspTree->getLeafs().size(); ++i)
  {
    _renderers.push_back(vvRendererFactory::create(vd, *this, _type.c_str(), _options));
  }

  for (size_t i = 0; i < _renderers.size(); ++i)
  {
    vvBspTree::box_type aabb = _bspTree->getLeafs().at(i)->getAabb();

    setVisibleRegion(_renderers.at(i), aabb);
  }

  return VV_OK;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
