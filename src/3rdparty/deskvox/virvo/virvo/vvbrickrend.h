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

#ifndef VV_BRICKREND_H
#define VV_BRICKREND_H

#include "math/forward.h"

#include "vvrenderer.h"
#include "vvrendererfactory.h"
#include "vvvoldesc.h"

class vvBspTree;
class vvShowBricksVisitor;
class vvVolDesc;

#include <vector>

class VIRVOEXPORT vvBrickRend : public vvRenderer
{
public:
  enum ErrorType
  {
    VV_OK = 0
  };

  /*! @param type     passed to renderer factory to create one renderer per brick
      @param options  passed to renderer factory to create one renderer per brick
   */
  vvBrickRend(vvVolDesc* vd, vvRenderState renderState, size_t numBricks,
              const std::string& type, const vvRendererFactory::Options& options);
  virtual ~vvBrickRend();

  /*! render the brick outlines if "showbricks" option is set
   */
  virtual void renderVolumeGL();

  virtual void setParameter(ParameterType param, const vvParam& newValue);
  virtual vvParam getParameter(ParameterType param) const;
  virtual void updateTransferFunction();
protected:
  vvBspTree* _bspTree;
  vvShowBricksVisitor* _showBricksVisitor;

  std::string _type;
  vvRendererFactory::Options _options;

  /*! build a bsp tree with numBricks leafs
   */
  bool buildBspTree(size_t numBricks);

  /*! update visibible region of renderer with border padding for interpolation
   */
  static void setVisibleRegion(vvRenderer* renderer, virvo::basic_aabb< ssize_t > const& box, size_t padding = 1);
};

#endif // VV_BRICKREND_H
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
