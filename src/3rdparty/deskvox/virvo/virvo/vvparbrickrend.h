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

#ifndef VV_PARBRICKREND_H
#define VV_PARBRICKREND_H

#include "vvbrickrend.h"
#include "vvbsptreevisitors.h"

#include <vector>

class vvRenderContext;
class vvVolDesc;

class VIRVOEXPORT vvParBrickRend : public vvBrickRend
{
public:
  struct Param
  {
    std::string display;
    bool reuseMainContext;
    int sockidx;           ///< reference to a socket in socketmap
    std::string filename;

    Param()
      : display("")
      , reuseMainContext(false)
      , sockidx(-1)
      , filename("")
    {

    }
  };

  vvParBrickRend(vvVolDesc* vd, vvRenderState rs,
                 const std::vector<Param>& params,
                 const std::string& type, const vvRendererFactory::Options& options);
  ~vvParBrickRend();

  void renderVolumeGL();

  void setParameter(ParameterType param, const vvParam& newValue);
  vvParam getParameter(ParameterType param) const;
  void updateTransferFunction();
private:
  struct Thread;

  vvSortLastVisitor* _sortLastVisitor;

  size_t _width;
  size_t _height;

  static void* renderFunc(void* args);
  static void render(Thread* thread);

  Thread* _thread;                                   ///< main thread
  std::vector<Thread*> _threads;                     ///< worker threads
  std::vector<vvSortLastVisitor::Texture> _textures;
};

#endif // VV_PARBRICKREND_H
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
