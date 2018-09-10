// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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

#ifndef VV_CUDAIMG_H
#define VV_CUDAIMG_H

#include "vvsoftimg.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include "cuda/utils.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#endif

#include "vvconfig.h"

class VIRVOEXPORT vvCudaImg : public vvSoftImg
{
public:
  enum Mode
  {
    TEXTURE,
    SW_FALLBACK
  };

  vvCudaImg(int w, int h, Mode mode = TEXTURE);
  ~vvCudaImg();
  void setSize(int w, int h);

  void setMapped(bool mapped);

  Mode getMode() const;

  void allocate();
  void deallocate();
  void map();
  void unmap();

#ifdef HAVE_CUDA
  uchar4* getDeviceImg() const;
#endif
  uchar* getHostImg() const;
private:
  bool _mapped;
  Mode _mode;

#ifdef HAVE_CUDA
  cudaGraphicsResource* _imgRes;            ///< CUDA resource mapped to PBO
  uchar4* d_img;
#endif
  uchar* h_img;

  void init();
};

#endif // VV_CUDAIMG_H

//============================================================================
// End of File
//============================================================================

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
