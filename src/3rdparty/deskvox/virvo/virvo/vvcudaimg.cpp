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

#include <cassert>
#include "vvdebugmsg.h"
#include "vvcudaimg.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#ifdef HAVE_CUDA
#include "cuda/utils.h"
#endif

vvCudaImg::vvCudaImg(const int w, const int h, const Mode mode)
  : vvSoftImg(w, h)
{
  _mode = mode;
  _mapped = false;
#ifdef HAVE_CUDA
  _imgRes = NULL;
#endif
  init();
  allocate();
}

vvCudaImg::~vvCudaImg()
{
  deallocate();
}

void vvCudaImg::setSize(int w, int h)
{
  deallocate();
  usePbo = (_mode==TEXTURE);
  vvSoftImg::setSize(w, h);
  allocate();
}

void vvCudaImg::setMapped(const bool mapped)
{
  _mapped = mapped;
}

vvCudaImg::Mode vvCudaImg::getMode() const
{
  return _mode;
}

void vvCudaImg::allocate()
{
#ifdef HAVE_CUDA
  bool ok = true;
  if (_mode==TEXTURE)
  {
    if ((width > 0) && (height > 0))
    {
      virvo::cuda::checkError(&ok, cudaGraphicsGLRegisterBuffer(&_imgRes, getPboName(), cudaGraphicsMapFlagsWriteDiscard),
                         "vvCudaImg::allocate() - map PBO to CUDA");
    }
  }
  else if (_mapped)
  {
    virvo::cuda::checkError(&ok, cudaHostAlloc(&h_img, width*height*vvSoftImg::PIXEL_SIZE, cudaHostAllocMapped),
                       "vvCudaImg::allocate() - img alloc");
    setBuffer(h_img);
    setSize(width, height);
    virvo::cuda::checkError(&ok, cudaHostGetDevicePointer(&d_img, h_img, 0), "get dev ptr img");
  }
  else
  {
    virvo::cuda::checkError(&ok, cudaMalloc(&d_img, width*height*vvSoftImg::PIXEL_SIZE),
                       "vvCudaImg::allocate() - cudaMalloc img");
  }
#else
  vvDebugMsg::msg(1, "HAVE_CUDA undefined");
#endif
}

void vvCudaImg::deallocate()
{
#ifdef HAVE_CUDA
  bool ok = true;
  if (_mode==TEXTURE)
  {
    if (_imgRes != NULL)
    {
      virvo::cuda::checkError(&ok, cudaGraphicsUnregisterResource(_imgRes),
                         "vvCudaImg::deallocate() - cudaGraphicsUnregisterResource");
    }
  }
  else if (_mapped)
  {
    virvo::cuda::checkError(&ok, cudaFreeHost(h_img),
                       "vvCudaImg::deallocate() - cudaFreeHost");
  }
  else
  {
    virvo::cuda::checkError(&ok, cudaFree(d_img),
                       "vvCudaImg::deallocate() - cudaFree");
  }
#else
  vvDebugMsg::msg(1, "HAVE_CUDA undefined");
#endif
}

void vvCudaImg::map()
{
#ifdef HAVE_CUDA
  bool ok = true;
  if (_mode==TEXTURE)
  {
    virvo::cuda::checkError(&ok, cudaGraphicsMapResources(1, &_imgRes, NULL),
                       "vvCudaImg::map() - map CUDA resource");
    size_t size;
    virvo::cuda::checkError(&ok, cudaGraphicsResourceGetMappedPointer((void**)&d_img, &size, _imgRes),
                       "vvCudaImg::map() - get PBO mapping");
    assert(size == static_cast<size_t>(width*height*vvSoftImg::PIXEL_SIZE));
  }
  else
  {
    clear();
  }
#else
  vvDebugMsg::msg(1, "HAVE_CUDA undefined");
#endif
}

void vvCudaImg::unmap()
{
#ifdef HAVE_CUDA
  bool ok = true;
  if (_mode==TEXTURE)
  {
    virvo::cuda::checkError(&ok, cudaGraphicsUnmapResources(1, &_imgRes, NULL),
                       "vvCudaImg::unmap() - unmap CUDA resource");
  }
  else if (_mapped)
  {
    virvo::cuda::checkError(&ok, cudaThreadSynchronize(),
                       "vvCudaImg::unmap() - cudaThreadSynchronize");
  }
  else
  {
    virvo::cuda::checkError(&ok, cudaMemcpy(data, d_img, width*height*vvSoftImg::PIXEL_SIZE, cudaMemcpyDeviceToHost),
                       "vvCudaImg::unmap() - cpy to host");
  }
#else
 vvDebugMsg::msg(1, "HAVE_CUDA undefined");
#endif
}

#ifdef HAVE_CUDA
uchar4* vvCudaImg::getDeviceImg() const
{
  return d_img;
}
#endif

uchar* vvCudaImg::getHostImg() const
{
  return h_img;
}

void vvCudaImg::init()
{
  if (_mode==TEXTURE)
  {
#ifdef HAVE_CUDA
    if (virvo::cuda::initGlInterop())
    {
      vvDebugMsg::msg(1, "using CUDA/GL interop");
      // avoid image copy from GPU to CPU and back
      setSize(width, height);
    }
    else
#else
      vvDebugMsg::msg(1, "HAVE_CUDA undefined");
#endif
    {
      vvDebugMsg::msg(1, "can't use CUDA/GL interop");
      _mode = SW_FALLBACK;
      setSize(width, height);
    }
  }
  else
  {
    vvDebugMsg::msg(1, "set img size");
    _mode = SW_FALLBACK;
    setSize(width, height);
  }
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
