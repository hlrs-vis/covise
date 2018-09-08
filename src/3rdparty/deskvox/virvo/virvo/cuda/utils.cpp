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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#ifdef HAVE_CUDA

#include "utils.h"

#include <string>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

#ifdef USE_X11
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#include "vvdebugmsg.h"
#include "private/vvlog.h"

bool virvo::cuda::checkError(bool *success, cudaError_t err, const char *msg, bool syncIfDebug)
{
  if (err == cudaSuccess)
  {
    if (!vvDebugMsg::isActive(2) || !syncIfDebug)
      return (success ? *success : true);

    if (syncIfDebug)
    {
     cudaThreadSynchronize();
     err = cudaGetLastError();
    }
  }

  if (!msg)
    msg = "virvo::cuda";

  if (err == cudaSuccess)
  {
    vvDebugMsg::msg(3, msg, ": ok");
    return (success ? *success : true);
  }

  std::string s(msg);
  s += " ";
  s += std::to_string(err);
  s += ": ";
  s += cudaGetErrorString(err);
  vvDebugMsg::msg(0, s.c_str());

  if (success)
  {
    *success = false;
    return *success;
  }

  return false;
}

int virvo::cuda::deviceCount()
{
  bool dummy = true;
  virvo::cuda::checkError(&dummy, cudaGetLastError(), "virvo::cuda::deviceCount: previous error", true);

  int cnt = 0;
  bool ok = true;
  if (!virvo::cuda::checkError(&ok, cudaGetDeviceCount(&cnt), "virvo::cuda::deviceCount", false))
  {
    return 0;
  }
  return cnt;
}

bool virvo::cuda::init()
{
  bool ok = true;
  if(!virvo::cuda::checkError(&ok, cudaSetDeviceFlags(cudaDeviceMapHost), "virvo::cuda::init (set device flags)", false))
    return false;

  return true;
}

bool virvo::cuda::initGlInterop()
{
  static bool done = false;
  if (done)
    return true;

#ifdef USE_X11
  GLXContext ctx = glXGetCurrentContext();
  Display *dsp = XOpenDisplay(NULL);
  if(dsp)
    XSynchronize(dsp, True);
  if(!dsp || !glXIsDirect(dsp, ctx))
  {
    vvDebugMsg::msg(1, "no CUDA/GL interop possible");
    return false;
  }
#endif

  cudaDeviceProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.major = 1;
  prop.minor = 0;

  int dev;
  bool ok = true;
  if(!virvo::cuda::checkError(&ok, cudaChooseDevice(&dev, &prop), "virvo::cuda::initGlInterop (choose device)", false))
    return false;
  if(!virvo::cuda::checkError(&ok, cudaGLSetGLDevice(dev), "virvo::cuda::initGlInterop (set device)", false))
    return false;

  done = true;
  return true;
}


namespace {

bool CheckConfig(cudaFuncAttributes const& attr, cudaDeviceProp const& prop, dim3 const& blockDim, size_t dynamicSharedMem)
{
  // Compute the number of threads per block
  size_t numThreads = blockDim.x * blockDim.y * blockDim.z;

  // Compare blockDim with the maximum block dimension for the current device and the current kernel
  if (numThreads > static_cast<size_t>(attr.maxThreadsPerBlock))
    return false;

  // Check the size of the shared memory
  if (attr.sharedSizeBytes/*static*/ + dynamicSharedMem > prop.sharedMemPerBlock)
    return false;

  return true;
}

} // namespace


bool virvo::cuda::findConfig(cudaFuncAttributes const& attr, dim3 const*& begin, dim3 const* end, size_t dynamicSharedMem)
{
  if (begin != end)
  {
    // The current device
    int device = -1;
    // The device properties for the current device
    cudaDeviceProp prop = cudaDevicePropDontCare;

    if (cudaSuccess == cudaGetDevice(&device) && cudaSuccess == cudaGetDeviceProperties(&prop, device))
    {
      // Test all possible block dimensions...
      for (; begin != end; ++begin)
      {
        if (CheckConfig(attr, prop, *begin, dynamicSharedMem))
          return true;
      }
    }
  }

  return false;
}


#endif // HAVE_CUDA

