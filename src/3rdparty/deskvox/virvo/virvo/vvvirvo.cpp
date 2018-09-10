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


#include "vvvirvo.h"

#include <algorithm>
#include <locale>
#include <string>

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#define VV_STRINGIFY(X) VV_STRINGIFY2(X)
#define VV_STRINGIFY2(X) #X

const char* virvo::version()
{
  return VV_STRINGIFY(VV_VERSION_MAJOR) "." VV_STRINGIFY(VV_VERSION_MINOR);
}

bool virvo::hasFeature(const char* name)
{
#ifndef HAVE_CONFIG_H
  return false;
#endif

  std::string str = name;
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if (str == "bonjour")
  {
#ifdef HAVE_BONJOUR
    return true;
#else
    return false;
#endif
  }
  else if (str == "cg")
  {
#ifdef HAVE_CG
    return true;
#else
    return false;
#endif
  }
  else if (str == "cuda")
  {
#ifdef HAVE_CUDA
    return true;
#else
    return false;
#endif
  }
  else if (str == "ffmpeg")
  {
#ifdef HAVE_FFMEPG
    return true;
#else
    return false;
#endif
  }
  else if (str == "gl" || str == "opengl")
  {
#ifdef HAVE_GL
    return true;
#else
    return false;
#endif
  }
  else if (str == "glu")
  {
#ifdef HAVE_GLU
    return true;
#else
    return false;
#endif
  }
  else if (str == "snappy")
  {
#ifdef HAVE_SNAPPY
    return true;
#else
    return false;
#endif
  }
  else if (str == "volpack")
  {
#ifdef HAVE_VOLPACK
    return true;
#else
    return false;
#endif
  }
  else if (str == "x11")
  {
#ifdef HAVE_X11
    return true;
#else
    return false;
#endif
  }
  else
  {
    return false;
  }
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
