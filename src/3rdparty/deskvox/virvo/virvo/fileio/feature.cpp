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


#include "feature.h"

#include <algorithm>
#include <locale>
#include <string>

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#define VV_STRINGIFY(X) VV_STRINGIFY2(X)
#define VV_STRINGIFY2(X) #X

const char* virvo::fileio::version()
{
  return VV_STRINGIFY(VV_VERSION_MAJOR) "." VV_STRINGIFY(VV_VERSION_MINOR);
}

bool virvo::fileio::hasFeature(const char* name)
{
#ifndef HAVE_CONFIG_H
  return false;
#endif

  std::string str = name;
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if (str == "nifti")
  {
#if VV_HAVE_NIFTI
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
