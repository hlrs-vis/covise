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

#ifndef VV_OPENGL_H
#define VV_OPENGL_H

#include "vvexport.h"

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__)
#define GL_GLEXT_PROTOTYPES 1
#ifndef GL_GLEXT_LEGACY
# define GL_GLEXT_LEGACY 1
#endif
#endif

#include "vvplatform.h"

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#ifndef _WIN32
#ifndef GLX_GLEXT_LEGACY
# define GLX_GLEXT_LEGACY 1
#endif
#include <GL/glext.h>
#include <GL/glx.h>
#endif
#endif

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
