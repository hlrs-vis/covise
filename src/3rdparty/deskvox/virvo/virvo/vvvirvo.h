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

#ifndef VV_VIRVO_H
#define VV_VIRVO_H

#include "vvexport.h"
#include "vvversion.h"

/** \mainpage Virvo
  <DL>
    <DT><B>Functionality</B>         <DD>VIRVO stands for VIrtual Reality VOlume renderer.
                                     It is a library for real-time volume rendering with hardware accelerated texture mapping.
    <DT><B>Developer Information</B> <DD>The library does not depend on other libraries except OpenGL.
                                     If you want to use Nvidia Cg pixel shaders,
                                     you need to define HAVE_CG at compile time.
                                     The main rendering classes are vvRenderer and vvTexRend. You can
                                     create new rendering classes by deriving them from vvRenderer.
                                     Transfer functions for volume rendering are managed by vvTransFunc.
                                     The class vvSocket allows system independent socket communication,
not limited to the transfer of volume data. vvDicom is a pretty good DICOM
image file reader which can be extended to any unknown formats. vvVector3/4
and vvMatrix are components of vvVecmath, a useful library for linear algebra.
<DT><B>Copyright</B>             <DD>(c) 1999-2005 J&uuml;rgen P. Schulze. All rights reserved.
<DT><B>Email</B>                 <DD>jschulze@ucsd.edu
<DT><B>Institution</B>           <DD>Brown University
</DL>
*/

namespace virvo
{
  VIRVOEXPORT char const* version();
  /** \brief  Query if the library was linked against one of the following:
    cg|cuda|gl|glu|volpack|x11|ffmpeg|snappy|volpack
   */
  VIRVOEXPORT bool hasFeature(const char* name);
}

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
