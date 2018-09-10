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

#ifndef VV_PRINTGL_H
#define VV_PRINTGL_H

#include "math/math.h"
#include "vvexport.h"

//============================================================================
// Class Definition
//============================================================================

/** This class allows 2D billboard type text to be printed on an OpenGL
    canvas.
    @author Jurgen P. Schulze
*/
class VIRVOEXPORT vvPrintGL
{
  private:
    virvo::vec4f font_color_;

  public:
    vvPrintGL();
    virtual ~vvPrintGL();
    void print(const float, const float, const char *, ...);

    void setFontColor(virvo::vec4f const& fontColor);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
