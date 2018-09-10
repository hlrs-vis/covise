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

#ifndef VV_COLOR_H
#define VV_COLOR_H

#include "vvexport.h"

/** Creates a general color class for RGB colors.
 */
class VIRVO_FILEIOEXPORT vvColor
{
  public:
    vvColor();
    vvColor(float, float, float);
    float operator[](const int) const;
    float& operator[](const int);
    vvColor operator+(vvColor) const;
    void setRGB(float, float, float);
    void setHSB(float, float, float);
    void getRGB(float&, float&, float&);
    void getHSB(float&, float&, float&);
  private:
    float e[3];

  public:
    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & e[0];
      a & e[1];
      a & e[2];
    }
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
