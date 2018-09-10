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

#ifndef _VVGLTOOLS_H_
#define _VVGLTOOLS_H_

#include "math/math.h"

#include "vvexport.h"

//============================================================================
// Class Definitions
//============================================================================

/** Collection of OpenGL raleted tools.
    Consists of static helper functions which are project independent.
    @author Juergen Schulze-Doebold
*/
class VIRVOEXPORT vvGLTools
{
  public:
    enum DisplayStyle                             /// string display style for extensions display
    {
      CONSECUTIVE = 0,                            ///< using entire line length
      ONE_BY_ONE  = 1                             ///< one extension per line
    };
    struct GLInfo
    {
      const char* vendor;
      const char* renderer;
      const char* version;
    };

    static bool enableGLErrorBacktrace(bool printBacktrace = true, bool abortOnError = false);
    static void printGLError(const char*);
    static GLInfo getGLInfo();
    static bool isGLVersionSupported(int major, int minor, int release);
    static bool isGLextensionSupported(const char*);
    static void displayOpenGLextensions(const DisplayStyle);
    static void checkOpenGLextensions();
    static void drawQuad(float x1 = -1.0f, float y1 = -1.0f, float x2 =  1.0f, float y2 =  1.0f);
    static virvo::vec4 queryClearColor();
};

namespace virvo
{
namespace gltools
{
std::string lastError(const std::string& file, int line);
}
}


#define VV_GLERROR virvo::gltools::lastError(__FILE__, __LINE__)

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
