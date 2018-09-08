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

#ifndef VV_DEBUGMSG_H
#define VV_DEBUGMSG_H

#include "vvexport.h"

/** Manager for run-time debug messages.
    Allows the programmer to output debug messages only when required.

    Initially, no messages are printed (level 0).
    Can be overridden by setting the environment variable VV_DEBUG to an integer value.
    @author Jurgen P. Schulze (jschulze@ucsd.edu)
*/
class VIRVO_FILEIOEXPORT vvDebugMsg
{
  public:
    enum LevelType                                /// Valid level types
    {
      NO_MESSAGES   =  0,                         ///< no messages are printed
      FEW_MESSAGES  =  1,                         ///< only the most important messages are printed
      MOST_MESSAGES =  2,                         ///< all other messages which don't appear frequently are also printed
      ALL_MESSAGES  =  3                          ///< also messages which appear frequently are printed
    };

  private:
    static const char* DEBUG_TEXT;                ///< string to be printed at debug message

  public:
    static void setDebugLevel(LevelType level);
    static void setDebugLevel(int level);
    static LevelType getDebugLevel();
    static void msg(int level, const char* text, bool perr = false);
    static void msg(int level, const char* text, int number);
    static void msg(int level, const char* text, int n1, int n2);
    static void msg(int level, const char* text, int n1, int n2, int n3);
    static void msg(int level, const char* text, int n1, int n2, int n3, int n4);
    static void msg(int level, const char* text, float number);
    static void msg(int level, const char* text, float n1, float n2);
    static void msg(int level, const char* text, float n1, float n2, float n3);
    static void msg(int level, const char* text, float n1, float n2, float n3, float n4);
    static void msg(int level, const char* text, const char* str);
    static bool isActive(int level);
};
#endif

//****************************************************************************
// End of File
//****************************************************************************
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
