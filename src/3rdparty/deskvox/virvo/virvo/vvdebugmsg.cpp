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

#include "vvdebugmsg.h"

#include "private/vvlog.h"

const char* vvDebugMsg::DEBUG_TEXT = "###Debug message: ";

void vvDebugMsg::setDebugLevel(LevelType level)
{
  virvo::logging::setLevel(static_cast<int>(level));
}

void vvDebugMsg::setDebugLevel(int level)
{
  virvo::logging::setLevel(level);
}

vvDebugMsg::LevelType vvDebugMsg::getDebugLevel()
{
  return static_cast<vvDebugMsg::LevelType>(virvo::logging::getLevel());
}

void vvDebugMsg::msg(int level, const char* text, bool perr)
{
  if (!virvo::logging::isActive(level))
  {
    return;
  }

  if (perr)
  {
    VV_LOG_ERROR() << DEBUG_TEXT << text;
  }
  else
  {
    VV_LOG(level) << DEBUG_TEXT << text;
  }
}

void vvDebugMsg::msg(int level, const char* text, int number)
{
  VV_LOG(level) << DEBUG_TEXT << text << number;
}

void vvDebugMsg::msg(int level, const char* text, int n1, int n2)
{
  VV_LOG(level) << DEBUG_TEXT << text << n1 << ", " << n2;
}

void vvDebugMsg::msg(int level, const char* text, int n1, int n2, int n3)
{
  VV_LOG(level) << DEBUG_TEXT << text << n1 << ", " << n2 << ", " << n3;
}

void vvDebugMsg::msg(int level, const char* text, int n1, int n2, int n3, int n4)
{
  VV_LOG(level) << DEBUG_TEXT << text << n1 << ", " << n2 << ", " << n3 << ", " << n4;
}

void vvDebugMsg::msg(int level, const char* text, float number)
{
  VV_LOG(level) << DEBUG_TEXT << text << number;
}

void vvDebugMsg::msg(int level, const char* text, float n1, float n2)
{
  VV_LOG(level) << DEBUG_TEXT << text << n1 << ", " << n2;
}

void vvDebugMsg::msg(int level, const char* text, float n1, float n2, float n3)
{
  VV_LOG(level) << DEBUG_TEXT << text << n1 << ", " << n2 << ", " << n3;
}

void vvDebugMsg::msg(int level, const char* text, float n1, float n2, float n3, float n4)
{
  VV_LOG(level) << DEBUG_TEXT << text << n1 << ", " << n2 << ", " << n3 << ", " << n4;
}

void vvDebugMsg::msg(int level, const char* text, const char* str)
{
  VV_LOG(level) << DEBUG_TEXT << text << str;
}

bool vvDebugMsg::isActive(int level)
{
  return virvo::logging::isActive(level) != 0;
}

//****************************************************************************
// End of File
//****************************************************************************
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
