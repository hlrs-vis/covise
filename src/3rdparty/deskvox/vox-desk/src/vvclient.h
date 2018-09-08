// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
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

#ifndef VV_CLIENT_H
#define VV_CLIENT_H

#ifdef WIN32
  #include <atlbase.h>
  #ifdef HAVE_SOAP  
    #import <msxml4.dll>
    #import <mssoap30.dll> named_guids \
      exclude("IStream", "IErrorInfo", "ISequentialStream", "_LARGE_INTEGER", \
      "_ULARGE_INTEGER", "tagSTATSTG", "_FILETIME")
  #endif
#endif

namespace vox 
{

/** Class providing communication routines to request and download SOAP/XML data
  sets from a server.
*/
class vvClient
{
  public:
    void test();
    unsigned char* getRegion(int, int, int, int, int, int, int, int, const char*, const char*);
    unsigned char* getRegionHighLevel(int, int, int, int, int, int, int, int, const char*, const char*);
    unsigned char* getRegionLowLevel(int, int, int, int, int, int, int, int, const char*, const char*);
};

}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
