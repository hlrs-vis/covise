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

#ifndef VV_SOCKETMAP_H
#define VV_SOCKETMAP_H

#include "vvexport.h"

class vvSocket;

namespace vvSocketMap
{
  /*! Store the socket and return an index that refers to it
   */
  VIRVOEXPORT int add(vvSocket* sock);

  VIRVOEXPORT void remove(int idx);

  VIRVOEXPORT vvSocket* get(int idx);

  /*! Get index to refer to socket or -1 if sock was not found
   */
  VIRVOEXPORT int getIndex(vvSocket* sock);
}

#endif

