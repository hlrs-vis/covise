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

#ifndef VSERVER_RESOURCE_MANAGER_H
#define VSERVER_RESOURCE_MANAGER_H

#include "server.h"

class vvResourceManager : public vvServer
{
    typedef vvServer BaseType;

public:
    typedef virvo::ConnectionPointer ConnectionPointer;
    typedef virvo::MessagePointer MessagePointer;

    // Constructor.
    vvResourceManager(ConnectionPointer conn);

    // Called when a new message has successfully been read from the server.
    virtual void on_read(MessagePointer message);

    // Called when a message has successfully been written to the server.
    virtual void on_write(MessagePointer message);
};

#endif
