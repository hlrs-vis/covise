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

#ifndef VSERVER_SERVER_H
#define VSERVER_SERVER_H

#include "server_manager.h"

class vvServer
{
public:
    // Constructor.
    vvServer(virvo::ConnectionPointer conn);

    // Destructor.
    virtual ~vvServer();

    // Caller when something interesting happens
    void handler(virvo::Connection::Reason reason, virvo::MessagePointer message, boost::system::error_code const& e);

    // Called when a new message has successfully been read from the server.
    virtual void on_read(virvo::MessagePointer message);

    // Called when a message has successfully been written to the server.
    virtual void on_write(virvo::MessagePointer message);

    // Returns the connection
    virvo::ConnectionPointer conn() const {
        return conn_;
    }

private:
    // The connection to the client
    virvo::ConnectionPointer conn_;
};

#endif
