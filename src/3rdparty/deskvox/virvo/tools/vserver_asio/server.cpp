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

#include "server.h"

using virvo::Connection;
using virvo::ConnectionPointer;
using virvo::MessagePointer;

vvServer::vvServer(ConnectionPointer conn)
    : conn_(conn)
{
    conn->set_handler(boost::bind(&vvServer::handler, this, _1, _2, _3));
}

vvServer::~vvServer()
{
}

void vvServer::handler(Connection::Reason reason, MessagePointer message, boost::system::error_code const& e)
{
    if (e)
        return;

    switch (reason)
    {
    case Connection::Read:
        on_read(message);
        break;
    case Connection::Write:
        on_write(message);
        break;
    }
}

void vvServer::on_read(MessagePointer /*message*/)
{
}

void vvServer::on_write(MessagePointer /*message*/)
{
}
