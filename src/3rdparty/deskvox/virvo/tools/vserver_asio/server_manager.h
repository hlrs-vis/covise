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

#ifndef VSERVER_SERVER_MANAGER_H
#define VSERVER_SERVER_MANAGER_H

#include <virvo/private/connection_manager.h>

#include <vector>

class vvServer;

class vvServerManager
{
public:
    enum Mode {
        SERVER,
        RM,
        RM_WITH_SERVER
    };

    // Constructor.
    vvServerManager(unsigned short port, bool useBonjour);

    // Destructor.
    virtual ~vvServerManager();

    // Starts a new accept operation
    void accept();

    // Runs the server
    void run();

    // Stops the server
    void stop();

private:
    // Called when a new connection is accepted.
    // Return false to discard the new connection, return conn->accept(...)
    // to finally accept the new connection.
    bool handle_new_connection(virvo::ConnectionPointer conn, boost::system::error_code const& e);

private:
    // The server manager
    boost::shared_ptr<virvo::ConnectionManager> manager_;
    // List of servers
    std::vector<boost::shared_ptr<vvServer> > serverList_;
    // indicating current server mode (default: single server)
    Mode serverMode_;
    // indicating the use of bonjour
    bool useBonjour_;
};

#endif
