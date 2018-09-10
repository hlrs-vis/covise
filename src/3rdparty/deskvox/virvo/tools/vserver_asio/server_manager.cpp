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

#include "server_manager.h"
#include "resource_manager.h"
#include "simple_server.h"

#include <cstdio>

vvServerManager::vvServerManager(unsigned short port, bool useBonjour)
    : manager_(virvo::makeConnectionManager(port))
    , serverMode_(SERVER)
    , useBonjour_(useBonjour)
{
}

vvServerManager::~vvServerManager()
{
}

void vvServerManager::accept()
{
    manager_->accept(boost::bind(&vvServerManager::handle_new_connection, this, _1, _2));
}

void vvServerManager::run()
{
    manager_->run();
}

void vvServerManager::stop()
{
    manager_->stop();
}

bool vvServerManager::handle_new_connection(virvo::ConnectionPointer conn, boost::system::error_code const& e)
{
    if (e)
    {
#ifndef NDEBUG
        printf("vserver: error: %s\n", e.message().c_str());
#endif
        return false;
    }

    // Create a new server...
    boost::shared_ptr<vvServer> server;

    if (serverMode_ == SERVER)
        server = boost::make_shared<vvSimpleServer>(conn);
    else
        server = boost::make_shared<vvResourceManager>(conn);

    // Add it to the list of active servers...
    serverList_.push_back(server);

    // Start a new accept operation...
    accept();

    return true;
}
