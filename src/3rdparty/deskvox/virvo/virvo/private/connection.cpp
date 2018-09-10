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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef DESKVOX_USE_ASIO

#include "connection.h"
#include "connection_manager.h"

#include <boost/asio/buffer.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

using virvo::Connection;
using virvo::ConnectionManager;
using virvo::MessagePointer;

using boost::asio::ip::tcp;

//--------------------------------------------------------------------------------------------------
// Connection
//--------------------------------------------------------------------------------------------------

Connection::Connection(ConnectionManager& manager)
    : manager_(manager)
    , socket_(manager.io_service_)
{
}

Connection::~Connection()
{
#if 0 // NEIN!!!
    close();
#endif
}

void Connection::start()
{
}

void Connection::stop()
{
}

void Connection::set_handler(SignalType::slot_function_type handler)
{
    // Remove existing handler.
    // Only a single handler is currently supported.
    remove_handler();

    slot_ = signal_.connect(handler);
}

void Connection::remove_handler()
{
    signal_.disconnect(slot_);
}

void Connection::close()
{
    manager_.close(shared_from_this());
}

void Connection::write(MessagePointer message)
{
    manager_.write(message, shared_from_this());
}

#endif // DESKVOX_USE_ASIO
