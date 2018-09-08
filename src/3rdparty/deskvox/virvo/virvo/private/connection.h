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

#ifndef VV_PRIVATE_CONNECTION_H
#define VV_PRIVATE_CONNECTION_H

// Boost.ASIO needs _WIN32_WINNT
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501 // Require Windows XP or later
#endif
#endif

#include "vvmessage.h"

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>

#include <boost/function.hpp>

#include <boost/signals2/connection.hpp>
#include <boost/signals2/signal.hpp>

#include <boost/smart_ptr/enable_shared_from_this.hpp>
#include <boost/smart_ptr/weak_ptr.hpp>

#include <list>

namespace virvo
{

class ConnectionManager;

class Connection
    : public boost::enable_shared_from_this<Connection>
{
    friend class ConnectionManager;

public:
    enum Reason { Read, Write };

    typedef boost::signals2::signal<void (Reason reason, MessagePointer message, boost::system::error_code const& e)> SignalType;

public:
    // Constructor.
    VVAPI Connection(ConnectionManager& manager);

    // Destructor.
    VVAPI ~Connection();

    // Start reading from the socket
    VVAPI void start();

    // Stop/Close the connection
    VVAPI void stop();

    // Sets the handler for this connection
    // Thread-safe.
    VVAPI void set_handler(SignalType::slot_function_type handler);

    // Removes the handler for this connection.
    // Thread-safe.
    VVAPI void remove_handler();

    // Close the connection
    VVAPI void close();

    // Sends a message to the other side.
    VVAPI void write(MessagePointer message);

private:
    // The manager for this connection
    ConnectionManager& manager_;
    // The underlying socket.
    boost::asio::ip::tcp::socket socket_;
    // Signal (called from ConnectionManager if anything happens)
    SignalType signal_;
    // Slot
    boost::signals2::connection slot_;
};

typedef boost::shared_ptr<Connection> ConnectionPointer;

} // namespace virvo

#endif // !VV_PRIVATE_CONNECTION_H
