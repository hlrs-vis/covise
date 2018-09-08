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

#ifndef VV_PRIVATE_CONNECTION_MANAGER_H
#define VV_PRIVATE_CONNECTION_MANAGER_H

#include "connection.h"

#include <boost/function.hpp>

#include <boost/thread.hpp>

#include <deque>
#include <vector>
#include <map>
#include <set>
#include <list>

namespace virvo
{

class Connection;
class ConnectionManager;

typedef boost::shared_ptr<ConnectionManager> ConnectionManagerPointer;

class ConnectionManager
    : public boost::enable_shared_from_this<ConnectionManager>
{
    friend class Connection;

public:
    typedef boost::function<bool(ConnectionPointer conn, boost::system::error_code const& e)> Handler;

public:
    // Constructor.
    VVAPI ConnectionManager();

    // Constructor.
    VVAPI ConnectionManager(unsigned short port);

    // Destructor.
    VVAPI ~ConnectionManager();

    // Bind the acceptor to the given port
    VVAPI void bind_port(unsigned short port);

    // Starts the message loop
    VVAPI void run();

    // Starts a new thread which in turn starts the message loop
    VVAPI void run_in_thread();

    // Wait for the thread to finish
    VVAPI void wait();

    // Stops the message loop
    VVAPI void stop();

    // Starts a new accept operation.
    // Use bind_port() to specifiy the port.
    VVAPI void accept(Handler handler);

    // Starts a new connect operation
    VVAPI void connect(std::string const& host, unsigned short port, Handler handler);

    // Starts a new connect operation and waits until the connection is connected
    VVAPI ConnectionPointer connect(std::string const& host, unsigned short port);

    // Returns an existing connection or creates a new one
    VVAPI ConnectionPointer get_or_connect(std::string const& host, unsigned short port);

    // Close the given connection
    VVAPI void close(ConnectionPointer conn);

    // Search for an existing connection
    VVAPI ConnectionPointer find(std::string const& host, unsigned short port);

private:
    // Start an accept operation
    void do_accept(Handler handler);

    // Handle completion of a accept operation.
    void handle_accept(boost::system::error_code const& e, ConnectionPointer conn, Handler handler);

    // Starts a new connect operation.
    void do_connect(std::string const& host, unsigned short port, Handler handler);

    // Handle completion of a connect operation.
    void handle_connect(boost::system::error_code const& e, ConnectionPointer conn, Handler handler);

    // Read the next message from the given client.
    void do_read(ConnectionPointer conn);

    // Called when a message header is read.
    void handle_read_header(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn);

    // Called when a complete message is read.
    void handle_read_data(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn);

    // Sends a message to the given client
    void write(MessagePointer msg, ConnectionPointer conn);

    // Starts a new write operation.
    void do_write(MessagePointer msg, ConnectionPointer conn);

    // Write the next message
    void do_write();

    // Called when a complete message is written.
    void handle_write(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn);

    // Add a new connection
    void add_connection(ConnectionPointer conn);

    // Remove an existing connection
    void remove_connection(ConnectionPointer conn);

private:
    typedef std::set<ConnectionPointer> Connections;
    typedef std::deque<std::pair<ConnectionPointer, MessagePointer> > Messages;

    // The IO service
    boost::asio::io_service io_service_;
    // The acceptor object used to accept incoming socket connections.
    boost::asio::ip::tcp::acceptor acceptor_;
    // To protect the list of messages...
    boost::asio::io_service::strand strand_;
    // To keep the io_service running...
    boost::shared_ptr<boost::asio::io_service::work> work_;
    // The list of active connections
    Connections connections_;
    // List of messages to be written
    Messages write_queue_;
    // A thread to process the message queue
    boost::thread runner_;
};

inline ConnectionManagerPointer makeConnectionManager()
{
    return boost::make_shared<ConnectionManager>();
}

inline ConnectionManagerPointer makeConnectionManager(unsigned short port)
{
    return boost::make_shared<ConnectionManager>(port);
}

} // namespace virvo

#endif // !VV_PRIVATE_CONNECTION_H
