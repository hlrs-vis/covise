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

#include "connection_manager.h"

#include "private/vvtimer.h"

#include <boost/asio/buffer.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>

#include <boost/bind.hpp>

#include <boost/ref.hpp>

#include <cstdio>

using virvo::MessagePointer;
using virvo::ConnectionPointer;
using virvo::ConnectionManager;

using boost::asio::ip::tcp;

#define TIME_READS 0
#define TIME_WRITES 0

//--------------------------------------------------------------------------------------------------
// Misc.
//--------------------------------------------------------------------------------------------------

template<class T>
inline std::string to_string(T const& x)
{
    std::ostringstream stream;
    stream << x;
    return stream.str();
}

//--------------------------------------------------------------------------------------------------
// ConnectionManager
//--------------------------------------------------------------------------------------------------

ConnectionManager::ConnectionManager()
    : io_service_()
    , acceptor_(io_service_)
    , strand_(io_service_)
    , work_(boost::make_shared<boost::asio::io_service::work>(boost::ref(io_service_)))
    , connections_()
    , write_queue_()
{
}

ConnectionManager::ConnectionManager(unsigned short port)
    : io_service_()
    , acceptor_(io_service_, tcp::endpoint(tcp::v6(), port))
    , strand_(io_service_)
    , work_(boost::make_shared<boost::asio::io_service::work>(boost::ref(io_service_)))
    , connections_()
    , write_queue_()
{
}

ConnectionManager::~ConnectionManager()
{
    try
    {
        stop();
    }
    catch (std::exception& e)
    {
        static_cast<void>(e);
    }
}

void ConnectionManager::bind_port(unsigned short port)
{
    tcp::endpoint endpoint = tcp::endpoint(tcp::v6(), port);

    acceptor_.open(endpoint.protocol());
    acceptor_.bind(endpoint);
}

void ConnectionManager::run()
{
#ifndef NDEBUG
    try
    {
        io_service_.run();
    }
    catch (std::exception& e)
    {
        printf("ConnectionManager::run: EXCEPTION caught: %s", e.what());
        throw;
    }
#else
    io_service_.run();
#endif
}

void ConnectionManager::run_in_thread()
{
    runner_ = boost::thread(&ConnectionManager::run, this);
}

void ConnectionManager::wait()
{
    runner_.join();
}

void ConnectionManager::stop()
{
    work_.reset();

    io_service_.stop();
    io_service_.reset();
}

void ConnectionManager::accept(Handler handler)
{
    // Start an accept operation for a new connection.
    strand_.post(boost::bind(&ConnectionManager::do_accept, this, handler));
}

void ConnectionManager::connect(std::string const& host, unsigned short port, Handler handler)
{
    // Start a new connection operation
    strand_.post(boost::bind(&ConnectionManager::do_connect, this, host, port, handler));
}

ConnectionPointer ConnectionManager::connect(std::string const& host, unsigned short port)
{
    using boost::asio::ip::tcp;

    ConnectionPointer conn(new Connection(*this));

    // Resolve the host name into an IP address.
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator I = resolver.resolve(query);
    tcp::resolver::iterator E;

    boost::system::error_code error_code;

    // Connect
    boost::asio::connect(conn->socket_, I, E, error_code);

    if (error_code)
    {
        return ConnectionPointer();
    }

    // Save the connection
    add_connection(conn);

    return conn;
}

ConnectionPointer ConnectionManager::get_or_connect(std::string const& host, unsigned short port)
{
    // Look for an existing connection
    ConnectionPointer conn = find(host, port);

    if (conn.get() == 0)
    {
        // No connection found.
        // Create a new one.
        conn = connect(host, port);
    }

    return conn;
}

void ConnectionManager::close(ConnectionPointer conn)
{
    Connections::iterator I = connections_.find(conn);

    if (I == connections_.end())
    {
        return;
    }

    // Remove the handler!
    conn->remove_handler();

    // Close the connection
    conn->socket_.shutdown(tcp::socket::shutdown_both);
    conn->socket_.close();

    // Remove from the list.
    // Eventually deletes the socket.
    connections_.erase(I);
}

ConnectionPointer ConnectionManager::find(std::string const& host, unsigned short port)
{
    using boost::asio::ip::tcp;

    // Get the endpoint
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator it = resolver.resolve(query);

    tcp::endpoint endpoint = *it;

    // Check if a connection with the required endpoint exists
    for (Connections::iterator I = connections_.begin(), E = connections_.end(); I != E; ++I)
    {
        tcp::endpoint remoteEndpoint = (*I)->socket_.remote_endpoint();

        if (endpoint == remoteEndpoint)
            return *I; // Found one
    }

    // There is currently no connection to the given endpoint...
    return ConnectionPointer();
}

//--------------------------------------------------------------------------------------------------
// Implementation
//--------------------------------------------------------------------------------------------------

void ConnectionManager::do_accept(Handler handler)
{
    ConnectionPointer conn(new Connection(*this));

    // Start an accept operation for a new connection.
    acceptor_.async_accept(
            conn->socket_,
            boost::bind(&ConnectionManager::handle_accept, this, boost::asio::placeholders::error, conn, handler)
            );
}

void ConnectionManager::handle_accept(boost::system::error_code const& e, ConnectionPointer conn, Handler handler)
{
    bool ok = handler(conn, e);

    if (!e)
    {
        if (ok)
        {
            // Save the connection
            add_connection(conn);
        }
    }
    else
    {
#ifndef NDEBUG
        printf("ConnectionManager::handle_accept: %s", e.message().c_str());
#endif
    }
}

void ConnectionManager::do_connect(std::string const& host, unsigned short port, Handler handler)
{
    ConnectionPointer conn(new Connection(*this));

    // Resolve the host name into an IP address.
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Start an asynchronous connect operation.
    boost::asio::async_connect(
            conn->socket_,
            endpoint_iterator,
            boost::bind(&ConnectionManager::handle_connect, this, boost::asio::placeholders::error, conn, handler)
            );
}

void ConnectionManager::handle_connect(boost::system::error_code const& e, ConnectionPointer conn, Handler handler)
{
    bool ok = handler(conn, e);

    if (!e)
    {
        if (ok)
        {
            // Successfully established connection.
            add_connection(conn);
        }
    }
    else
    {
#ifndef NDEBUG
        printf("ConnectionManager::handle_connect: %s", e.message().c_str());
#endif
    }
}

void ConnectionManager::do_read(ConnectionPointer conn)
{
    MessagePointer message = makeMessage();

    // Issue a read operation to read exactly the number of bytes in a header.
    boost::asio::async_read(
            conn->socket_,
            boost::asio::buffer(&message->header_, sizeof(message->header_)),
            boost::bind(&ConnectionManager::handle_read_header, this, boost::asio::placeholders::error, message, conn)
            );
}

void ConnectionManager::handle_read_header(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
{
    if (!e)
    {
        //
        // TODO:
        // Need to deserialize the message-header!
        //

        // Allocate memory for the message data
        message->data_.resize(message->header_.size_);

        assert( message->header_.size_ != 0 );
        assert( message->header_.size_ == message->data_.size() );

        // Start an asynchronous call to receive the data.
        boost::asio::async_read(
                conn->socket_,
                boost::asio::buffer(&message->data_[0], message->data_.size()),
                boost::bind(&ConnectionManager::handle_read_data, this, boost::asio::placeholders::error, message, conn)
                );
    }
    else
    {
#ifndef NDEBUG
        printf("ConnectionManager::handle_read_header: %s", e.message().c_str());
#endif

#if 1
        // Call the connection's slot
        conn->signal_(Connection::Read, message, e);
#endif
        // Delete the connection
        remove_connection(conn);
    }
}

void ConnectionManager::handle_read_data(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
{
#if TIME_READS
    static virvo::FrameCounter counter;

    printf("CM: reads/sec: %.2f\n", counter.registerFrame());
#endif

    // Call the connection's slot
    conn->signal_(Connection::Read, message, e);

    if (!e)
    {
        // Read the next message
        do_read(conn);
    }
    else
    {
#ifndef NDEBUG
        printf("ConnectionManager::handle_read_data: %s", e.message().c_str());
#endif

        remove_connection(conn);
    }
}

void ConnectionManager::write(MessagePointer msg, ConnectionPointer conn)
{
    strand_.post(boost::bind(&ConnectionManager::do_write, this, msg, conn));
}

void ConnectionManager::do_write(MessagePointer msg, ConnectionPointer conn)
{
    write_queue_.push_back(std::make_pair(conn, msg));

    if (write_queue_.size() == 1)
    {
        do_write();
    }
}

void ConnectionManager::do_write()
{
    // Get the next message from the queue
    std::pair<ConnectionPointer, MessagePointer> msg = write_queue_.front();

    //
    // TODO:
    // Need to serialize the message-header!
    //

    assert( msg.second->header_.size_ != 0 );
    assert( msg.second->header_.size_ == msg.second->data_.size() );

    // Send the header and the data in a single write operation.
    std::vector<boost::asio::const_buffer> buffers;

    buffers.push_back(boost::asio::const_buffer(&msg.second->header_, sizeof(msg.second->header_)));
    buffers.push_back(boost::asio::const_buffer(&msg.second->data_[0], msg.second->data_.size()));

    // Start the write operation.
    boost::asio::async_write(
            msg.first->socket_,
            buffers,
            boost::bind(&ConnectionManager::handle_write, this, boost::asio::placeholders::error, msg.second, msg.first)
            );
}

void ConnectionManager::handle_write(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
{
#if TIME_WRITES
    static virvo::FrameCounter counter;

    printf("CM: writes/sec: %.2f\n", counter.registerFrame());
#endif

    // Call the connection's slot
    conn->signal_(Connection::Write, message, e);

    // Remove the message from the queue
    write_queue_.pop_front();

    if (!e)
    {
        // Message successfully sent.
        // Send the next one -- if any.
        if (!write_queue_.empty())
        {
            do_write();
        }
    }
    else
    {
#ifndef NDEBUG
        printf("ConnectionManager::handle_write: %s", e.message().c_str());
#endif

        remove_connection(conn);
    }
}

void ConnectionManager::add_connection(ConnectionPointer conn)
{
    // Save the connection
    connections_.insert(conn);

    // Start reading messages
    do_read(conn);
}

void ConnectionManager::remove_connection(ConnectionPointer conn)
{
    // Delete the connection
    connections_.erase(conn);
}

#endif // DESKVOX_USE_ASIO
