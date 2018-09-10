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

#ifndef VV_PRIVATE_WORK_QUEUE_H
#define VV_PRIVATE_WORK_QUEUE_H

#include <boost/asio/io_service.hpp>
#include <boost/asio/strand.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

namespace virvo
{

class WorkQueue
{
    // The io_service object
    boost::asio::io_service io_service_;
    // Prevents the io_service from exiting if it has nothing to do
    boost::asio::io_service::work work_;
    // Provides serialised handler execution
    boost::asio::io_service::strand strand_;
    // The worker thread
    boost::thread thread_;

public:
    // Constructor.
    WorkQueue()
        : io_service_()
        , work_(io_service_)
        , strand_(io_service_)
        , thread_()
    {
    }

    // Destructor.
    // Stops the worker thread.
    ~WorkQueue()
    {
        stop();
    }

    // Start working
    void run()
    {
        io_service_.run();
    }

    // Start working
    void run_in_thread()
    {
        thread_ = boost::thread(boost::bind(&WorkQueue::run, this));
    }

    // Stop working and wait for the thread to finish
    void stop()
    {
        // Stop the io_service
        io_service_.stop();
        // Wait for the worker thread
        thread_.join();
    }

    // Schedule some work to be executed
    template <class Function>
    void post(Function func)
    {
        strand_.post(func);
    }
};

} // namespace virvo

#endif
