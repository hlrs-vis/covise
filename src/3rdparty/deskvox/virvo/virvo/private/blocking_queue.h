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

#ifndef VV_PRIVATE_BLOCKING_QUEUE_H
#define VV_PRIVATE_BLOCKING_QUEUE_H

#include "vvpthread.h"

#include <deque>

namespace virvo
{

template <class T>
class BlockingQueue
{
public:
    typedef std::deque<T> queue_type;

private:
    // The queue
    queue_type queue_;
    // A mutex to protect the queue.
    Mutex lock_;
    // Signaled if the queue becomes non-empty
    Semaphore non_empty_;

public:
    void push_back(T const& object)
    {
        {
            ScopedLock guard(&lock_);

            // Add the object
            queue_.push_back(object);
        }

        non_empty_.signal();
    }

    template <class BinaryPredicate>
    bool replace_back_if(T const& object, BinaryPredicate pred)
    {
        bool replace = false;

        {
            ScopedLock guard(&lock_);

            replace = !queue_.empty() && pred(queue_.front(), object);

            if (replace)
            {
                queue_.back() = object;
            }
            else
            {
                queue_.push_back(object);
            }
        }

        if (!replace)
        {
            non_empty_.signal();
        }

        return replace;
    }

    T pop_front()
    {
        T next;

        // Wait for a new message
        non_empty_.wait();

        {
            ScopedLock guard(&lock_);

            assert(!queue_.empty());

            // Return the front of the queue
            next = queue_.front();
            // Remove the object from the queue
            queue_.pop_front();
        }

        return next;
    }
};

} // namespace virvo

#endif
