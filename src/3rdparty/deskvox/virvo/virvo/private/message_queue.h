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

#ifndef VV_PRIVATE_MESSAGE_QUEUE_H
#define VV_PRIVATE_MESSAGE_QUEUE_H

#include "vvmessage.h"
#include "blocking_queue.h"

namespace virvo
{

// Thread safe queue to store messages
class MessageQueue
{
    BlockingQueue<MessagePointer> queue_;

public:
    // Unconditionally add a message to the queue.
    void push_back(MessagePointer message)
    {
        queue_.push_back(message);
    }

    // Add or replace a message.
    // The message is only added if the queue is empty or the last message of the queue
    // and the message to be added have the same type.
    // Otherwise this will replace the last message with the new one.
    bool push_back_merge(MessagePointer message)
    {
        return queue_.replace_back_if(message, TypesEqual());
    }

    // Get the next message from the queue.
    MessagePointer pop_front()
    {
        return queue_.pop_front();
    }

private:
    struct TypesEqual
    {
        bool operator ()(MessagePointer const& lhs, MessagePointer const& rhs) const {
            return lhs->type() == rhs->type();
        }
    };
};

} // namespace virvo

#endif
