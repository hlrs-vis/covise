/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __XenomaiEvent_h
#define __XenomaiEvent_h

#include <native/event.h>

class XenomaiEvent
{
public:
    XenomaiEvent(unsigned long ivalue, const std::string &name = "no name", int mode = EV_FIFO)
    {
        int ret_create = rt_event_create(&rt_event_desc, name.c_str(), ivalue, mode);
        if (ret_create)
        {
            std::cerr << "XenomaiEvent::XenomaiEvent(): rt_event_create: " << strerror(-ret_create) << std::endl;
        }
    }

    ~XenomaiEvent()
    {
        int ret_delete = rt_event_delete(&rt_event_desc);
        if (ret_delete)
        {
            std::cerr << "XenomaiEvent::XenomaiEvent(): rt_event_delete: " << strerror(-ret_delete) << std::endl;
        }
    }

    int wait(unsigned long mask, unsigned long *rmask = NULL, int mode = EV_ANY, RTIME timeout = TM_INFINITE)
    {
        int ret_acquire = rt_event_wait(&rt_event_desc, mask, rmask, mode, timeout);
        if (ret_acquire)
        {
            if (ret_acquire != -ETIMEDOUT)
            {
                std::cerr << "XenomaiEvent::XenomaiEvent(): rt_event_wait: " << strerror(-ret_acquire) << std::endl;
            }
        }
        return ret_acquire;
    }

    int clear(unsigned long mask, unsigned long *rmask = NULL)
    {
        int ret_acquire = rt_event_clear(&rt_event_desc, mask, rmask);
        if (ret_acquire)
        {
            std::cerr << "XenomaiEvent::XenomaiEvent(): rt_event_clear: " << strerror(-ret_acquire) << std::endl;
        }
        return ret_acquire;
    }

    int signal(unsigned long mask)
    {
        int ret_acquire = rt_event_signal(&rt_event_desc, mask);
        if (ret_acquire)
        {
            std::cerr << "XenomaiEvent::XenomaiEvent(): rt_event_signal: " << strerror(-ret_acquire) << std::endl;
        }
        return ret_acquire;
    }

protected:
    RT_EVENT rt_event_desc;
};

#endif
