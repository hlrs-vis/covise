/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __XenomaiMutex_h
#define __XenomaiMutex_h

#ifdef MERCURY
#include <alchemy/mutex.h>
#else
#include <native/mutex.h>
#endif

class XenomaiMutex
{
public:
    XenomaiMutex(const std::string &name = "no name")
    {
        int ret_create = rt_mutex_create(&rt_mutex_desc, name.c_str());
        if (ret_create)
        {
            std::cerr << "XenomaiMutex::XenomaiMutex(): rt_mutex_create: " << strerror(-ret_create) << std::endl;
        }
    }

    ~XenomaiMutex()
    {
        int ret_delete = rt_mutex_delete(&rt_mutex_desc);
        if (ret_delete)
        {
            std::cerr << "XenomaiMutex::XenomaiMutex(): rt_mutex_delete: " << strerror(-ret_delete) << std::endl;
        }
    }

    int acquire(RTIME timeout = TM_INFINITE)
    {
        int ret_acquire = rt_mutex_acquire(&rt_mutex_desc, timeout);
        if (ret_acquire)
        {
            std::cerr << "XenomaiMutex::XenomaiMutex(): rt_mutex_acquire: " << strerror(-ret_acquire) << std::endl;
        }
        return ret_acquire;
    }

    int release()
    {
        int ret_release = rt_mutex_release(&rt_mutex_desc);
        if (ret_release)
        {
            std::cerr << "XenomaiMutex::XenomaiMutex(): rt_mutex_release: " << strerror(-ret_release) << std::endl;
        }
        return ret_release;
    }

protected:
    RT_MUTEX rt_mutex_desc;
};

#endif
