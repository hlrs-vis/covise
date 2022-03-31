#include "threadname.h"

#ifdef __linux
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <cstring>
#include <iostream>
#endif

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace covise
{

bool setThreadName(std::string name)
{
#ifdef __linux
#if __GLIBC__>=2 && __GLIBC_MINOR__>=12
    const size_t maxlen = 15;
    if (name.length() > maxlen) {
        std::cerr << "truncating thread name " << name << " to " << maxlen << " characters" << std::endl;
        name = name.substr(0, maxlen);
    }
    int err = pthread_setname_np(pthread_self(), name.c_str());
    if (err == 0) {
        return true;
    }
    std::cerr << "setThreadName failed: " << strerror(err) << std::endl;
#endif
#endif

#ifdef __APPLE__
    pthread_setname_np(name.c_str());
    return true;
#endif
    return false;
}

}
