// Virvo - Virtual Reality Volume Rendering
// Contact: Stefan Zellmann, zellmans@uni-koeln.de
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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#if VV_HAVE_PTHREADS

#include "vvpthread.h"

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#ifdef VV_USE_CUSTOM_BARRIER_IMPLEMENTATION
int pthread_barrier_init(pthread_barrier_t* barrier,
                         const pthread_barrierattr_t* attr,
                         const unsigned int count)
{
  (void)attr;
  barrier->count = count;
  barrier->waited = 0;
  pthread_mutex_init(&barrier->mutex, NULL);
  pthread_cond_init(&barrier->cond, NULL);
  return 0;
}

int pthread_barrier_destroy(pthread_barrier_t* barrier)
{
  pthread_mutex_destroy(&barrier->mutex);
  pthread_cond_destroy(&barrier->cond);
  return 0;
}

int pthread_barrier_wait(pthread_barrier_t* barrier)
{
  pthread_mutex_lock(&barrier->mutex);
  ++barrier->waited;
  if (barrier->waited == barrier->count)
  {
    barrier->waited = 0;
    pthread_cond_broadcast(&barrier->cond);
  }
  else
  {
    pthread_cond_wait(&barrier->cond, &barrier->mutex);
  }
  pthread_mutex_unlock(&barrier->mutex);
  return 0;
}
#endif

#endif // VV_HAVE_PTHREADS

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
