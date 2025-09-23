// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/semaphore.h>

#include <iostream>

namespace lamure
{

semaphore::
semaphore()
: signal_count_(0),
  shutdown_(false),
  min_signal_count_(1),
  max_signal_count_(1) {


}

semaphore::
~semaphore() {

}

void semaphore::
wait() {
#if 1
  {
    std::unique_lock<std::mutex> ulock(mutex_);
    signal_lock_.wait(ulock, [&]{ return signal_count_ >= min_signal_count_ || shutdown_; });
    if (signal_count_ >= min_signal_count_) {
      signal_count_ -= min_signal_count_;
    }
  }
#else 
  while (true) {
    if (signal_count_ >= min_signal_count_) {
      mutex_.lock();
      if (signal_count_ >= min_signal_count_) {
        signal_count_ -= min_signal_count_;
        mutex_.unlock();
        break;
      }
      mutex_.unlock();
    }
    
    if (shutdown_) {
      break;
    }
  
  }

#endif

}

void semaphore::
signal(const size_t signal_count) {
#if 1
  {
    std::unique_lock<std::mutex> ulock(mutex_);
    if (signal_count_+signal_count <= max_signal_count_) {
      signal_count_ += signal_count;
    }
  }
  signal_lock_.notify_all();
#else 
  mutex_.lock();
  if (signal_count_+signal_count <= max_signal_count_) {
    signal_count_ += signal_count;
  }
  mutex_.unlock();
#endif

}

const size_t semaphore::
num_signals() {
    std::lock_guard<std::mutex> lock(mutex_);
    return signal_count_;
}

void semaphore::
lock() {
    mutex_.lock();
}

void semaphore::
unlock() {
    mutex_.unlock();
}

void semaphore::
shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    signal_lock_.notify_all();
}

} // namespace lamure

