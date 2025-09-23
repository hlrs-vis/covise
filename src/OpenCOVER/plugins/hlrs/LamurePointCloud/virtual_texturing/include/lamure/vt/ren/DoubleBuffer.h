// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_DOUBLEBUFFER_H
#define LAMURE_DOUBLEBUFFER_H

#include <lamure/vt/common.h>
#include <mutex>

namespace vt
{
template <typename T>
class DoubleBuffer
{
  public:
    explicit DoubleBuffer(T *front, T *back)
    {
        _front = front;
        _back = back;
        _new_data = false;
    }
    virtual ~DoubleBuffer(){};

    T *get_front() { return _front; }
    T *get_back() { return _back; }

    virtual void start_writing() { _back_lock.lock(); }
    virtual void stop_writing()
    {
        if(_front_lock.try_lock())
        {
            deliver();
            _front_lock.unlock();

            _new_data = false;
        }
        else
        {
            _new_data = true;
        }

        _back_lock.unlock();
    }
    virtual void start_reading()
    {
        _front_lock.lock();

        if(_back_lock.try_lock())
        {
            if(_new_data)
            {
                deliver();
                _new_data = false;
            }

            _back_lock.unlock();
        }
    }
    virtual void stop_reading() { _front_lock.unlock(); }

  protected:
    T *_front, *_back;

    virtual void deliver() = 0;

  private:
    std::mutex _front_lock, _back_lock;
    bool _new_data;
};
}

#endif // LAMURE_DOUBLEBUFFER_H
