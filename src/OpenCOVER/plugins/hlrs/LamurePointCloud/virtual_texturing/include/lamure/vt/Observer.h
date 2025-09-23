// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OBSERVER_H
#define VT_OBSERVER_H

#include <cstdint>
#include <lamure/vt/common.h>

namespace vt {

    typedef uint32_t event_type;

    class Observable;

    class Observer
    {
    protected:
    public:
        Observer() = default;
        virtual void inform(event_type event, Observable *observable) = 0;
    };

}

#endif //TILE_PROVIDER_OBSERVER_H
