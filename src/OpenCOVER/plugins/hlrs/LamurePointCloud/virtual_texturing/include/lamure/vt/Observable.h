// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OBSERVABLE_H
#define VT_OBSERVABLE_H

#include <vector>
#include <cstdint>
#include <map>
#include <set>
#include <lamure/vt/Observer.h>

namespace vt {

    class Observable
{
    protected:
        std::map<event_type, std::set<Observer*>> _events;
    public:
        Observable() = default;

        void observe(event_type event, Observer *observer);

        void unobserve(event_type event, Observer *observer);

        virtual void inform(event_type event);
    };

}


#endif //TILE_PROVIDER_OBSERVABLE_H
