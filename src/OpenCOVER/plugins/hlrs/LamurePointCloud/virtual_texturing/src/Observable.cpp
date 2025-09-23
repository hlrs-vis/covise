// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/Observable.h>

namespace vt
{
void Observable::observe(event_type event, Observer *observer)
{
    auto eventIter = _events.find(event);

    if(eventIter == _events.end())
    {
        eventIter = _events.insert(std::pair<event_type, std::set<Observer *>>(event, std::set<Observer *>())).first;
    }

    auto observerSet = eventIter->second;

    auto observerIter = observerSet.find(observer);

    if(observerIter != observerSet.end())
    {
        return;
    }

    eventIter->second.insert(observer);
}

void Observable::unobserve(event_type event, Observer *observer)
{
    auto eventIter = _events.find(event);

    if(eventIter == _events.end())
    {
        return;
    }

    auto observerSet = eventIter->second;

    auto observerIter = observerSet.find(observer);

    if(observerIter == observerSet.end())
    {
        return;
    }

    //(*observerIter) = nullptr;
    observerSet.erase(observer);
}

void Observable::inform(event_type event)
{
    Observer **observers;
    size_t len;

    {
        auto iter = _events.find(event);

        if(iter == _events.end())
        {
            return;
        }

        len = iter->second.size();
        observers = new Observer *[len];

        size_t i = 0;

        for(Observer *ptr : iter->second)
        {
            observers[i++] = ptr;
        }
    }

    for(size_t i = 0; i < len; ++i)
    {
        observers[i]->inform(event, this);
    }

    delete observers;
}
}