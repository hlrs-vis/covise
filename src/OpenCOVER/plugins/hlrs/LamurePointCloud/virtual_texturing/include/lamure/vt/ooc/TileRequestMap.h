// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OOC_TILEREQUESTMAP_H
#define VT_OOC_TILEREQUESTMAP_H


#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/vt/ooc/TileRequest.h>
#include <lamure/vt/Observer.h>
#include <chrono>

namespace vt{
    namespace ooc{
    class VIRTUAL_TEXTURING_DLL TileRequestMap : public Observer
    {
        protected:
            std::map<std::pair<pre::AtlasFile*, uint64_t>, TileRequest*> _map;
            std::mutex _lock;

            std::condition_variable _allRequestsProcessed;

        public:
            TileRequestMap();

            ~TileRequestMap();

            TileRequest *getRequest(pre::AtlasFile *resource, uint64_t id);

            bool insertRequest(TileRequest *req);

            void inform(event_type event, Observable *observable);

            bool waitUntilEmpty(std::chrono::milliseconds maxTime = std::chrono::milliseconds::zero());
        };
    }
}


#endif //TILE_PROVIDER_TILEREQUESTMAP_H
