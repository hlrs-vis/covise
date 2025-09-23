// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OOC_HEAP_PROCESSOR_H
#define VT_OOC_HEAP_PROCESSOR_H


#include <thread>
#include <atomic>
#include <condition_variable>
#include <lamure/vt/ooc/TileCache.h>
#include <lamure/vt/TileRequestPriorityQueue.h>
#include <lamure/vt/ooc/TileRequest.h>

namespace vt {
    namespace ooc {
    class VIRTUAL_TEXTURING_DLL HeapProcessor
    {
        protected:
            TileRequestPriorityQueue<uint32_t> _requests;

            std::atomic<bool> _running;
            std::atomic<size_t> _currentlyProcessing;
            std::thread *_thread;

            TileCache *_cache;
        public:
            HeapProcessor();
            ~HeapProcessor();

            void request(TileRequest *request);

            void start();

            void run();

            virtual void beforeStart() = 0;

            virtual void process(TileRequest *req) = 0;

            virtual void beforeStop() = 0;

            void writeTo(TileCache *cache);

            void stop();

            size_t pendingCount();

            bool currentlyProcessing();
        };
    }
}

#endif //TILE_PROVIDER_HEAP_PROCESSOR_H
