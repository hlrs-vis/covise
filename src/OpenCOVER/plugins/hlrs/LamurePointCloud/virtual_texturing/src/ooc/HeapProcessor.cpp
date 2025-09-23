// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/ooc/HeapProcessor.h>

namespace vt {
    namespace ooc {
        HeapProcessor::HeapProcessor() {
            _thread = nullptr;
            _cache = nullptr;
        }

        HeapProcessor::~HeapProcessor(){
            if(_thread != nullptr){
                delete _thread;
            }
        }

        void HeapProcessor::request(TileRequest *request){
            _requests.push(request);
        }

        void HeapProcessor::start(){
            if(_thread != nullptr){
                throw std::runtime_error("HeapProcessor is already started.");
            }

            if(_cache == nullptr){
                throw std::runtime_error("Cache needs to be set.");
            }

            _running = true;
            _thread = new std::thread(&HeapProcessor::run, this);
        }

        void HeapProcessor::run(){
            beforeStart();

            while(_running.load()){
                TileRequest *req;

                if(!_requests.pop(req, std::chrono::milliseconds(200))) {
                    continue;
                }

                _currentlyProcessing.store(_currentlyProcessing.load() + 1);
                process(req);
                _currentlyProcessing.store(_currentlyProcessing.load() - 1);
            }

            beforeStop();
        }

        void HeapProcessor::writeTo(TileCache *cache){
            _cache = cache;
        }

        void HeapProcessor::stop(){
            _running = false;
            _thread->join();
        }

        size_t HeapProcessor::pendingCount(){
            return _requests.getSize();
        }

        bool HeapProcessor::currentlyProcessing(){
            return _currentlyProcessing > 0;
        }
    }
}