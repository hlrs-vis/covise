// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OOC_TILEREQUEST_H
#define VT_OOC_TILEREQUEST_H


//#include <lamure/vt/PriorityHeap.h>
#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/vt/Observable.h>

namespace vt{
    namespace ooc{
    class VIRTUAL_TEXTURING_DLL TileRequest : /*public PriorityHeapContent<uint32_t>,*/ public Observable
    {
        protected:
            pre::AtlasFile *_resource;
            uint64_t _id;
            uint32_t _priority;
            bool _aborted;

        public:
            explicit TileRequest();

            void setResource(pre::AtlasFile *resource);

            pre::AtlasFile *getResource();

            void setId(uint64_t id);

            uint64_t getId();

            void setPriority(uint32_t priority){
                _priority = priority;
            }

            uint32_t getPriority(){
                return _priority;
            }

            void erase();

            void abort();

            bool isAborted();
        };
    }
}


#endif //TILE_PROVIDER_TILEREQUEST_H
