// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OOC_TILECACHE_H
#define VT_OOC_TILECACHE_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <iostream>
#include <lamure/vt/pre/AtlasFile.h>
#include <queue>
#include <map>
#include <condition_variable>
#include <chrono>

namespace vt {
    namespace ooc {
        class TileCache;

        class VIRTUAL_TEXTURING_DLL TileCacheSlot
        {
        public:
            enum STATE{
                FREE = 1,
                WRITING,
                READING,
                OCCUPIED
            };

        protected:
            STATE _state;
            uint8_t *_buffer;
            size_t _size;
            size_t _id;
            std::mutex _lock;
            //assoc_data_type _assocData;

            pre::AtlasFile *_resource;
            uint64_t _tileId;

            TileCache *_cache;

        public:
            TileCacheSlot();

            ~TileCacheSlot();

            bool hasState(STATE state);

            void setTileId(uint64_t tileId);

            uint64_t getTileId();

            void setCache(TileCache *cache);

            void setState(STATE state);

            void setId(size_t id);

            size_t getId();

            void setBuffer(uint8_t *buffer);

            uint8_t *getBuffer();

            void setSize(size_t size);

            size_t getSize();

            void setResource(pre::AtlasFile* res);

            pre::AtlasFile *getResource();

            void removeFromIDS();
        };

        class TileCache {
        protected:
            typedef TileCacheSlot slot_type;

            size_t _tileByteSize;
            size_t _slotCount;
            uint8_t *_buffer;
            slot_type *_slots;
            std::mutex *_locks;

            std::mutex _lruLock;
            std::condition_variable _lruCondVar;
            std::queue<TileCacheSlot*> _lru;

            std::mutex _idsLock;
            std::map<std::pair<pre::AtlasFile *, uint64_t>, slot_type *> _ids;

            uint64_t _loaded = 0;

        public:
            uint64_t tiles_loaded();

            TileCache(size_t tileByteSize, size_t slotCount);

            slot_type *readSlotById(pre::AtlasFile *resource, uint64_t id);

            slot_type *writeSlot(std::chrono::milliseconds maxTime = std::chrono::milliseconds::zero());

            void setSlotReady(slot_type *slot);

            void unregisterId(pre::AtlasFile *resource, uint64_t id);

            ~TileCache();

            void print();
        };
    }
}


#endif //TILE_PROVIDER_TILECACHE_H
