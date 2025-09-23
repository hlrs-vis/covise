// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OOC_TILEPROVIDER_H
#define VT_OOC_TILEPROVIDER_H

#include <cstddef>
#include <cstdint>
#include <set>
#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/vt/ooc/TileRequestMap.h>
#include <lamure/vt/ooc/TileCache.h>
#include <lamure/vt/ooc/TileLoader.h>

namespace vt{
    namespace ooc{
        typedef uint64_t id_type;
        typedef uint32_t priority_type;

        class VIRTUAL_TEXTURING_DLL TileProvider
        {
        protected:
            std::mutex _resourcesLock;
            std::set<pre::AtlasFile*> _resources;
            TileRequestMap _requests;
            TileLoader _loader;
            TileCache *_cache;

            pre::Bitmap::PIXEL_FORMAT _pxFormat;
            size_t _tilePxWidth;
            size_t _tilePxHeight;
            size_t _tileByteSize;

            uint64_t _requested = 0;
            uint64_t _loaded = 0;

        public:
            TileProvider();

            ~TileProvider();

            void start(size_t maxMemSize);

            pre::AtlasFile *addResource(const char *fileName);

            TileCacheSlot *getTile(pre::AtlasFile *resource, id_type id, priority_type priority);

            void ungetTile(TileCacheSlot *slot);

            void stop();

            void print();

            bool wait(std::chrono::milliseconds maxTime = std::chrono::milliseconds::zero());

            bool ungetTile(pre::AtlasFile *resource, id_type id);

            uint64_t get_requested();
            uint64_t get_loaded();
        };
    }
}

#endif //TILE_PROVIDER_TILEPROVIDER_H
