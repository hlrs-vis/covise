// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/ooc/TileProvider.h>

namespace vt{
    namespace ooc{
        TileProvider::TileProvider() {
            _cache = nullptr;
            _tileByteSize = 0;
        }

        TileProvider::~TileProvider(){
            for(auto resource : _resources){
                delete resource;
            }

            delete _cache;
        }

        void TileProvider::start(size_t maxMemSize){
            if(_tileByteSize == 0){
                throw std::runtime_error("TileProvider tries to start loading Tiles of size 0.");
            }

            auto slotCount = maxMemSize / _tileByteSize;

            if(slotCount == 0){
                throw std::runtime_error("TileProvider tries to start with Cache of size 0.");
            }

            _cache = new TileCache(_tileByteSize, slotCount);
            _loader.writeTo(_cache);
            _loader.start();
        }

        pre::AtlasFile *TileProvider::addResource(const char *fileName){
            std::lock_guard<std::mutex> lock(_resourcesLock);

            auto atlas = new pre::AtlasFile(fileName);

            if(_tileByteSize == 0){
                _pxFormat = atlas->getPixelFormat();
                _tilePxWidth = atlas->getTileWidth();
                _tilePxHeight = atlas->getTileHeight();
                _tileByteSize = atlas->getTileByteSize();
            }

            if(_pxFormat != atlas->getPixelFormat() ||
                    _tilePxWidth != atlas->getTileWidth() ||
                    _tilePxHeight != atlas->getTileHeight() ||
                    _tileByteSize != atlas->getTileByteSize()){
                throw std::runtime_error("Trying to add resource with conflicting format.");
            }

            _resources.insert(atlas);

            return atlas;
        }

        TileCacheSlot *TileProvider::getTile(pre::AtlasFile *resource, id_type id, priority_type priority){
            if(_cache == nullptr){
                throw std::runtime_error("Trying to get Tile before starting TileProvider.");
            }

            auto slot = _cache->readSlotById(resource, id);

            if(slot != nullptr){
                return slot;
            }

            auto req = _requests.getRequest(resource, id);

            if(req != nullptr){
                // if one wants to rensert according to priority, this should happen here
                req->setPriority(priority);

                return nullptr;
            }

            ++_requested;

            req = new TileRequest;

            req->setResource(resource);
            req->setId(id);
            req->setPriority(priority);

            _requests.insertRequest(req);

            _loader.request(req);

            return nullptr;
        }

        void TileProvider::ungetTile(TileCacheSlot *slot){
            _cache->setSlotReady(slot);
        }

        void TileProvider::stop(){
            _loader.stop();
        }

        void TileProvider::print(){
            _cache->print();
        }

        bool TileProvider::wait(std::chrono::milliseconds maxTime){
            return _requests.waitUntilEmpty(maxTime);
        }

        bool TileProvider::ungetTile(pre::AtlasFile *resource, id_type id) {
            if(_cache == nullptr){
                throw std::runtime_error("Trying to unget Tile before starting TileProvider.");
            }

            auto slot = _cache->readSlotById(resource, id);

            if(slot == nullptr){
                return false;
            }

            _cache->setSlotReady(slot);

            return true;
        }

    uint64_t TileProvider::get_requested(){
        auto requested = _requested;

        _requested = 0;

        return requested;
    }

    uint64_t TileProvider::get_loaded(){
        return _cache->tiles_loaded();
    }
    }
}
