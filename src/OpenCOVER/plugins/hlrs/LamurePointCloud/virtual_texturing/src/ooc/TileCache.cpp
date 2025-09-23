// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/ooc/TileCache.h>

namespace vt {
    namespace ooc {
        typedef TileCacheSlot slot_type;

        TileCacheSlot::TileCacheSlot(){
            _state = STATE::FREE;
            _buffer = nullptr;
            _cache = nullptr;
            _size = 0;
            _tileId = 0;
        }

        TileCacheSlot::~TileCacheSlot(){
            //std::cout << "del slot " << this << std::endl;
        }

        bool TileCacheSlot::hasState(STATE state){
            return _state == state;
        }

        void TileCacheSlot::setTileId(uint64_t tileId){
            _tileId = tileId;
        }

        uint64_t TileCacheSlot::getTileId(){
            return _tileId;
        }

        void TileCacheSlot::setCache(TileCache *cache){
            _cache = cache;
        }

        void TileCacheSlot::setState(STATE state){
            _state = state;
        }

        void TileCacheSlot::setId(size_t id){
            _id = id;
        }

        size_t TileCacheSlot::getId(){
            return _id;
        }

        void TileCacheSlot::setBuffer(uint8_t *buffer){
            _buffer = buffer;
        }

        uint8_t *TileCacheSlot::getBuffer(){
            return _buffer;
        }

        void TileCacheSlot::setSize(size_t size){
            _size = size;
        }

        size_t TileCacheSlot::getSize(){
            return _size;
        }

        void TileCacheSlot::setResource(pre::AtlasFile* res){
            _resource = res;
        }

        pre::AtlasFile *TileCacheSlot::getResource(){
            return _resource;
        }

        void TileCacheSlot::removeFromIDS(){
            std::lock_guard<std::mutex> lock(_lock);

            if(_cache != nullptr){
                _cache->unregisterId(_resource, _tileId);
            }
        }

        TileCache::TileCache(size_t tileByteSize, size_t slotCount) {
            _tileByteSize = tileByteSize;
            _slotCount = slotCount;
            _buffer = new uint8_t[tileByteSize * slotCount];
            _slots = new slot_type[slotCount];
            _locks = new std::mutex[slotCount];

            //_counter = 0;

            for(size_t i = 0; i < slotCount; ++i){
                _slots[i].setId(i);
                _slots[i].setBuffer(&_buffer[tileByteSize * i]);
                _slots[i].setCache(this);

                _lru.push(&_slots[i]);
            }
        }

        slot_type *TileCache::readSlotById(pre::AtlasFile *resource, uint64_t id){
            std::lock_guard<std::mutex> lock(_idsLock);

            auto iter = _ids.find(std::make_pair(resource, id));

            if(iter == _ids.end()){
                return nullptr;
            }

            auto slot = iter->second;

            slot->setState(slot_type::STATE::READING);

            return slot;
        }

        slot_type *TileCache::writeSlot(std::chrono::milliseconds maxTime){
            std::unique_lock<std::mutex> lock(_lruLock);

            if(_lru.empty()){
                auto destTime = std::chrono::system_clock::now() + maxTime;

                if(!_lruCondVar.wait_until(lock, destTime, [this]{
                    return !_lru.empty();
                })){
                    return nullptr;
                }
            }

            TileCacheSlot *slot = nullptr;

            do{
                slot = _lru.front();
                _lru.pop();

                if(slot->hasState(TileCacheSlot::STATE::FREE) || slot->hasState(TileCacheSlot::STATE::OCCUPIED)){
                    break;
                }
            }while(!_lru.empty());

            lock.unlock();

            if(slot->hasState(TileCacheSlot::STATE::READING) || slot->hasState(TileCacheSlot::STATE::WRITING)) {
                return nullptr;
            }

            if(!slot->hasState(TileCacheSlot::STATE::FREE)) {
                slot->removeFromIDS();
            }

            slot->setState(slot_type::STATE::WRITING);

            return slot;
        }

        uint64_t TileCache::tiles_loaded(){
            auto loaded = _loaded;

            _loaded = 0;

            return loaded;
        }

        void TileCache::setSlotReady(slot_type *slot){
            std::unique_lock<std::mutex> lockSlot(_locks[slot->getId()]);

            if(slot->hasState(slot_type::STATE::WRITING)){
                std::lock_guard<std::mutex> lock(_idsLock);
                _ids.insert(std::make_pair(std::make_pair(slot->getResource(), slot->getTileId()), slot));
                ++_loaded;
            }

            slot->setState(slot_type::STATE::OCCUPIED);

            lockSlot.unlock();

            std::unique_lock<std::mutex> lock(_lruLock);
            bool wasEmpty = _lru.empty();
            _lru.push(slot);
            lock.unlock();

            if(wasEmpty){
                _lruCondVar.notify_all();
            }
        }

        void TileCache::unregisterId(pre::AtlasFile *resource, uint64_t id){
            _ids.erase(std::make_pair(resource, id));
        }

        TileCache::~TileCache(){
            delete[] _buffer;
            delete[] _slots;
            delete[] _locks;
        }

        void TileCache::print(){
            std::cout << "Slots: " << std::endl;

            for(size_t i = 0; i < _slotCount; ++i){
                auto slot = &_slots[i];

                if(slot->hasState(slot_type::STATE::FREE)) {
                    std::cout << "\tx ";
                }else{
                    std::cout << "\t  ";
                }

                std::cout << slot->getTileId() << std::endl;
            }

            std::cout << std::endl << "IDs:" << std::endl;

            for(auto pair : _ids){
                std::cout << "\t" << pair.first.first << " " << pair.first.second << " --> " << pair.second->getResource() << " " << pair.second->getTileId() << std::endl;
            }

            std::cout << std::endl;
        }
    }
}