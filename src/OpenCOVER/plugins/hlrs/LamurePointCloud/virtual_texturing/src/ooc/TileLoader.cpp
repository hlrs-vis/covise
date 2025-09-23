// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/ooc/TileLoader.h>

namespace vt {
    namespace ooc {
        TileLoader::TileLoader() : HeapProcessor(){

        }

        void TileLoader::beforeStart(){
        }

        void TileLoader::process(TileRequest *req){
            if(!req->isAborted()) {
                auto res = req->getResource();
                auto slot = _cache->writeSlot(std::chrono::milliseconds(10));

                if (slot == nullptr) {
                    throw std::runtime_error("Cache seems to be full.");
                }

                res->getTile(req->getId(), slot->getBuffer());

                // provide information on contained tile
                slot->setSize(res->getTileByteSize());
                slot->setResource(res);
                slot->setTileId(req->getId());

                // make slot accessible for reading
                _cache->setSlotReady(slot);
            }

            // erase request, because it is processed
            req->erase();
            //delete req;
        }

        void TileLoader::beforeStop(){

        }
    }
}