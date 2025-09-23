// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_OOC_HEAPPROCESSOR_H
#define VT_OOC_HEAPPROCESSOR_H

#include <lamure/vt/ooc/HeapProcessor.h>

namespace vt {
    namespace ooc {
    class VIRTUAL_TEXTURING_DLL TileLoader : public HeapProcessor
    {
        public:
            TileLoader();

            void beforeStart() override;

            void process(TileRequest *req) override;

            void beforeStop() override;
        };
    }
}


#endif //TILE_PROVIDER_TILELOADER_H
