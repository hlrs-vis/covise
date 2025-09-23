// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_OFFSETINDEX_H
#define LAMURE_OFFSETINDEX_H

#include "Index.h"
#include "AtlasFile.h"

namespace vt {
    namespace pre {
    class VIRTUAL_TEXTURING_DLL OffsetIndex : public Index<uint64_t>
    {
        private:
            const uint64_t EXISTS_BIT = 0x8000000000000000;
            AtlasFile::LAYOUT _layout;
            size_t _idToIdx(uint64_t id);

        public:
            explicit OffsetIndex(size_t size, AtlasFile::LAYOUT layout);
            bool exists(uint64_t id);
            size_t getOffset(uint64_t id);
            size_t getLength(uint64_t id);
            void set(uint64_t id, uint64_t offset, size_t byteSize);
        };
    }
}


#endif //LAMURE_OFFSETINDEX_H
