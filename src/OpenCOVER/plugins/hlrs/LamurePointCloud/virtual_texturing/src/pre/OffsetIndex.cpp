// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/pre/OffsetIndex.h>

namespace vt {
    namespace pre {
        size_t OffsetIndex::_idToIdx(uint64_t id){
            size_t idx = 0;

            switch(_layout){
                case AtlasFile::LAYOUT::RAW:
                    idx = id;
                    break;
                case AtlasFile::LAYOUT::PACKED:
                    idx = id + 1;
                    break;
                default:
                    throw std::runtime_error("Unknown file layout.");
            }

            return idx;
        }

        OffsetIndex::OffsetIndex(size_t size, AtlasFile::LAYOUT layout) : Index<uint64_t>(size + 1) {
            _layout = layout;
        }

        bool OffsetIndex::exists(uint64_t id) {
            return (_data[_idToIdx(id)] & EXISTS_BIT) != 0;
        }

        size_t OffsetIndex::getOffset(uint64_t id){
            return _data[_idToIdx(id)] & ~EXISTS_BIT;
        }

        size_t OffsetIndex::getLength(uint64_t id){
            size_t idx = _idToIdx(id);
            size_t nextIdx = 0;

            switch(_layout){
                case AtlasFile::LAYOUT::RAW:
                    nextIdx = idx + 1;
                    break;
                case AtlasFile::LAYOUT::PACKED:
                    nextIdx = idx - 1;
                    break;
                default:
                    throw std::runtime_error("Unknown file layout.");
            }

            uint64_t offset = _data[idx] & ~EXISTS_BIT;
            uint64_t nextOffset = _data[nextIdx] & ~EXISTS_BIT;

            return nextOffset - offset;
        }

        void OffsetIndex::set(uint64_t id, uint64_t offset, size_t byteSize){
            size_t idx = _idToIdx(id);
            size_t nextIdx = 0;

            switch(_layout){
                case AtlasFile::LAYOUT::RAW:
                    nextIdx = idx + 1;
                    break;
                case AtlasFile::LAYOUT::PACKED:
                    nextIdx = idx - 1;
                    break;
                default:
                    throw std::runtime_error("Unknown file layout.");
            }

            _data[idx] = offset | EXISTS_BIT;
            _data[nextIdx] = (offset + byteSize) & ~EXISTS_BIT;
        }
    }
}