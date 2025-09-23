// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef TILE_PROVIDER_QUADTREE_H
#define TILE_PROVIDER_QUADTREE_H


#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <lamure/vt/common.h>

namespace vt{
    namespace pre {
    class VIRTUAL_TEXTURING_DLL QuadTree
    {
        public:
            enum NEIGHBOUR {
                TOP = 1,
                BOTTOM,
                LEFT,
                RIGHT
            };

            static inline uint64_t firstIdOfLevel(size_t level) {
                return (uint64_t) 0x5555555555555555 ^ ((uint64_t) 0x5555555555555555 << (level << 1));
            }

            static inline uint64_t getWidthOfLevel(size_t level) {
                return (uint64_t) 0x1 << level;
            }

            static uint64_t getIdFromCoordinates(uint64_t u, uint64_t v, size_t level);

            static uint64_t getIdFromNormalCoordinates(float u, float v, size_t level);

            static void getCoordinatesInLevel(uint64_t relId, size_t level, uint64_t &x, uint64_t &y);

            static size_t getDepth(size_t width, size_t height);

            static uint64_t getNeighbour(uint64_t relId, NEIGHBOUR neighbour);
        };
    }
}


#endif //TILE_PROVIDER_QUADTREE_H
