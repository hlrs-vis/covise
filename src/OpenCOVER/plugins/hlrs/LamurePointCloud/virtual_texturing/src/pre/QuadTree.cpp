// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/vt/pre/QuadTree.h"

namespace vt {
    namespace pre {
        uint64_t QuadTree::getIdFromCoordinates(uint64_t u, uint64_t v, size_t level) {
            uint64_t id = 0;
            uint64_t total = 0;

            for (uint8_t i = 1; i <= level; ++i) {
                uint8_t inv_level = level - i;
                uint8_t quadrant = (uint8_t) (((u >> inv_level) & 0x1) | (((v >> inv_level) & 0x1) << 1));
                id |= (quadrant << (inv_level << 1));
            }

            return id + QuadTree::firstIdOfLevel(level);
        }

        uint64_t QuadTree::getIdFromNormalCoordinates(float u, float v, size_t level) {
            uint64_t tile_len = QuadTree::getWidthOfLevel(level);
            uint64_t coord_u = tile_len * u;
            uint64_t coord_v = tile_len * v;

            return getIdFromCoordinates(coord_u, coord_v, level);
        }

        void QuadTree::getCoordinatesInLevel(uint64_t relId, size_t level, uint64_t &x, uint64_t &y) {
            x = 0;
            y = 0;

            while (true) {
                x <<= 1;
                y <<= 1;

                auto quadrant = (uint8_t) ((relId >> (level << 1)) & 0x3);

                if (quadrant & 1) {
                    x |= 1;
                }

                if (quadrant & 2) {
                    y |= 1;
                }

                if (level == 0) {
                    break;
                }

                --level;
            }
        }

        size_t QuadTree::getDepth(size_t width, size_t height) {
            size_t maxSize = std::max(width, height);
            bool notPowerOfTwo = false;

            for (uint8_t byte = 0; byte < sizeof(size_t); ++byte) {
                for (uint8_t bit = 0; bit < 8; ++bit) {
                    if (notPowerOfTwo) {
                        if (maxSize == 0) {
                            return (byte << 3) + bit + 1;
                        }
                    } else {
                        if (maxSize == 1) {
                            return (byte << 3) + bit + 1;
                        } else if ((maxSize & 1) == 1) {
                            notPowerOfTwo = true;
                        }
                    }

                    maxSize >>= 1;
                }
            }

            return 0;
        }

        uint64_t QuadTree::getNeighbour(uint64_t relId, NEIGHBOUR neighbour) {
            switch (neighbour) {
                case NEIGHBOUR::TOP:
                    if (relId == 0 || (relId & 2) == 2) {
                        return relId & ~2;
                    } else {
                        auto relIdUpperLevel = relId >> 2;
                        auto neighbourIdUpperLevel = QuadTree::getNeighbour(relIdUpperLevel, neighbour);

                        if (relIdUpperLevel == neighbourIdUpperLevel) {
                            return relId;
                        }

                        return QuadTree::getNeighbour(((neighbourIdUpperLevel << 2) | (relId & 1)), NEIGHBOUR::BOTTOM);
                    }

                    break;
                case NEIGHBOUR::BOTTOM:
                    if ((relId & 2) == 0) {
                        return relId | 2;
                    } else {
                        auto relIdUpperLevel = relId >> 2;
                        auto neighbourIdUpperLevel = QuadTree::getNeighbour(relIdUpperLevel, neighbour);

                        return (neighbourIdUpperLevel << 2) | (relId & 1);
                    }

                    break;
                case NEIGHBOUR::LEFT:
                    if (relId == 0 || (relId & 1) == 1) {
                        return relId & ~1;
                    } else {
                        auto relIdUpperLevel = relId >> 2;
                        auto neighbourIdUpperLevel = QuadTree::getNeighbour(relIdUpperLevel, neighbour);

                        if (relIdUpperLevel == neighbourIdUpperLevel) {
                            return relId;
                        }

                        return QuadTree::getNeighbour(((neighbourIdUpperLevel << 2) | (relId & 2)), NEIGHBOUR::RIGHT);
                    }

                    break;
                case NEIGHBOUR::RIGHT:
                    if ((relId & 1) == 0) {
                        return relId | 1;
                    } else {
                        auto relIdUpperLevel = relId >> 2;
                        auto neighbourIdUpperLevel = QuadTree::getNeighbour(relIdUpperLevel, neighbour);

                        return (neighbourIdUpperLevel << 2) | (relId & 2);
                    }

                    break;
            }
        }
    }
}