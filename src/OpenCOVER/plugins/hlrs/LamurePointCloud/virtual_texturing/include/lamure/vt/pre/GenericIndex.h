// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef TILE_PROVIDER_GENERICINDEX_H
#define TILE_PROVIDER_GENERICINDEX_H


#include <cstdint>
#include <cstddef>
#include <lamure/vt/pre/QuadTree.h>

namespace vt{
    namespace pre {
    class VIRTUAL_TEXTURING_DLL GenericIndex
    {
        private:
            uint64_t _imageWidth;
            uint64_t _imageHeight;
            size_t _tileWidth;
            size_t _tileHeight;
            size_t _tileSize;

            size_t _indexSize;
            uint64_t *_index;

        public:
            GenericIndex(uint64_t imageWidth, uint64_t imageHeight, size_t tileWidth, size_t tileHeight,
                         size_t tileSize) {
                _imageWidth = imageWidth;
                _imageHeight = imageHeight;
                _tileWidth = tileWidth;
                _tileHeight = tileHeight;
                _tileSize = tileSize;

                size_t levelTileWidth = (imageWidth + tileWidth - 1) / tileWidth;
                size_t levelTileHeight = (imageHeight + tileHeight - 1) / tileHeight;


                auto treeDepth = QuadTree::getDepth(levelTileWidth, levelTileHeight);
                auto tilesInTree = QuadTree::firstIdOfLevel(treeDepth);

                _indexSize = tilesInTree;
                _index = new uint64_t[tilesInTree];

                uint64_t x;
                uint64_t y;

                uint64_t firstIdOfLevel = tilesInTree;
                uint64_t absId = firstIdOfLevel;
                uint64_t offsetInFile = 0;

                for (size_t level = treeDepth - 1;; --level) {
                    firstIdOfLevel = QuadTree::firstIdOfLevel(level);

                    do {
                        --absId;

                        QuadTree::getCoordinatesInLevel(absId - firstIdOfLevel, level, x, y);

                        if (x < levelTileWidth && y < levelTileHeight) {
                            _index[absId] = offsetInFile;
                            offsetInFile += tileSize;
                        } else {
                            _index[absId] = UINT64_MAX;
                        }
                    } while (absId > firstIdOfLevel);

                    if (level == 0) {
                        break;
                    }

                    levelTileWidth = (levelTileWidth + 1) >> 1;
                    levelTileHeight = (levelTileHeight + 1) >> 1;
                }
            }

            uint64_t getOffset(uint64_t id) const {
                if (id >= _indexSize) {
                    return UINT64_MAX;
                }

                return _index[id];
            }
        };
    }
}


#endif //TILE_PROVIDER_GENERICINDEX_H
