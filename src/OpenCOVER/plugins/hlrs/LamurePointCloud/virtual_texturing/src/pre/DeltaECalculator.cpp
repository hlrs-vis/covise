// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/pre/DeltaECalculator.h>

namespace vt {
    namespace pre {
        DeltaECalculator::DeltaECalculator(const char *fileName) : AtlasFile(fileName) {

        }

        void DeltaECalculator::calculate(size_t maxMemory) {
            uint64_t actualLevelPxWidth = _imageWidth;
            uint64_t actualLevelPxHeight = _imageHeight;
            uint64_t actualLevelTileWidth = _imageTileWidth;
            uint64_t actualLevelTileHeight = _imageTileHeight;

            size_t innerTilePxCount = _innerTileWidth * _innerTileHeight;
            size_t distBufferByteSize = (_treeDepth + 1) * innerTilePxCount;
            auto distBuffer = new double[distBufferByteSize];

            maxMemory -= distBufferByteSize;

            uint64_t leafLevelFirstId = QuadTree::firstIdOfLevel(_treeDepth - 1);

            auto blackTileBuffer = new uint8_t[_tileByteSize];
            std::memset(blackTileBuffer, 0, _tileByteSize);

            maxMemory -= _tileByteSize;

            size_t rootBufferTileSize = (maxMemory >> 1) / _tileByteSize;
            size_t rootBufferByteSize = rootBufferTileSize * _tileByteSize;
            uint64_t rootBufferOffset = 0;
            auto rootBuffer = new uint8_t[rootBufferByteSize];

            size_t leafBufferTileSize = (maxMemory >> 1) / _tileByteSize;
            size_t leafBufferByteSize = leafBufferTileSize * _tileByteSize;
            uint64_t leafBufferOffset = 0;
            auto leafBuffer = new uint8_t[leafBufferByteSize];

            Bitmap rootBitmap(_tileWidth, _tileHeight, _pxFormat, rootBuffer);
            Bitmap leafBitmap(_tileWidth, _tileHeight, _pxFormat, leafBuffer);

            Bitmap rootLab(_innerTileWidth, _innerTileHeight, Bitmap::PIXEL_FORMAT::LAB);
            Bitmap leafLab(_innerTileWidth, _innerTileHeight, Bitmap::PIXEL_FORMAT::LAB);

            auto rootData = (float *) rootLab.getData();
            auto leafData = (float *) leafLab.getData();

            if(_treeDepth > 1) {
                for (uint32_t level = _treeDepth - 2;; --level) {
                    rootBufferOffset = UINT64_MAX;
                    leafBufferOffset = UINT64_MAX;

                    actualLevelPxWidth = (actualLevelPxWidth + 1) >> 1;
                    actualLevelPxHeight = (actualLevelPxHeight + 1) >> 1;
                    actualLevelTileWidth = (actualLevelTileWidth + 1) >> 1;
                    actualLevelTileHeight = (actualLevelTileHeight + 1) >> 1;
                    uint64_t actualLevelTiles = actualLevelTileWidth * actualLevelTileHeight;

                    uint64_t levelFirstId = QuadTree::firstIdOfLevel(level);
                    uint64_t levelWidth = QuadTree::getWidthOfLevel(level);
                    uint64_t levelTiles = levelWidth * levelWidth;
                    uint64_t firstTileInLevelFileOffset = UINT64_MAX;

                    uint32_t iterTreeDepth = _treeDepth - level;
                    uint32_t iterLevel = iterTreeDepth - 1;
                    uint64_t iterLevelWidth = QuadTree::getWidthOfLevel(iterLevel);
                    uint64_t iterLevelTiles = iterLevelWidth * iterLevelWidth;

#ifdef DELTA_E_CALCULATOR_LOG_PROGRESS
                    auto start = std::chrono::high_resolution_clock::now();
                    uint8_t progress = 0;
                    uint64_t tilesLoaded = 0;

                    std::cout << "Calculating Delta-E for " << (actualLevelTiles) << " Tiles in Level " << level << std::endl;
                    std::cout << std::setw(3) << (int)progress << " %";
                    std::cout.flush();
#endif

                    for (uint64_t rootLevelRelId = levelTiles - 1;; --rootLevelRelId) {
                        uint64_t rootLevelAbsId = levelFirstId + rootLevelRelId;

                        uint64_t rootXCoord;
                        uint64_t rootYCoord;

                        QuadTree::getCoordinatesInLevel(rootLevelRelId, level, rootXCoord, rootYCoord);

                        if (rootXCoord < actualLevelTileWidth && rootYCoord < actualLevelTileHeight) {
                            uint64_t tileOffset = _offsetIndex->getOffset(rootLevelAbsId);

                            if (tileOffset < rootBufferOffset ||
                                tileOffset >= (rootBufferOffset + rootBufferByteSize)) {
                                if (firstTileInLevelFileOffset == UINT64_MAX) {
                                    firstTileInLevelFileOffset = tileOffset;
                                }

                                _file.seekg(_payloadOffset + tileOffset);
                                _file.read((char *) rootBuffer,
                                           std::min(rootBufferByteSize, firstTileInLevelFileOffset +
                                                                        (actualLevelTileWidth *
                                                                         actualLevelTileHeight *
                                                                         _tileByteSize) - tileOffset));

                                rootBufferOffset = tileOffset;
                            }

                            rootBitmap.setData(&rootBuffer[tileOffset - rootBufferOffset]);
                            rootLab.copyRectFrom(rootBitmap, _padding, _padding, 0, 0, _innerTileWidth,
                                                 _innerTileHeight);

                            for (uint64_t relIterId = iterLevelTiles - 1;; --relIterId) {
                                uint64_t leafLevelRelId = rootLevelRelId * iterLevelTiles + relIterId;
                                uint64_t leafLevelAbsId = leafLevelFirstId + leafLevelRelId;

                                if (_offsetIndex->exists(leafLevelAbsId)) {
                                    uint64_t tileOffset = _offsetIndex->getOffset(leafLevelAbsId);

                                    if (tileOffset < leafBufferOffset ||
                                        tileOffset >= (leafBufferOffset + leafBufferByteSize)) {
                                        _file.seekg(_payloadOffset + tileOffset);
                                        _file.read((char *) leafBuffer, std::min(leafBufferByteSize,
                                                                                 (_imageTileWidth * _imageTileHeight *
                                                                                  _tileByteSize) - tileOffset));

                                        leafBufferOffset = tileOffset;
                                    }

                                    leafBitmap.setData(&leafBuffer[tileOffset - leafBufferOffset]);
                                } else {
                                    leafBitmap.setData(blackTileBuffer);
                                }

                                leafLab.copyRectFrom(leafBitmap, _padding, _padding, 0, 0, _innerTileWidth,
                                                     _innerTileHeight);

                                uint64_t leafXCoord;
                                uint64_t leafYCoord;

                                QuadTree::getCoordinatesInLevel(relIterId, iterLevel, leafXCoord, leafYCoord);

                                size_t xOffsetInRoot = leafXCoord * _innerTileWidth / iterLevelWidth;
                                size_t yOffsetInRoot = leafYCoord * _innerTileHeight / iterLevelWidth;

                                float *rootPx;
                                float *leafPx;
                                float distL;
                                float distA;
                                float distB;

                                for (size_t y = 0; y < _innerTileHeight; ++y) {
                                    for (size_t x = 0; x < _innerTileWidth; ++x) {
                                        rootPx = &rootData[
                                                ((yOffsetInRoot + (y >> iterLevel)) * _innerTileWidth +
                                                 (xOffsetInRoot + (x >> iterLevel))) * 3];
                                        leafPx = &leafData[(y * _innerTileWidth + x) * 3];

                                        distL = rootPx[0] - leafPx[0];
                                        distA = rootPx[1] - leafPx[1];
                                        distB = rootPx[2] - leafPx[2];

                                        distBuffer[y * _innerTileWidth + x] = std::sqrt(
                                                distL * distL + distA * distA + distB * distB);
                                    }
                                }

                                size_t currentLevel = 1;
                                uint64_t currentId = relIterId;
                                size_t halfTileWidth = (_innerTileWidth >> 1);
                                size_t halfTileHeight = (_innerTileHeight >> 1);
                                auto lastLevelBuffer = distBuffer;
                                double *currentLevelBuffer;

                                do {
                                    currentLevelBuffer = &distBuffer[currentLevel * innerTilePxCount];
                                    size_t xOffset = (currentId & 1) * halfTileWidth;
                                    size_t yOffset = ((currentId & 2) >> 1) * halfTileHeight;
                                    double avrgDist;

                                    for (size_t y = 0; y < halfTileHeight; ++y) {
                                        for (size_t x = 0; x < halfTileWidth; ++x) {
                                            avrgDist =
                                                    ((lastLevelBuffer[(y << 1) * _innerTileWidth + (x << 1)] +
                                                      lastLevelBuffer[(y << 1) * _innerTileWidth + (x << 1) + 1]) / 2 +
                                                     (lastLevelBuffer[((y << 1) + 1) * _innerTileWidth + (x << 1)] +
                                                      lastLevelBuffer[((y << 1) + 1) * _innerTileWidth + (x << 1) +
                                                                      1]) /
                                                     2) / 2;

                                            currentLevelBuffer[(yOffset + y) * _innerTileWidth +
                                                               (xOffset + x)] = avrgDist;
                                        }
                                    }

                                    if (currentId == 0) {
                                        // iterated up as far as possible
                                        break;
                                    }

                                    currentId >>= 2;
                                    ++currentLevel;
                                    lastLevelBuffer = currentLevelBuffer;
                                } while ((currentId & 3) == 0);

#ifdef DELTA_E_CALCULATOR_LOG_PROGRESS
                                ++tilesLoaded;

                                auto currentProgress = (uint8_t)(tilesLoaded * 100 / (actualLevelTiles * iterLevelTiles));

                                if(currentProgress != progress) {
                                    progress = currentProgress;
                                    std::cout << '\r' << std::setw(3) << (int)progress << " %";
                                    std::cout.flush();
                                }
#endif

                                if (relIterId == 0) {
                                    // whole tile deltas are calculated
                                    size_t oldLen = innerTilePxCount;

                                    for (size_t len = (oldLen >> 1); len > 0; len = (oldLen >> 1)) {
                                        for (size_t i = 0; i < len; ++i) {
                                            if (i == (len - 1)) {
                                                if ((oldLen & 1) == 0) {
                                                    currentLevelBuffer[i] = currentLevelBuffer[i << 1] / 2 +
                                                                            currentLevelBuffer[(i << 1) + 1] / 2;
                                                } else {
                                                    currentLevelBuffer[i] = currentLevelBuffer[i << 1] / 3 +
                                                                            currentLevelBuffer[(i << 1) + 1] / 3 +
                                                                            currentLevelBuffer[(i << 1) + 2] / 3;
                                                }
                                            } else {
                                                currentLevelBuffer[i] = currentLevelBuffer[i << 1] / 2 +
                                                                        currentLevelBuffer[(i << 1) + 1] / 2;
                                            }
                                        }

                                        oldLen = len;
                                    }

                                    _cielabIndex->set(rootLevelAbsId, (float) currentLevelBuffer[0]);
                                    break;
                                }
                            }
                        }

                        if (rootLevelRelId == 0) {
                            break;
                        }
                    }

#ifdef DELTA_E_CALCULATOR_LOG_PROGRESS
                    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms)" << std::endl << std::endl;
#endif

                    if (level == 0) {
                        break;
                    }
                }
            }

            std::fstream indexFile(_fileName, std::ios_base::binary | std::ios_base::out | std::ios_base::in);
            indexFile.seekp(_cielabIndexOffset, std::ios_base::beg);
            _cielabIndex->writeToFile(indexFile);
            indexFile.close();

            delete[] distBuffer;
            delete[] rootBuffer;
            delete[] leafBuffer;
            delete[] blackTileBuffer;
        }
    }
}