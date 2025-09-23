// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/pre/Preprocessor.h>

namespace vt{
    namespace pre {
        bool Preprocessor::_isPowerOfTwo(size_t val) {
            return val != 0 && (val & (val - 1)) == 0;
        }

        void memset_volatile(volatile uint8_t *s, uint8_t val, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                s[i] = val;
            }
        }

        void Preprocessor::_deflate(size_t writeBufferSize) {
            size_t writeBufferTileSize = writeBufferSize / _destTileByteSize;
            writeBufferSize = writeBufferTileSize * _destTileByteSize;

            uint64_t writeBufferOffset = 0;
            uint64_t writeBufferFirstId = 0;
            uint64_t writeBufferLastId = 0;
            uint64_t offsetAfterLastTile = QuadTree::firstIdOfLevel(_treeDepth - 1) * _destTileByteSize;
            uint64_t lastWriteOffset = offsetAfterLastTile;
            uint64_t *idLookup = nullptr;

            if (_destLayout == AtlasFile::LAYOUT::RAW) {
                writeBufferOffset = offsetAfterLastTile -
                                    (QuadTree::firstIdOfLevel(_treeDepth - 1) % writeBufferTileSize) *
                                    _destTileByteSize;
                writeBufferFirstId = writeBufferOffset / _destTileByteSize;
            } else {
                writeBufferOffset = _imageTileWidth * _imageTileHeight * _destTileByteSize;
                idLookup = new uint64_t[writeBufferTileSize];
                writeBufferFirstId = offsetAfterLastTile / _destTileByteSize - 1;
                writeBufferLastId = writeBufferFirstId;

                for (size_t i = 0; i < writeBufferTileSize; ++i) {
                    idLookup[i] = writeBufferFirstId;
                }
            }

            size_t bufferSize = _destTileByteSize * 12;
            auto buffer = new uint8_t[bufferSize];
            auto writeBuffer = new uint8_t[writeBufferSize];

            Bitmap bufferBitmap0(_tileWidth, _tileHeight, _destPxFormat, buffer);
            Bitmap bufferBitmap1(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize]);
            Bitmap bufferBitmap2(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 2]);
            Bitmap bufferBitmap3(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 3]);
            Bitmap bufferBitmap4(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 4]);
            Bitmap bufferBitmap5(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 5]);
            Bitmap bufferBitmap6(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 6]);
            Bitmap bufferBitmap7(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 7]);
            Bitmap bufferBitmap8(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 8]);
            Bitmap bufferBitmap9(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 9]);
            Bitmap bufferBitmap10(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 10]);
            Bitmap writeBitmap(_tileWidth, _tileHeight, _destPxFormat, &buffer[_destTileByteSize * 11]);

            auto levelTileWidth = _imageTileWidth;
            auto levelTileHeight = _imageTileHeight;
            auto levelPixelWidth = _imageWidth;
            auto levelPixelHeight = _imageHeight;

            uint64_t currentOffset = 0;

            if (_destLayout != AtlasFile::LAYOUT::RAW) {
                auto firstId = QuadTree::firstIdOfLevel(_treeDepth - 1);
                currentOffset = _offsetIndex->getOffset(firstId);
                currentOffset += _destTileByteSize;
            }

            if (_treeDepth > 1) {
                for (size_t iterationLevel = _treeDepth - 2; /* iterationLevel > 0 */; --iterationLevel) {
                    levelPixelWidth = (levelPixelWidth + 1) >> 1;
                    levelPixelHeight = (levelPixelHeight + 1) >> 1;
                    auto iterationLevelTileWidth = (levelTileWidth + 1) >> 1;
                    auto iterationLevelTileHeight = (levelTileHeight + 1) >> 1;
                    auto iterLevelFullWidth = QuadTree::getWidthOfLevel(iterationLevel);
                    auto tilesInIterationLevel = iterLevelFullWidth * iterLevelFullWidth;
                    auto firstIdOfIterationLevel = QuadTree::firstIdOfLevel(iterationLevel);
                    auto firstIdOfCurrentLevel = QuadTree::firstIdOfLevel(iterationLevel + 1);

#ifdef PREPROCESSOR_LOG_PROGRESS
                    auto start = std::chrono::high_resolution_clock::now();
                    uint64_t tilesWritten = 0;
                    uint8_t progress = 0;

                    std::cout << "Deflating " << (levelTileWidth * levelTileHeight) << " Tiles in Level "
                              << (iterationLevel + 1) << std::endl;
                    std::cout << std::setw(3) << (int) progress << " %";
                    std::cout.flush();
#endif

                    for (uint64_t relIterationId =
                            tilesInIterationLevel - 1; /* relIterationId > 0 */; --relIterationId) {
                        uint64_t x;
                        uint64_t y;

                        QuadTree::getCoordinatesInLevel(relIterationId, iterationLevel, x, y);

                        if (x >= iterationLevelTileWidth || y >= iterationLevelTileHeight) {
                            continue;
                        }

                        uint64_t absIterationId = firstIdOfIterationLevel + relIterationId;
                        size_t bufferOffset = 0;

                        for (uint8_t relQuadId = 3;; --relQuadId) {
                            uint64_t relId = (relIterationId << 2) + relQuadId;
                            uint64_t absId = firstIdOfCurrentLevel + relId;

                            uint64_t len = 0;

                            uint64_t childX;
                            uint64_t childY;

                            QuadTree::getCoordinatesInLevel(relId, iterationLevel + 1, childX, childY);

                            if (childX < levelTileWidth && childY < levelTileHeight) {
                                len = _getTileById(absId, writeBuffer, writeBufferFirstId, writeBufferLastId, idLookup,
                                                   writeBufferTileSize, &buffer[bufferOffset]);
                            }

                            if (len == 0) {
                                std::memset(&buffer[bufferOffset], 0, _destTileByteSize);
                            }

                            if (relQuadId == 0) {
                                break;
                            }

                            bufferOffset += _destTileByteSize;
                        }

                        uint64_t relId = relIterationId << 2;

                        uint64_t relId1 = QuadTree::getNeighbour(relId, QuadTree::NEIGHBOUR::LEFT);
                        uint64_t relId2 = QuadTree::getNeighbour(relId1, QuadTree::NEIGHBOUR::BOTTOM);
                        uint64_t relId0 = QuadTree::getNeighbour(relId1, QuadTree::NEIGHBOUR::TOP);

                        uint64_t len = 0;

                        bufferOffset += _destTileByteSize;
                        if (relId1 != relId)
                            len = _getTileById(firstIdOfCurrentLevel + relId2, writeBuffer, writeBufferFirstId,
                                               writeBufferLastId, idLookup, writeBufferTileSize, &buffer[bufferOffset]);
                        if (len == 0) memset_volatile(&buffer[bufferOffset], 0, _destTileByteSize);

                        bufferOffset += _destTileByteSize;
                        if (relId1 != relId)
                            len = _getTileById(firstIdOfCurrentLevel + relId1, writeBuffer, writeBufferFirstId,
                                               writeBufferLastId, idLookup, writeBufferTileSize, &buffer[bufferOffset]);
                        if (len == 0) memset_volatile(&buffer[bufferOffset], 0, _destTileByteSize);

                        len = 0;

                        bufferOffset += _destTileByteSize;
                        if (relId1 != relId && relId0 != relId1)
                            len = _getTileById(firstIdOfCurrentLevel + relId0, writeBuffer, writeBufferFirstId,
                                               writeBufferLastId, idLookup, writeBufferTileSize, &buffer[bufferOffset]);
                        if (len == 0) memset_volatile(&buffer[bufferOffset], 0, _destTileByteSize);

                        relId0 = QuadTree::getNeighbour(relId, QuadTree::NEIGHBOUR::TOP);
                        relId1 = QuadTree::getNeighbour(relId0, QuadTree::NEIGHBOUR::RIGHT);

                        len = 0;

                        QuadTree::getCoordinatesInLevel(relId2, iterationLevel + 1, x, y);

                        bufferOffset += _destTileByteSize;
                        if (relId0 != relId)
                            len = _getTileById(firstIdOfCurrentLevel + relId1, writeBuffer, writeBufferFirstId,
                                               writeBufferLastId, idLookup, writeBufferTileSize, &buffer[bufferOffset]);
                        if (len == 0) memset_volatile(&buffer[bufferOffset], 0, _destTileByteSize);

                        bufferOffset += _destTileByteSize;
                        if (relId0 != relId)
                            len = _getTileById(firstIdOfCurrentLevel + relId0, writeBuffer, writeBufferFirstId,
                                               writeBufferLastId, idLookup, writeBufferTileSize, &buffer[bufferOffset]);
                        if (len == 0) memset_volatile(&buffer[bufferOffset], 0, _destTileByteSize);

                        relId0 = relIterationId;
                        relId1 = QuadTree::getNeighbour(relId0, QuadTree::NEIGHBOUR::RIGHT);
                        relId2 = QuadTree::getNeighbour(relId0, QuadTree::NEIGHBOUR::BOTTOM);

                        bufferOffset += _destTileByteSize;
                        len = _getTileById(firstIdOfIterationLevel + relId2, writeBuffer, writeBufferFirstId,
                                           writeBufferLastId, idLookup, writeBufferTileSize, &buffer[bufferOffset]);
                        if (len == 0) memset_volatile(&buffer[bufferOffset], 0, _destTileByteSize);

                        bufferOffset += _destTileByteSize;
                        len = _getTileById(firstIdOfIterationLevel + relId1, writeBuffer, writeBufferFirstId,
                                           writeBufferLastId, idLookup, writeBufferTileSize, &buffer[bufferOffset]);
                        if (len == 0) memset_volatile(&buffer[bufferOffset], 0, _destTileByteSize);

                        bufferOffset += _destTileByteSize;

                        size_t halfTileWidthInner = _innerTileWidth >> 1;
                        size_t halfTileHeightInner = _innerTileHeight >> 1;

                        QuadTree::getCoordinatesInLevel(relIterationId, iterationLevel, x, y);

                        std::memset((void *) &buffer[bufferOffset], 0, _destTileByteSize);

                        writeBitmap.deflateRectFrom(bufferBitmap0,
                                                    _padding, _padding,
                                                    _padding + halfTileWidthInner, _padding + (_innerTileHeight >> 1),
                                                    _innerTileWidth, _innerTileHeight);

                        writeBitmap.deflateRectFrom(bufferBitmap1,
                                                    _padding, _padding,
                                                    _padding, _padding + halfTileHeightInner,
                                                    _innerTileWidth, _innerTileHeight);

                        writeBitmap.deflateRectFrom(bufferBitmap2,
                                                    _padding, _padding,
                                                    _padding + halfTileWidthInner, _padding,
                                                    _innerTileWidth, _innerTileHeight);

                        writeBitmap.deflateRectFrom(bufferBitmap3,
                                                    _padding, _padding,
                                                    _padding, _padding,
                                                    _innerTileWidth, _innerTileHeight);

                        if (x == 0) {
                            writeBitmap.smearHorizontal(_padding, _padding,
                                                        0, _padding,
                                                        _padding, _innerTileHeight);
                        } else {
                            // pad lower left side
                            writeBitmap.deflateRectFrom(bufferBitmap4,
                                                        _padding + _innerTileWidth - (_padding << 1), _padding,
                                                        0, _padding + halfTileHeightInner,
                                                        _padding << 1, _innerTileHeight);

                            // pad upper left side
                            writeBitmap.deflateRectFrom(bufferBitmap5,
                                                        _padding + _innerTileWidth - (_padding << 1), _padding,
                                                        0, _padding,
                                                        _padding << 1, _innerTileHeight);

                            if (y > 0) {
                                // pad upper left corner
                                writeBitmap.deflateRectFrom(bufferBitmap6,
                                                            _padding + _innerTileWidth - (_padding << 1),
                                                            _padding + _innerTileHeight - (_padding << 1),
                                                            0, 0,
                                                            _padding << 1, _padding << 1);
                            }
                        }

                        if (y == 0) {
                            // pad top side
                            writeBitmap.smearVertical(0, _padding,
                                                      0, 0,
                                                      _padding + _innerTileWidth, _padding);
                        } else {
                            // pad right top side
                            writeBitmap.deflateRectFrom(bufferBitmap7,
                                                        _padding, _padding + _innerTileHeight - (_padding << 1),
                                                        _padding + halfTileWidthInner, 0,
                                                        _innerTileWidth, _padding << 1);

                            // pad left top side
                            writeBitmap.deflateRectFrom(bufferBitmap8,
                                                        _padding, _padding + _innerTileHeight - (_padding << 1),
                                                        _padding, 0,
                                                        _innerTileWidth, _padding << 1);

                            if (x == 0) {
                                // pad upper left corner
                                writeBitmap.smearHorizontal(_padding, 0,
                                                            0, 0,
                                                            _padding, _padding);
                            }
                        }

                        bool xIsLast = x == (iterationLevelTileWidth - 1);
                        bool yIsLast = y == (iterationLevelTileHeight - 1);

                        size_t padWidth = _padding;
                        size_t padHeight = _padding;

                        if (xIsLast) {
                            padWidth += ((levelPixelWidth - 1) % _innerTileWidth) + 1;
                        } else {
                            padWidth += _innerTileWidth;
                        }

                        if (yIsLast) {
                            padHeight += ((levelPixelHeight - 1) % _innerTileHeight) + 1;
                        } else {
                            padHeight += _innerTileHeight;
                        }

                        if (yIsLast) {
                            // pad bottom side
                            writeBitmap.smearVertical(0, padHeight - 1,
                                                      0, padHeight,
                                                      padWidth, _padding);

                            uint8_t transPx[4] = {0x00, 0x00, 0x00, 0x00};

                            writeBitmap.fillRect(transPx, Bitmap::PIXEL_FORMAT::RGBA8, 0, padHeight + _padding,
                                                 padWidth + _padding, _tileHeight - padHeight - _padding);
                        } else {
                            // pad bottom side
                            writeBitmap.copyRectFrom(bufferBitmap9,
                                                     0, _padding,
                                                     0, _padding + _innerTileHeight,
                                                     padWidth + _padding, _padding);
                        }

                        if (xIsLast) {
                            // pad right side
                            writeBitmap.smearHorizontal(padWidth - 1, 0,
                                                        padWidth, 0,
                                                        _padding, padHeight + _padding);

                            uint8_t transPx[4] = {0x00, 0x00, 0x00, 0x00};

                            writeBitmap.fillRect(transPx, Bitmap::PIXEL_FORMAT::RGBA8, padWidth + _padding, 0,
                                                 _tileWidth - padWidth - _padding, _tileHeight);
                        } else {
                            // pad right side
                            writeBitmap.copyRectFrom(bufferBitmap10,
                                                     _padding, 0,
                                                     _padding + _innerTileWidth, 0,
                                                     _padding, padHeight + _padding);
                        }

                        if (_destLayout == AtlasFile::LAYOUT::RAW) {
                            currentOffset = absIterationId * _destTileByteSize;
                            _offsetIndex->set(absIterationId, currentOffset, _destTileByteSize);

                            while (currentOffset < writeBufferOffset) {
                                std::memset(writeBuffer, 0x00, lastWriteOffset - writeBufferOffset);

                                _destPayloadFile.seekp(_destPayloadOffset + writeBufferOffset);
                                _destPayloadFile.write((char *) writeBuffer, std::min(writeBufferSize,
                                                                                      offsetAfterLastTile -
                                                                                      writeBufferOffset));

                                lastWriteOffset = writeBufferOffset;
                                writeBufferOffset -= writeBufferSize;
                                writeBufferFirstId -= writeBufferTileSize;
                            }

                            std::memset(&writeBuffer[currentOffset - writeBufferOffset + _destTileByteSize], 0x00,
                                        lastWriteOffset - currentOffset - _destTileByteSize);
                            std::memcpy(&writeBuffer[currentOffset - writeBufferOffset], (char *) &buffer[bufferOffset],
                                        _destTileByteSize);
                            lastWriteOffset = currentOffset;
                        } else {
                            if ((currentOffset + _destTileByteSize) > (writeBufferOffset + writeBufferSize)) {
                                _destPayloadFile.seekp(_destPayloadOffset + writeBufferOffset);
                                _destPayloadFile.write((char *) writeBuffer, writeBufferSize);

                                writeBufferOffset += writeBufferSize;
                            }

                            //std::cout << currentOffset << std::endl;
                            _offsetIndex->set(absIterationId, currentOffset, _destTileByteSize);

                            std::memcpy(&writeBuffer[currentOffset - writeBufferOffset], (char *) &buffer[bufferOffset],
                                        _destTileByteSize);
                            writeBufferLastId = idLookup[(((currentOffset - writeBufferOffset) / _destTileByteSize) + 1) % writeBufferTileSize];
                            writeBufferFirstId = absIterationId;
                            idLookup[(currentOffset - writeBufferOffset) / _destTileByteSize] = absIterationId;
                            currentOffset += _destTileByteSize;
                        }

#ifdef PREPROCESSOR_LOG_PROGRESS
                        ++tilesWritten;
                        auto currentProgress = (uint8_t) (tilesWritten * 100 / iterationLevelTileWidth /
                                                          iterationLevelTileHeight);

                        if (currentProgress != progress) {
                            progress = currentProgress;
                            std::cout << '\r' << std::setw(3) << (int) progress << " %";
                            std::cout.flush();
                        }
#endif

                        if (relIterationId == 0) {
                            break;
                        }
                    }

                    levelTileWidth = iterationLevelTileWidth;
                    levelTileHeight = iterationLevelTileHeight;

#ifdef PREPROCESSOR_LOG_PROGRESS
                    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start).count() << " ms)" << std::endl
                              << std::endl;
                    std::cout.flush();
#endif

                    if (iterationLevel == 0) {
                        break;
                    }
                }
            }

            _destPayloadFile.seekp(_destPayloadOffset + writeBufferOffset);

            if (_destLayout == AtlasFile::RAW) {
                _destPayloadFile.write((char *) writeBuffer,
                                       std::min(writeBufferSize, offsetAfterLastTile - writeBufferOffset));
            } else if (_destLayout == AtlasFile::PACKED) {
                _destPayloadFile.write((char *) writeBuffer, (currentOffset - writeBufferOffset));
            }

            _destPayloadFile.flush();

            _destIndexFile->seekp(_destOffsetIndexOffset);
            _offsetIndex->writeToFile(*_destIndexFile);
            _destIndexFile->seekp(_destCielabIndexOffset);
            _cielabIndex->writeToFile(*_destIndexFile);

            delete[] buffer;
            delete[] writeBuffer;
            delete[] idLookup;
        }

        size_t Preprocessor::_getTileById(uint64_t id, const uint8_t *buffer, uint64_t firstIdInBuffer,
                                          uint64_t lastIdInBuffer, const uint64_t *idLookup, size_t bufferTileLen,
                                          uint8_t *out) {
            size_t len = _getBufferedTileById(id, buffer, firstIdInBuffer, lastIdInBuffer, idLookup, bufferTileLen,
                                              out);

            if (len == 0) {
                len = _loadTileById(id, out);
            }

            return len;
        }

        size_t Preprocessor::_loadTileById(uint64_t id, uint8_t *out) {
            uint64_t offset;
            uint64_t len = _destTileByteSize;

            if (_destLayout == AtlasFile::LAYOUT::RAW) {
                offset = id * len;
            } else {
                if (!_offsetIndex->exists(id)) {
                    return 0;
                }

                offset = _offsetIndex->getOffset(id);
            }

            _destPayloadFile.clear();
            _destPayloadFile.seekg(_destPayloadOffset + offset, std::ios_base::beg);
            _destPayloadFile.read((char *) out, len);

            if (!_destPayloadFile.good()) {
                throw std::runtime_error("Cannot read Tiles from File.");
            }

            return (size_t) len;
        }

        size_t Preprocessor::_getBufferedTileById(uint64_t id, const uint8_t *buffer, uint64_t firstIdInBuffer,
                                                  uint64_t lastIdInBuffer, const uint64_t *idLookup,
                                                  size_t bufferTileLen, uint8_t *out) {
            size_t len = 0;

            if (_destLayout == AtlasFile::LAYOUT::RAW) {
                if (id >= firstIdInBuffer && id < firstIdInBuffer + bufferTileLen) {
                    len = _destTileByteSize;
                    std::memcpy(out, (char *) &buffer[(id - firstIdInBuffer) * _destTileByteSize], _destTileByteSize);
                }
            } else {
                size_t offset = bufferTileLen;

                if (id >= firstIdInBuffer && id <= lastIdInBuffer) {
                    for (size_t i = 0; i < bufferTileLen; ++i) {
                        if (idLookup[i] == id) {
                            offset = i;
                            break;
                        }
                    }
                }

                if (offset != bufferTileLen) {
                    len = _destTileByteSize;
                    std::memcpy(out, (char *) &buffer[offset * _destTileByteSize], _destTileByteSize);
                }
            }

            return len;
        }

        Preprocessor::Preprocessor(const std::string &srcFileName,
                                   Bitmap::PIXEL_FORMAT srcPxFormat,
                                   size_t imageWidth,
                                   size_t imageHeight) {
            _destHeaderFile = nullptr;
            _destIndexFile = nullptr;
            _destCombined = DEST_COMBINED::NONE;

            _srcFileName = srcFileName;
            _srcPxFormat = srcPxFormat;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;

            //_srcFile.rdbuf()->pubsetbuf(nullptr, 0);
            _srcFile.open(srcFileName, std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);

            if (!_srcFile.is_open()) {
                throw std::runtime_error("Could not open File \"" + srcFileName + "\".");
            }

            size_t pxSize = Bitmap::pixelSize(srcPxFormat);
            uint64_t expSrcFileSize = imageWidth * imageHeight * pxSize;
            _srcFileSize = (uint64_t) _srcFile.tellg();

            _offsetIndex = nullptr;
            _cielabIndex = nullptr;

            if (_srcFileSize != expSrcFileSize) {
                throw std::runtime_error("File \"" + srcFileName + "\" expected to be of Size " +
                                         std::to_string(expSrcFileSize) + " Bytes, actually has " +
                                         std::to_string(_srcFileSize) + " Bytes.");
            }
        }

        Preprocessor::~Preprocessor() {
            _srcFile.close();
            _destPayloadFile.close();

            delete _offsetIndex;
            delete _cielabIndex;

            if (_destCombined == DEST_COMBINED::NOT_COMBINED) {
                delete _destHeaderFile;
                delete _destIndexFile;
            }
        }

        void Preprocessor::_putLE(uint64_t num, uint8_t *out) {
            out[0] = (uint8_t) (num & 0xff);
            out[1] = (uint8_t) ((num >> 8) & 0xff);
            out[2] = (uint8_t) ((num >> 16) & 0xff);
            out[3] = (uint8_t) ((num >> 24) & 0xff);
            out[4] = (uint8_t) ((num >> 32) & 0xff);
            out[5] = (uint8_t) ((num >> 40) & 0xff);
            out[6] = (uint8_t) ((num >> 48) & 0xff);
            out[7] = (uint8_t) ((num >> 56) & 0xff);
        }

        void Preprocessor::_putPixelFormat(Bitmap::PIXEL_FORMAT pxFormat, uint8_t *out) {
            switch (pxFormat) {
                case Bitmap::PIXEL_FORMAT::R8:
                    out[0] = 1;
                    break;
                case Bitmap::PIXEL_FORMAT::RGB8:
                    out[0] = 2;
                    break;
                case Bitmap::PIXEL_FORMAT::RGBA8:
                    out[0] = 3;
                    break;
                default:
                    throw std::runtime_error("Trying to save unknown pixel format.");
            }
        }

        void Preprocessor::_putFileFormat(AtlasFile::LAYOUT fileFormat, uint8_t *out) {
            switch (fileFormat) {
                case AtlasFile::LAYOUT::RAW:
                    out[0] = 1;
                    break;
                case AtlasFile::LAYOUT::PACKED:
                    out[0] = 2;
                    break;
                default:
                    throw std::runtime_error("Trying to save unknown file format.");
            }
        }

        void Preprocessor::_writeHeader() {
            uint8_t data[_HEADER_SIZE];

            data[0] = 'A';
            data[1] = 'T';
            data[2] = 'L';
            data[3] = 'A';
            data[4] = 'S';

            _putLE(_imageWidth, &data[5]);
            _putLE(_imageHeight, &data[13]);
            _putLE(_tileWidth, &data[21]);
            _putLE(_tileHeight, &data[29]);
            _putLE(_padding, &data[37]);

            _putPixelFormat(_destPxFormat, &data[45]);
            _putFileFormat(_destLayout, &data[46]);

            _putLE(_destOffsetIndexOffset, &data[47]);
            _putLE(_destCielabIndexOffset, &data[55]);
            _putLE(_destPayloadOffset, &data[63]);

            _destHeaderFile->seekp(_destHeaderOffset);
            _destHeaderFile->write((char *) data, _HEADER_SIZE);
            _destHeaderFile->flush();
        }

        void Preprocessor::setOutput(const std::string &destFileName,
                                     Bitmap::PIXEL_FORMAT destPxFormat,
                                     AtlasFile::LAYOUT format,
                                     size_t tileWidth,
                                     size_t tileHeight,
                                     size_t padding,
                                     bool combine) {
            _destFileName = destFileName;
            _destPxFormat = destPxFormat;
            _destLayout = format;
            _tileWidth = tileWidth;
            _tileHeight = tileHeight;
            _padding = padding;

            if(_destLayout != AtlasFile::LAYOUT::PACKED){
                throw std::runtime_error("All formats but packed are deprecated.");
            }

            if (!_isPowerOfTwo(tileWidth)) {
                throw std::runtime_error("Tile Width needs to be a Power of 2.");
            }

            if (!_isPowerOfTwo(tileHeight)) {
                throw std::runtime_error("Tile Height needs to be a Power of 2.");
            }

            size_t minTileSize = std::min(tileWidth, tileHeight);

            if (padding >= (minTileSize >> 1)) {
                throw std::runtime_error("Padding needs to be smaller than " + std::to_string(minTileSize >> 1) + ".");
            }

            _innerTileWidth = _tileWidth - (_padding << 1);
            _innerTileHeight = _tileHeight - (_padding << 1);
            _imageTileWidth = (_imageWidth + _innerTileWidth - 1) / _innerTileWidth;
            _imageTileHeight = (_imageHeight + _innerTileHeight - 1) / _innerTileHeight;
            _treeDepth = QuadTree::getDepth(_imageTileWidth, _imageTileHeight);

            _destTileByteSize = _tileWidth * _tileHeight * Bitmap::pixelSize(_destPxFormat);

            if (_destCombined != DEST_COMBINED::NONE) {
                _destPayloadFile.close();
                _destPayloadFile.clear();

                if (_destCombined == DEST_COMBINED::NOT_COMBINED) {
                    _destHeaderFile->close();
                    delete _destHeaderFile;
                    _destHeaderFile = nullptr;

                    _destIndexFile->close();
                    delete _destIndexFile;
                    _destIndexFile = nullptr;
                }
            }

            if (combine) {
                _destCombined = DEST_COMBINED::COMBINED;
                _destPayloadFile.open(destFileName + ".atlas", std::ios::in | std::ios::out | std::ios::binary |
                                                               std::ios::trunc);
                //_destPayloadFile.rdbuf()->pubsetbuf(nullptr, 0);

                if (!_destPayloadFile.is_open()) {
                    throw std::runtime_error("Could not open File \"" + destFileName + ".atlas\".");
                }

                _destHeaderFile = &_destPayloadFile;
                _destIndexFile = &_destPayloadFile;

                auto totalTileCount = QuadTree::firstIdOfLevel(_treeDepth);

                _destHeaderOffset = 0;
                _destOffsetIndexOffset = _destHeaderOffset + _HEADER_SIZE;
                _offsetIndex = new vt::pre::OffsetIndex(totalTileCount, format);
                _destCielabIndexOffset = _destOffsetIndexOffset + _offsetIndex->getByteSize();
                _cielabIndex = new vt::pre::CielabIndex(totalTileCount);
                _destPayloadOffset = _destCielabIndexOffset + _cielabIndex->getByteSize();

                /*std::ifstream indexFile(destFileName + ".atlas");

                indexFile.seekg(_destOffsetIndexOffset);
                _offsetIndex->readFromFile(indexFile);

                indexFile.close();*/
            } else {
                _destCombined = DEST_COMBINED::NOT_COMBINED;
                //_destPayloadFile.rdbuf()->pubsetbuf(nullptr, 0);
                _destPayloadFile.open(destFileName + ".atlas.data", std::ios::in | std::ios::out | std::ios::binary |
                                                                    std::ios::trunc);

                if (!_destPayloadFile.is_open()) {
                    throw std::runtime_error("Could not open File \"" + destFileName + ".atlas.data\".");
                }

                _destHeaderFile = new std::fstream(destFileName + ".atlas.header", std::ios::in | std::ios::out |
                                                                                   std::ios::binary | std::ios::trunc);
                //_destHeaderFile->rdbuf()->pubsetbuf(nullptr, 0);

                if (!_destHeaderFile->is_open()) {
                    throw std::runtime_error("Could not open File \"" + destFileName + ".atlas.header\".");
                }

                _destIndexFile = new std::fstream(destFileName + ".atlas.index", std::ios::in | std::ios::out |
                                                                                 std::ios::binary | std::ios::trunc);
                //_destIndexFile->rdbuf()->pubsetbuf(nullptr, 0);

                if (!_destIndexFile->is_open()) {
                    throw std::runtime_error("Could not open File \"" + destFileName + ".atlas.index\".");
                }

                auto totalTileCount = QuadTree::firstIdOfLevel(_treeDepth);

                _destHeaderOffset = 0;
                _destOffsetIndexOffset = 0;
                _offsetIndex = new vt::pre::OffsetIndex(totalTileCount, format);
                _destCielabIndexOffset = _destOffsetIndexOffset + _offsetIndex->getByteSize();
                _cielabIndex = new vt::pre::CielabIndex(totalTileCount);
                _destPayloadOffset = 0;
            }
        }

        void Preprocessor::_extract(size_t bufferTileWidth, size_t writeBufferSize) {
            if (!_isPowerOfTwo(bufferTileWidth)) {
                throw std::runtime_error("Cache Width needs to be a Power of 2.");
            }

#ifdef PREPROCESSOR_LOG_PROGRESS
            auto start = std::chrono::high_resolution_clock::now();
            uint64_t tilesWritten = 0;
            uint8_t progress = 0;

            std::cout << "Extracting " << (_imageTileWidth * _imageTileHeight) << " Tiles: " << std::endl;
            std::cout << std::setw(3) << (int) progress << " %";
            std::cout.flush();
#endif

            auto srcPxSize = Bitmap::pixelSize(_srcPxFormat);

            if (bufferTileWidth > QuadTree::getWidthOfLevel(_treeDepth - 1)) {
                bufferTileWidth = QuadTree::getWidthOfLevel(_treeDepth - 1);
            }

            auto bufferTileHeight = bufferTileWidth;

            auto bufferPxWidthInner = bufferTileWidth * _innerTileWidth;
            auto bufferPxHeightInner = bufferTileHeight * _innerTileHeight;
            auto bufferPxWidth = bufferPxWidthInner + (_padding << 1);
            auto bufferPxHeight = bufferPxHeightInner + (_padding << 1);

            auto bufferSize = bufferPxWidth * bufferPxHeight * srcPxSize;
            auto buffer = new uint8_t[bufferSize];

            size_t writeBufferTileSize = writeBufferSize / _destTileByteSize;
            writeBufferSize = writeBufferTileSize * _destTileByteSize;

            auto outTile = new uint8_t[_destTileByteSize];
            auto writeBuffer = new uint8_t[writeBufferSize];
            uint64_t writeBufferOffset = 0;
            uint64_t offsetAfterLastTile = QuadTree::firstIdOfLevel(_treeDepth) * _destTileByteSize;
            uint64_t lastWriteOffset = offsetAfterLastTile;

            if (_destLayout == AtlasFile::LAYOUT::RAW) {
                uint64_t tilesInFinestLevel =
                        QuadTree::getWidthOfLevel(_treeDepth - 1) * QuadTree::getWidthOfLevel(_treeDepth - 1);
                writeBufferOffset =
                        offsetAfterLastTile - (tilesInFinestLevel % writeBufferTileSize) * _destTileByteSize;
            }

            Bitmap bufferBitmap(bufferPxWidth, bufferPxHeight, _srcPxFormat, buffer);
            Bitmap writeBitmap(_tileWidth, _tileHeight, _destPxFormat, outTile);

            size_t finestLevel = _treeDepth - 1;
            size_t bufferLevel = QuadTree::getDepth(bufferTileWidth, bufferTileHeight) - 1;
            size_t iterationLevel = finestLevel - bufferLevel;

            auto iterLevelWidth = QuadTree::getWidthOfLevel(iterationLevel);
            uint64_t tilesToIterate = iterLevelWidth * iterLevelWidth;

            size_t iterTileWidth = (_imageTileWidth + bufferTileWidth - 1) / bufferTileWidth;
            size_t iterTileHeight = (_imageTileHeight + bufferTileHeight - 1) / bufferTileHeight;

            auto bufLevelWidth = QuadTree::getWidthOfLevel(bufferLevel);
            uint64_t tilesInBuffer = bufLevelWidth * bufLevelWidth;
            uint64_t currentOffset = 0;

            auto firstId = QuadTree::firstIdOfLevel(finestLevel);

            for (uint64_t relIterationId = tilesToIterate - 1; /*relIterationId >= 0*/; --relIterationId) {
                uint64_t x;
                uint64_t y;

                QuadTree::getCoordinatesInLevel(relIterationId, iterationLevel, x, y);

                if (x >= iterTileWidth || y >= iterTileHeight) {
                    if (_destLayout == AtlasFile::LAYOUT::RAW) {
                        currentOffset = (firstId + relIterationId * tilesInBuffer) * _destTileByteSize;
                        auto dataLen = (uint64_t) bufferTileWidth * bufferTileHeight * _destTileByteSize;
                        auto data = new uint8_t[dataLen];

                        std::memset(data, 0x00, dataLen);

                        _destPayloadFile.seekp(_destPayloadOffset + currentOffset);
                        _destPayloadFile.write((char *) data, dataLen);

                        delete data;
                    }

                    continue;
                }

                size_t offsetX = (size_t) x * bufferPxWidthInner;
                size_t offsetY = (size_t) y * bufferPxHeightInner;

                size_t readWidth = bufferPxWidth;
                size_t readHeight = bufferPxHeight;

                size_t offsetBufferX = 0;
                size_t offsetBufferY = 0;

                if (offsetX < _padding) {
                    offsetBufferX = _padding - offsetX;
                    readWidth -= offsetBufferX;
                    offsetX = 0;
                } else {
                    offsetX -= _padding;
                }

                if (offsetY < _padding) {
                    offsetBufferY = _padding - offsetY;
                    readHeight -= offsetBufferY;
                    offsetY = 0;
                } else {
                    offsetY -= _padding;
                }

                if ((offsetX + readWidth) > _imageWidth) {
                    readWidth = _imageWidth - offsetX;
                }

                if ((offsetY + readHeight) > _imageHeight) {
                    readHeight = _imageHeight - offsetY;
                }

                auto cachePtr = &buffer[offsetBufferY * bufferPxWidth * srcPxSize + offsetBufferX * srcPxSize];
                uint64_t fileOffset = offsetY * _imageWidth * srcPxSize + offsetX * srcPxSize;

                for (size_t line = 0; line < readHeight; ++line) {
                    _srcFile.seekg(fileOffset);
                    _srcFile.read((char *) cachePtr, readWidth * srcPxSize);

                    if (!_srcFile.good()) {
                        throw std::runtime_error("Cannot read from File.");
                    }

                    fileOffset += _imageWidth * srcPxSize;
                    cachePtr = &cachePtr[bufferPxWidth * srcPxSize];
                }

                // pad left side
                bufferBitmap.smearHorizontal(offsetBufferX, offsetBufferY,
                                             0, offsetBufferY,
                                             offsetBufferX, readHeight);

                // pad right side
                bufferBitmap.smearHorizontal(offsetBufferX + readWidth - 1, offsetBufferY,
                                             offsetBufferX + readWidth, offsetBufferY,
                                             std::min<size_t>(_padding, bufferPxWidth - offsetBufferX - readWidth),
                                             readHeight);

                // pad top side
                bufferBitmap.smearVertical(0, offsetBufferY,
                                           0, 0,
                                           bufferPxWidth, offsetBufferY);

                // pad bottom side
                bufferBitmap.smearVertical(0, offsetBufferY + readHeight - 1,
                                           0, offsetBufferY + readHeight,
                                           bufferPxWidth,
                                           std::min<size_t>(_padding, bufferPxHeight - offsetBufferY - readHeight));

                for (auto relBufferId = (size_t) (tilesInBuffer - 1); /*relBufferId >= tilesInBuffer*/; --relBufferId) {
                    auto relId = relIterationId * tilesInBuffer + relBufferId;
                    auto absId = firstId + relId;

                    uint64_t bufferTileX;
                    uint64_t bufferTileY;

                    QuadTree::getCoordinatesInLevel(relBufferId, bufferLevel, bufferTileX, bufferTileY);

                    auto absTileX = x * bufferTileWidth + bufferTileX;
                    auto absTileY = y * bufferTileHeight + bufferTileY;

                    if (absTileX >= _imageTileWidth || absTileY >= _imageTileHeight) {
                        continue;
                    }

                    writeBitmap.copyRectFrom(bufferBitmap,
                                             (size_t) bufferTileX * _innerTileWidth,
                                             (size_t) bufferTileY * _innerTileHeight,
                                             0, 0,
                                             _tileWidth, _tileHeight);

                    size_t padWidth = _tileWidth;

                    if (absTileX == (_imageTileWidth - 1)) {
                        padWidth = ((_imageWidth - 1) % _innerTileWidth) + 1 + (_padding << 1);
                    }

                    if (absTileY == (_imageTileHeight - 1)) {
                        uint8_t transPx[] = {0x00, 0x00, 0x00, 0x00};

                        writeBitmap.fillRect(transPx, Bitmap::PIXEL_FORMAT::RGBA8,
                                             0, ((_imageHeight - 1) % _innerTileHeight) + 1 + (_padding << 1),
                                             padWidth,
                                             _tileHeight - ((_imageHeight - 1) % _innerTileHeight) - 1 -
                                             (_padding << 1));
                    }

                    if (absTileX == (_imageTileWidth - 1)) {
                        uint8_t transPx[] = {0x00, 0x00, 0x00, 0x00};

                        writeBitmap.fillRect(transPx, Bitmap::PIXEL_FORMAT::RGBA8, padWidth, 0, _tileWidth - padWidth,
                                             _tileHeight);
                    }

                    _offsetIndex->set(absId, currentOffset, _destTileByteSize);

                    if (_destLayout == AtlasFile::LAYOUT::RAW) {
                        currentOffset = _destTileByteSize * absId;

                        while (currentOffset < writeBufferOffset) {
                            std::memset(writeBuffer, 0x00, lastWriteOffset - writeBufferOffset);

                            _destPayloadFile.seekp(_destPayloadOffset + writeBufferOffset);
                            _destPayloadFile.write((char *) writeBuffer,
                                                   std::min(writeBufferSize, offsetAfterLastTile - writeBufferOffset));

                            lastWriteOffset = writeBufferOffset;
                            writeBufferOffset -= writeBufferSize;
                        }

                        std::memset(&writeBuffer[currentOffset - writeBufferOffset + _destTileByteSize], 0x00,
                                    lastWriteOffset - currentOffset - _destTileByteSize);
                        std::memcpy(&writeBuffer[currentOffset - writeBufferOffset], outTile, _destTileByteSize);
                        lastWriteOffset = currentOffset;
                    } else {
                        if ((currentOffset + _destTileByteSize) > (writeBufferOffset + writeBufferSize)) {
                            _destPayloadFile.seekp(_destPayloadOffset + writeBufferOffset);
                            _destPayloadFile.write((char *) writeBuffer, (currentOffset - writeBufferOffset));
                            _destPayloadFile.flush();

                            writeBufferOffset = currentOffset;
                        }

                        std::memcpy(&writeBuffer[currentOffset - writeBufferOffset], outTile, _destTileByteSize);
                        currentOffset += _destTileByteSize;
                    }

#ifdef PREPROCESSOR_LOG_PROGRESS
                    ++tilesWritten;
                    auto currentProgress = (uint8_t) (tilesWritten * 100 / _imageTileWidth / _imageTileHeight);

                    if (currentProgress != progress) {
                        progress = currentProgress;
                        std::cout << '\r' << std::setw(3) << (int) progress << " %";
                        std::cout.flush();
                    }
#endif

                    if (relBufferId == 0) {
                        break;
                    }
                }

                if (relIterationId == 0) {
                    break;
                }
            }

#ifdef PREPROCESSOR_LOG_PROGRESS
            std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start).count() << " ms)" << std::endl << std::endl;
#endif

            _destPayloadFile.seekp(_destPayloadOffset + writeBufferOffset);

            if (_destLayout == AtlasFile::RAW) {
                _destPayloadFile.write((char *) writeBuffer,
                                       std::min(writeBufferSize, offsetAfterLastTile - writeBufferOffset));
            } else if (_destLayout == AtlasFile::PACKED) {
                _destPayloadFile.write((char *) writeBuffer, (currentOffset - writeBufferOffset));
            }

            _destPayloadFile.flush();
            _destIndexFile->seekp(_destOffsetIndexOffset);
            _offsetIndex->writeToFile(*_destIndexFile);
            _destIndexFile->seekp(_destCielabIndexOffset);
            _cielabIndex->writeToFile(*_destIndexFile);

            delete[] buffer;
            delete[] outTile;
            delete[] writeBuffer;
        }

        void Preprocessor::run(size_t maxMemory) {
#ifdef PREPROCESSOR_LOG_PROGRESS
            auto start = std::chrono::high_resolution_clock::now();

            std::cout << "Preprocessing \"" << _srcFileName << "\"" << std::endl;

            if (_destCombined == DEST_COMBINED::COMBINED) {
                std::cout << "--> \"" << _destFileName << ".atlas\"" << std::endl;
            } else {
                std::cout << "--> \"" << _destFileName << ".atlas.header\"" << std::endl;
                std::cout << "--> \"" << _destFileName << ".atlas.index\"" << std::endl;
                std::cout << "--> \"" << _destFileName << ".atlas.data\"" << std::endl;
            }

            std::cout << std::endl;

            std::cout << "Writing Header ... ";
#endif
            _writeHeader();

#ifdef PREPROCESSOR_LOG_PROGRESS
            std::cout << "Done" << std::endl << std::endl;
#endif

            size_t srcTileSize = _tileWidth * _tileHeight * Bitmap::pixelSize(_srcPxFormat);

            auto bufferSideLen = (size_t) std::sqrt(maxMemory / srcTileSize);
            bufferSideLen = (size_t) 1 << ((size_t) std::log2(bufferSideLen));

#ifdef PREPROCESSOR_LOG_PROGRESS
            std::cout << "Readbuffer Size: " << bufferSideLen << "x" << bufferSideLen << " Tiles\n";
            std::cout << "Writebuffer Size: " << (maxMemory - bufferSideLen * bufferSideLen * srcTileSize) << " Bytes\n"
                      << std::endl;
#endif

            _extract(bufferSideLen, maxMemory - bufferSideLen * bufferSideLen * srcTileSize);
            _deflate(maxMemory);
            //_calcDeltaE(maxMemory);

#ifdef  PREPROCESSOR_LOG_PROGRESS
            std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl << std::endl;
#endif
        }

        void Preprocessor::_loadToSeqBuffer(uint8_t *buffer, uint64_t id, size_t count) {
            switch (_destLayout) {
                case AtlasFile::LAYOUT::RAW:
                    break;
                case AtlasFile::LAYOUT::PACKED:
                    uint64_t destId = 0;

                    if (id > count) {
                        destId = id + 1 - count;
                    }

                    uint64_t firstAvailableId = UINT64_MAX;
                    uint64_t lastAvailableId = UINT64_MAX;
                    uint64_t countAvailableIds = 0;

                    for (uint64_t i = id;; --i) {
                        if (_offsetIndex->exists(i)) {
                            if (firstAvailableId == UINT64_MAX) {
                                firstAvailableId = i;
                            }

                            lastAvailableId = i;
                            ++countAvailableIds;
                        }

                        if (i == destId) {
                            break;
                        }
                    }

                    uint64_t offset = _offsetIndex->getOffset(firstAvailableId);

                    _destPayloadFile.seekg(_destPayloadOffset + offset);
                    _destPayloadFile.read((char *) buffer, countAvailableIds * _destTileByteSize);

                    auto tile = &buffer[(countAvailableIds - 1) * _destPayloadOffset];

                    for (uint64_t i = 0; i < count; ++i) {
                        auto slot = count - i - 1;
                        auto absId = id - slot;

                        if (slot > id || absId > firstAvailableId || absId < lastAvailableId ||
                            !_offsetIndex->exists(absId)) {
                            std::memset(&buffer[slot * _destTileByteSize], 0, _destTileByteSize);
                        } else {
                            std::memcpy(&buffer[slot * _destTileByteSize], tile, _destTileByteSize);
                            tile = (uint8_t *) ((size_t) tile - _destTileByteSize);
                        }
                    }

                    break;
            }
        }
    }
}