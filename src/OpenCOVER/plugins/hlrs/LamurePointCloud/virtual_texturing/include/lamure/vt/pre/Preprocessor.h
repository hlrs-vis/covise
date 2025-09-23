// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef TILE_PROVIDER_PREPROCESSOR_H
#define TILE_PROVIDER_PREPROCESSOR_H

//#define PREPROCESSOR_LOG_PROGRESS

#ifdef PREPROCESSOR_LOG_PROGRESS
#include <iostream>
#endif

#include <cstddef>
#include <string>
#include <fstream>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cmath>

#include <lamure/vt/pre/QuadTree.h>
#include <lamure/vt/pre/Bitmap.h>
#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/vt/pre/OffsetIndex.h>
#include <lamure/vt/pre/CielabIndex.h>

namespace vt{
    namespace pre {
    class VIRTUAL_TEXTURING_DLL Preprocessor
    {
        public:
            enum DEST_COMBINED{
                NONE = 1,
                COMBINED,
                NOT_COMBINED
            };

        protected:
            static constexpr size_t _HEADER_SIZE = 71;

            std::string _srcFileName;
            Bitmap::PIXEL_FORMAT _srcPxFormat;

            std::string _destFileName;
            Bitmap::PIXEL_FORMAT _destPxFormat;
            DEST_COMBINED _destCombined;
            AtlasFile::LAYOUT _destLayout;

            size_t _imageWidth;
            size_t _imageHeight;
            size_t _imageTileWidth;
            size_t _imageTileHeight;

            size_t _tileWidth;
            size_t _tileHeight;
            size_t _padding;
            size_t _innerTileWidth;
            size_t _innerTileHeight;
            size_t _destTileByteSize;

            size_t _treeDepth;

            std::ifstream _srcFile;
            uint64_t _srcFileSize;

            std::fstream *_destHeaderFile;
            uint64_t _destHeaderOffset;

            std::fstream *_destIndexFile;

            vt::pre::OffsetIndex *_offsetIndex;
            uint64_t _destOffsetIndexOffset;

            vt::pre::CielabIndex *_cielabIndex;
            uint64_t _destCielabIndexOffset;

            std::fstream _destPayloadFile;
            uint64_t _destPayloadOffset;

            bool _isPowerOfTwo(size_t val);

            size_t _loadTileById(uint64_t id, uint8_t *out);
            size_t _getTileById(uint64_t id, const uint8_t *buffer, uint64_t firstIdInBuffer, uint64_t lastIdInBuffer, const uint64_t *idLookup, size_t bufferTileLen, uint8_t *out);
            size_t _getBufferedTileById(uint64_t id, const uint8_t *buffer, uint64_t firstIdInBuffer, uint64_t lastIdInBuffer, const uint64_t *idLookup, size_t bufferTileLen, uint8_t *out);

            void _writeHeader();
            void _extract(size_t bufferTileWidth, size_t writeBufferTileSize);
            void _deflate(size_t tilesInWriteBuffer);

            void _putLE(uint64_t num, uint8_t *out);
            void _putPixelFormat(Bitmap::PIXEL_FORMAT pxFormat, uint8_t *out);
            void _putFileFormat(AtlasFile::LAYOUT fileFormat, uint8_t *out);

            void _loadToSeqBuffer(uint8_t *buffer, uint64_t id, size_t count);

        public:
            Preprocessor(const std::string &srcFileName,
                         Bitmap::PIXEL_FORMAT srcPxFormat,
                         size_t imageWidth,
                         size_t imageHeight);

            ~Preprocessor();

            void setOutput(const std::string &destFileName,
                           Bitmap::PIXEL_FORMAT destPxFormat,
                           AtlasFile::LAYOUT format,
                           size_t tileWidth,
                           size_t tileHeight,
                           size_t padding,
                           bool combine = true);

            void run(size_t maxMemory);
        };
    }
}

#endif //TILE_PROVIDER_PREPROCESSOR_H
