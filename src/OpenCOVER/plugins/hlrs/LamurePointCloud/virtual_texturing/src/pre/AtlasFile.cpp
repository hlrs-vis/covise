// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/vt/pre/OffsetIndex.h>

namespace vt{
    namespace pre {
        uint64_t AtlasFile::_getLE(uint8_t *data){
            uint64_t num = ((uint64_t)data[0]) |
                           (((uint64_t)data[1]) << 8) |
                           (((uint64_t)data[2]) << 16) |
                           (((uint64_t)data[3]) << 24) |
                           (((uint64_t)data[4]) << 32) |
                           (((uint64_t)data[5]) << 40) |
                           (((uint64_t)data[6]) << 48) |
                           (((uint64_t)data[7]) << 56);

            return num;
        }

        Bitmap::PIXEL_FORMAT AtlasFile::_getPixelFormat(uint8_t *data){
            switch(data[0]){
                case 1:
                    return Bitmap::PIXEL_FORMAT::R8;
                case 2:
                    return Bitmap::PIXEL_FORMAT::RGB8;
                case 3:
                    return Bitmap::PIXEL_FORMAT::RGBA8;
                default:
                    throw std::runtime_error("Trying to load unknown Pixel Format.");
            }
        }

        AtlasFile::LAYOUT AtlasFile::_getFormat(uint8_t *data){
            switch(data[0]){
                case 1:
                    return LAYOUT::RAW;
                case 2:
                    return LAYOUT::PACKED;
                default:
                    throw std::runtime_error("Trying to load unknown File Format.");
            }
        }

        AtlasFile::AtlasFile(const char *fileName){
            _fileName = fileName;
            _file.open(fileName, std::ios::binary | std::ios::ate);

            if(!_file.is_open()){
                throw std::runtime_error("Could not open Atlas-File.");
            }

            auto fileLen = (uint64_t)_file.tellg();

            if(fileLen < HEADER_SIZE){
                throw std::runtime_error("Could not read Header of Atlas-File.");
            }

            uint8_t data[HEADER_SIZE];

            _file.clear();
            _file.seekg(0, std::ios::beg);
            _file.read((char*)data, HEADER_SIZE);

            if(data[0] != 'A' || data[1] != 'T' || data[2] != 'L' || data[3] != 'A' || data[4] != 'S'){
                throw std::runtime_error("Trying to open wrong File Format.");
            }

            _imageWidth = _getLE(&data[5]);
            _imageHeight = _getLE(&data[13]);
            _tileWidth = _getLE(&data[21]);
            _tileHeight = _getLE(&data[29]);
            _padding = _getLE(&data[37]);

            _pxFormat = _getPixelFormat(&data[45]);
            _format = _getFormat(&data[46]);

            _offsetIndexOffset = _getLE(&data[47]);
            _cielabIndexOffset = _getLE(&data[55]);
            _payloadOffset = _getLE(&data[63]);

            _innerTileWidth = _tileWidth - (_padding << 1);
            _innerTileHeight = _tileHeight - (_padding << 1);

            _imageTileWidth = (_imageWidth + _innerTileWidth - 1) / _innerTileWidth;
            _imageTileHeight = (_imageHeight + _innerTileHeight - 1) / _innerTileHeight;
            _pxSize = Bitmap::pixelSize(_pxFormat);
            _tilePxSize = _tileWidth * _tileHeight;
            _tileByteSize = _tilePxSize * _pxSize;

            _treeDepth = QuadTree::getDepth(_imageTileWidth, _imageTileHeight);

            auto imageTileWidth = _imageTileWidth;
            auto imageTileHeight = _imageTileHeight;

            _totalTileCount = 0;
            _filledTileCount = 0;

            for(auto level = _treeDepth - 1; ; --level){
                auto widthOfLevel = QuadTree::getWidthOfLevel(level);

                _totalTileCount += widthOfLevel * widthOfLevel;
                _filledTileCount += imageTileWidth * imageTileHeight;

                if(level == 0){
                    break;
                }

                imageTileWidth = (imageTileWidth + 1) >> 1;
                imageTileHeight = (imageTileHeight + 1) >> 1;
            }

            if(false && _format == LAYOUT::RAW){
                if(fileLen != HEADER_SIZE + _totalTileCount * _tileByteSize){
                    throw std::runtime_error("Atlas-File does not have the expected Size.");
                }

                _offsetIndex = nullptr;
            }else if(_format == LAYOUT::PACKED){
                if(fileLen != _payloadOffset + _filledTileCount * _tileByteSize){
                    throw std::runtime_error("Atlas-File does not have the expected Size.");
                }

                _offsetIndex = new OffsetIndex(_totalTileCount, _format);
                _file.seekg(_offsetIndexOffset);
                _offsetIndex->readFromFile(_file);

            }

            _cielabIndex = new CielabIndex(_totalTileCount);
            _file.seekg(_cielabIndexOffset);
            _cielabIndex->readFromFile(_file);
        }

        AtlasFile::~AtlasFile(){
            _file.close();
            delete _offsetIndex;
            delete _cielabIndex;
        }

        uint64_t AtlasFile::getFilledTiles(){
            return _filledTileCount;
        }

        uint64_t AtlasFile::getTotalTiles(){
            return _totalTileCount;
        }

        uint32_t AtlasFile::getDepth(){
            return _treeDepth;
        }

        uint64_t AtlasFile::getImageWidth(){
            return _imageWidth;
        }

        uint64_t AtlasFile::getImageHeight(){
            return _imageHeight;
        }

        uint64_t AtlasFile::getTileWidth(){
            return _tileWidth;
        }

        uint64_t AtlasFile::getTileHeight(){
            return _tileHeight;
        }

        uint64_t AtlasFile::getInnerTileWidth(){
            return _innerTileWidth;
        }

        uint64_t AtlasFile::getInnerTileHeight(){
            return _innerTileHeight;
        }

        uint64_t AtlasFile::getTileByteSize(){
            return _tileByteSize;
        }

        uint64_t AtlasFile::getPadding(){
            return _padding;
        }

        Bitmap::PIXEL_FORMAT AtlasFile::getPixelFormat(){
            return _pxFormat;
        }

        AtlasFile::LAYOUT AtlasFile::getFormat(){
            return _format;
        }

        uint64_t AtlasFile::_getOffset(uint64_t id){
            if(id >= _totalTileCount){
                return UINT64_MAX;
            }

            return id * _tileByteSize;
        }

        float AtlasFile::getCielabValue(uint64_t id){
            return _cielabIndex->getCielabValue(id);
        }

        bool AtlasFile::getTile(uint64_t id, uint8_t *out){
            uint64_t offset = 0;

            if(_format == LAYOUT::PACKED) {
                if(_offsetIndex->exists(id)){
                    offset = _offsetIndex->getOffset(id);
                } else {
                    offset = UINT64_MAX;
                }
            }else{
                offset = _getOffset(id);
            }

            if(offset == UINT64_MAX){
                std::memset((char*)out, 0x00, _tileByteSize);

                return false;
            }

            _file.seekg(_payloadOffset + offset);
            _file.read((char*)out, _tileByteSize);

            return true;
        }

        void AtlasFile::extractLevel(uint32_t level, const char *fileName){

            std::ofstream file(fileName, std::ios::binary | std::ios::trunc);

            if(!file.is_open()){
                throw std::runtime_error("Could not open File.");
            }

            auto firstId = QuadTree::firstIdOfLevel(level);
            auto tileWidth = QuadTree::getWidthOfLevel(level);

            auto readBufferSize = _tileByteSize;
            auto innerTileByteSize = _innerTileWidth * _innerTileHeight * _pxSize;
            auto writeBufferTileSize = tileWidth;
            auto writeBufferSize = writeBufferTileSize * innerTileByteSize;
            auto readBuffer = new uint8_t[readBufferSize + writeBufferSize];
            auto writeBuffer = &readBuffer[readBufferSize];

            Bitmap readBitmap(_tileWidth, _tileHeight, _pxFormat, readBuffer);
            Bitmap writeBitmap(_innerTileWidth * writeBufferTileSize, _innerTileHeight, _pxFormat, writeBuffer);

            uint64_t firstRelId = 0;
            uint64_t relId = firstRelId;

            for(uint64_t y = 0; y < tileWidth; ++y){
                for(uint64_t x = 0; x < tileWidth; ++x){
                    getTile(firstId + relId, readBuffer);
                    writeBitmap.copyRectFrom(readBitmap, _padding, _padding, x * _innerTileWidth, 0, _innerTileWidth, _innerTileHeight);

                    relId = QuadTree::getNeighbour(relId, QuadTree::NEIGHBOUR::RIGHT);
                }

                file.write((char*)writeBuffer, writeBufferSize);

                firstRelId = QuadTree::getNeighbour(firstRelId, QuadTree::NEIGHBOUR::BOTTOM);
                relId = firstRelId;
            }

            file.close();
            delete[] readBuffer;
        }

        uint64_t AtlasFile::getOffsetIndexOffset(){
            return this->_offsetIndexOffset;
        }

        uint64_t AtlasFile::getCielabIndexOffset(){
            return this->_cielabIndexOffset;
        }

        uint64_t AtlasFile::getPayloadOffset(){
            return _payloadOffset;
        }

        uint64_t AtlasFile::getImageTiledWidth() {
            return _imageTileWidth;
        }

        uint64_t AtlasFile::getImageTiledHeight() {
            return _imageTileHeight;
        }

        const char *AtlasFile::getFileName() {
            return _fileName;
        }
    }
}
