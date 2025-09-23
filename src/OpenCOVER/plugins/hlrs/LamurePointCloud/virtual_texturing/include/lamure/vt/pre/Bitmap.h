// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef TILE_PROVIDER_BITMAP_H
#define TILE_PROVIDER_BITMAP_H


#include <cstdint>
#include <stdexcept>
#include <lamure/vt/common.h>

namespace vt{
    namespace pre{
        //#define BITMAP_ENABLE_SAFETY_CHECKS

        class VIRTUAL_TEXTURING_DLL Bitmap
    {
        public:
            enum PIXEL_FORMAT{
                R8 = 1,
                RGB8,
                RGBA8,
                LAB
            };

            static constexpr double CIELAB_E = 0.008856; // 216 / 24389
            static constexpr double CIELAB_K = 903.3; // 24389 / 27

            static constexpr double CIELAB_REF_X = 94.811;
            static constexpr double CIELAB_REF_Y = 100.0;
            static constexpr double CIELAB_REF_Z = 107.304;

        protected:
            size_t _width;
            size_t _height;
            size_t _byteSize;

            PIXEL_FORMAT _format;

            bool _externData;
            uint8_t *_data;

            static void _copyPixel(const uint8_t * const srcPx, PIXEL_FORMAT srcFormat, uint8_t * const destPx, PIXEL_FORMAT destFormat);
            static void _deflatePixels(const uint8_t * const srcPx0, const uint8_t * const srcPx1, const uint8_t * const srcPx2, const uint8_t * const srcPx3, PIXEL_FORMAT srcFormat, uint8_t * const destPx, PIXEL_FORMAT destFormat);
            static void _inflatePixel(const uint8_t * const srcPx, PIXEL_FORMAT srcFormat, uint8_t * const destPx0, uint8_t * const destPx1, uint8_t * const destPx2, uint8_t * const destPx3, PIXEL_FORMAT destFormat);

        public:
            Bitmap(size_t width, size_t height, PIXEL_FORMAT pixelFormat, uint8_t *data = nullptr);
            ~Bitmap();

            uint8_t *getData() const;
            size_t getWidth() const;
            size_t getHeight() const;
            size_t getByteSize() const;

            void copyRectFrom(const Bitmap &src, size_t srcX, size_t srcY, size_t destX, size_t destY, size_t cpyWidth, size_t cpyHeight);
            void deflateRectFrom(const Bitmap &src, size_t srcX, size_t srcY, size_t destX, size_t destY, size_t cpyWidth, size_t cpyHeight);
            void inflateRectFrom(const Bitmap &src, size_t srcX, size_t srcY, size_t destX, size_t destY, size_t cpyWidth, size_t cpyHeight);
            void smearHorizontal(size_t srcX, size_t srcY, size_t destX, size_t destY, size_t width, size_t height);
            void smearVertical(size_t srcX, size_t srcY, size_t destX, size_t destY, size_t width, size_t height);
            void fillRect(const uint8_t * const px, PIXEL_FORMAT format, size_t x, size_t y, size_t width, size_t height);

            void setData(uint8_t *data);

            static size_t pixelSize(PIXEL_FORMAT pixelFormat);

        };
    }
}

#endif //TILE_PROVIDER_BITMAP_H
