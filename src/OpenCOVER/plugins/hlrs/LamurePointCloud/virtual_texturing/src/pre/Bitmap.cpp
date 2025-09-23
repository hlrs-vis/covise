// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/pre/Bitmap.h>
#include <cmath>


namespace vt {
    namespace pre {
        inline double cielabF(double t){
            if(t > Bitmap::CIELAB_E){
                return std::cbrt(t);
            }else{
                return (7.787 * t) + (16 / 116);
            }
        }

        inline double cielabNormaliseRGB(double rgb){
            rgb /= 255;

            if(rgb > 0.04045){
                rgb = std::pow((rgb + 0.055) / 1.055, 2.4);
            }else{
                rgb = rgb / 12.92;
            }

            return rgb * 100;
        }

        void Bitmap::_copyPixel(const uint8_t * const srcPx, PIXEL_FORMAT srcFormat, uint8_t * const destPx, PIXEL_FORMAT destFormat){
            switch(srcFormat){
                case PIXEL_FORMAT::R8:
                    switch(destFormat) {
                        case PIXEL_FORMAT::R8:
                            destPx[0] = srcPx[0];

                            break;
                        case PIXEL_FORMAT::RGBA8:
                            destPx[3] = 0xff;
                        case PIXEL_FORMAT::RGB8:
                            destPx[0] = srcPx[0];
                            destPx[1] = srcPx[0];
                            destPx[2] = srcPx[0];

                            break;
                        case PIXEL_FORMAT::LAB: {
                            // 10° D65
                            double varR = cielabNormaliseRGB(srcPx[0]);
                            double varG = varR;
                            double varB = varR;

                            double x = (double) 0.4124564 * varR +
                                       (double) 0.3575761 * varG +
                                       (double) 0.1804375 * varB;
                            double y = (double) 0.2126729 * varR +
                                       (double) 0.7151522 * varG +
                                       (double) 0.0721750 * varB;
                            double z = (double) 0.0193339 * varR +
                                       (double) 0.1191920 * varG +
                                       (double) 0.9503041 * varB;

                            double ye = y / Bitmap::CIELAB_REF_Y;

                            double fx = cielabF(x / Bitmap::CIELAB_REF_X);
                            double fy = cielabF(ye);
                            double fz = cielabF(z / Bitmap::CIELAB_REF_Z);

                            auto labDestPx = (float *) destPx;

                            if(ye > Bitmap::CIELAB_E){
                                labDestPx[0] = (float) ((double)116 * std::cbrt(ye) - 16); // L
                            }else{
                                labDestPx[0] = (float) (Bitmap::CIELAB_K * ye); // L
                            }

                            labDestPx[1] = (float) ((double)500 * (fx - fy)); // a
                            labDestPx[2] = (float) ((double)200 * (fy - fz)); // b

                            break;
                        }
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                case PIXEL_FORMAT::RGB8:
                    switch(destFormat){
                        case PIXEL_FORMAT::R8:
                            destPx[0] = (uint8_t)(((uint16_t)srcPx[0] + srcPx[1] + srcPx[2]) / 3);

                            break;
                        case PIXEL_FORMAT::RGBA8:
                            destPx[3] = 0xff;
                        case PIXEL_FORMAT::RGB8:
                            destPx[0] = srcPx[0];
                            destPx[1] = srcPx[1];
                            destPx[2] = srcPx[2];

                            break;
                        case PIXEL_FORMAT::LAB: {
                            // 10° D65
                            double varR = cielabNormaliseRGB(srcPx[0]);
                            double varG = cielabNormaliseRGB(srcPx[1]);
                            double varB = cielabNormaliseRGB(srcPx[2]);

                            double x = (double) 0.4124564 * varR +
                                       (double) 0.3575761 * varG +
                                       (double) 0.1804375 * varB;
                            double y = (double) 0.2126729 * varR +
                                       (double) 0.7151522 * varG +
                                       (double) 0.0721750 * varB;
                            double z = (double) 0.0193339 * varR +
                                       (double) 0.1191920 * varG +
                                       (double) 0.9503041 * varB;

                            double ye = y / Bitmap::CIELAB_REF_Y;

                            double fx = cielabF(x / Bitmap::CIELAB_REF_X);
                            double fy = cielabF(ye);
                            double fz = cielabF(z / Bitmap::CIELAB_REF_Z);

                            auto labDestPx = (float *) destPx;

                            if(ye > 0.008856){
                                labDestPx[0] = (float) ((double)116 * std::cbrt(ye) - 16); // L
                            }else{
                                labDestPx[0] = (float) ((double)903.3 * ye); // L
                            }

                            labDestPx[1] = (float) ((double)500 * (fx - fy)); // a
                            labDestPx[2] = (float) ((double)200 * (fy - fz)); // b

                            break;
                        }
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                case PIXEL_FORMAT::RGBA8:
                    switch(destFormat){
                        case PIXEL_FORMAT::R8:
                            destPx[0] = (uint8_t)(((uint16_t)srcPx[0] + srcPx[1] + srcPx[2]) / 3);

                            break;
                        case PIXEL_FORMAT::RGB8:
                            destPx[0] = srcPx[0];
                            destPx[1] = srcPx[1];
                            destPx[2] = srcPx[2];

                            break;
                        case PIXEL_FORMAT::RGBA8:
                            destPx[0] = srcPx[0];
                            destPx[1] = srcPx[1];
                            destPx[2] = srcPx[2];
                            destPx[3] = srcPx[3];

                            break;
                        case PIXEL_FORMAT::LAB: {
                            // 10° D65
                            double varR = (double)srcPx[0] / 255;
                            double varG = (double)srcPx[1] / 255;
                            double varB = (double)srcPx[2] / 255;

                            if(varR > 0.04045){
                                varR = std::pow((varR + 0.055) / 1.055, 2.4);
                            }else{
                                varR = varR / 12.92;
                            }

                            if(varG > 0.04045){
                                varG = std::pow((varG + 0.055) / 1.055, 2.4);
                            }else{
                                varG = varG / 12.92;
                            }

                            if(varB > 0.04045){
                                varB = std::pow((varB + 0.055) / 1.055, 2.4);
                            }else{
                                varB = varB / 12.92;
                            }

                            varR *= 100;
                            varG *= 100;
                            varB *= 100;

                            double x = (double) 0.4124564 * varR +
                                       (double) 0.3575761 * varG +
                                       (double) 0.1804375 * varB;
                            double y = (double) 0.2126729 * varR +
                                       (double) 0.7151522 * varG +
                                       (double) 0.0721750 * varB;
                            double z = (double) 0.0193339 * varR +
                                       (double) 0.1191920 * varG +
                                       (double) 0.9503041 * varB;

                            double xe = x / Bitmap::CIELAB_REF_X;
                            double ye = y / Bitmap::CIELAB_REF_Y;
                            double ze = z / Bitmap::CIELAB_REF_Z;

                            double fx = cielabF(xe);
                            double fy = cielabF(ye);
                            double fz = cielabF(ze);

                            auto labDestPx = (float *) destPx;

                            labDestPx[0] = (float) ((double)116 * fy - 16); // L
                            labDestPx[1] = (float) ((double)500 * (fx - fy)); // a
                            labDestPx[2] = (float) ((double)200 * (fy - fz)); // b

                            break;
                        }
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                default:
                    throw std::runtime_error("Unknown Pixel Format.");
            }
        }

        void Bitmap::_deflatePixels(const uint8_t * const srcPx0, const uint8_t * const srcPx1, const uint8_t * const srcPx2, const uint8_t * const srcPx3, PIXEL_FORMAT srcFormat, uint8_t * const destPx, PIXEL_FORMAT destFormat){
            switch(srcFormat){
                case PIXEL_FORMAT::R8:
                    switch(destFormat){
                        case PIXEL_FORMAT::R8:
                            destPx[0] = (uint8_t)(((uint16_t)srcPx0[0] + srcPx1[0] + srcPx2[0] + srcPx3[0]) >> 2);

                            break;
                        case PIXEL_FORMAT::RGBA8:
                            destPx[3] = 0xff;
                        case PIXEL_FORMAT::RGB8:
                            destPx[0] = (uint8_t)(((uint16_t)srcPx0[0] + srcPx1[0] + srcPx2[0] + srcPx3[0]) >> 2);
                            destPx[1] = (uint8_t)(((uint16_t)srcPx0[0] + srcPx1[0] + srcPx2[0] + srcPx3[0]) >> 2);
                            destPx[2] = (uint8_t)(((uint16_t)srcPx0[0] + srcPx1[0] + srcPx2[0] + srcPx3[0]) >> 2);

                            break;
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                case PIXEL_FORMAT::RGB8:
                    switch(destFormat){
                        case PIXEL_FORMAT::R8:
                            destPx[0] = (uint8_t)(((uint32_t)srcPx0[0] + srcPx0[1] + srcPx0[2] + srcPx1[0] + srcPx1[1] + srcPx1[2] + srcPx2[0] + srcPx2[1] + srcPx2[2] + srcPx3[0] + srcPx3[1] + srcPx3[2]) / 12);

                            break;
                        case PIXEL_FORMAT::RGBA8:
                            destPx[3] = 0xff;
                        case PIXEL_FORMAT::RGB8:
                            destPx[0] = (uint8_t)(((uint16_t)srcPx0[0] + srcPx1[0] + srcPx2[0] + srcPx3[0]) >> 2);
                            destPx[1] = (uint8_t)(((uint16_t)srcPx0[1] + srcPx1[1] + srcPx2[1] + srcPx3[1]) >> 2);
                            destPx[2] = (uint8_t)(((uint16_t)srcPx0[2] + srcPx1[2] + srcPx2[2] + srcPx3[2]) >> 2);

                            break;
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                case PIXEL_FORMAT::RGBA8:
                    switch(destFormat){
                        case PIXEL_FORMAT::R8:
                            destPx[0] = (uint8_t)(((uint32_t)srcPx0[0] + srcPx0[1] + srcPx0[2] + srcPx1[0] + srcPx1[1] + srcPx1[2] + srcPx2[0] + srcPx2[1] + srcPx2[2] + srcPx3[0] + srcPx3[1] + srcPx3[2]) / 12);

                            break;
                        case PIXEL_FORMAT::RGBA8:
                            destPx[3] = (uint8_t)(((uint16_t)srcPx0[3] + srcPx1[3] + srcPx2[3] + srcPx3[3]) >> 2);
                        case PIXEL_FORMAT::RGB8:
                            destPx[0] = (uint8_t)(((uint16_t)srcPx0[0] + srcPx1[0] + srcPx2[0] + srcPx3[0]) >> 2);
                            destPx[1] = (uint8_t)(((uint16_t)srcPx0[1] + srcPx1[1] + srcPx2[1] + srcPx3[1]) >> 2);
                            destPx[2] = (uint8_t)(((uint16_t)srcPx0[2] + srcPx1[2] + srcPx2[2] + srcPx3[2]) >> 2);

                            break;
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                default:
                    throw std::runtime_error("Unknown Pixel Format.");
            }
        }

        void Bitmap::_inflatePixel(const uint8_t * const srcPx, PIXEL_FORMAT srcFormat, uint8_t * const destPx0, uint8_t * const destPx1, uint8_t * const destPx2, uint8_t * const destPx3, PIXEL_FORMAT destFormat){
            switch(srcFormat){
                case PIXEL_FORMAT::R8:
                    switch(destFormat){
                        case PIXEL_FORMAT::RGBA8:
                            destPx0[3] = 0xff;

                            destPx1[3] = 0xff;

                            destPx2[3] = 0xff;

                            destPx3[3] = 0xff;
                        case PIXEL_FORMAT::RGB8:
                            destPx0[1] = srcPx[0];
                            destPx0[2] = srcPx[0];

                            destPx1[1] = srcPx[0];
                            destPx1[2] = srcPx[0];

                            destPx2[1] = srcPx[0];
                            destPx2[2] = srcPx[0];

                            destPx3[1] = srcPx[0];
                            destPx3[2] = srcPx[0];
                        case PIXEL_FORMAT::R8:
                            destPx0[0] = srcPx[0];

                            destPx1[0] = srcPx[0];

                            destPx2[0] = srcPx[0];

                            destPx3[0] = srcPx[0];

                            break;
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                case PIXEL_FORMAT::RGB8:
                    switch(destFormat){
                        case PIXEL_FORMAT::R8: {
                            uint8_t avrg = (uint8_t) (((uint16_t) srcPx[0] + srcPx[1] + srcPx[2]) / 3);

                            destPx0[0] = avrg;

                            destPx1[0] = avrg;

                            destPx2[0] = avrg;

                            destPx3[0] = avrg;

                            break;
                        }
                        case PIXEL_FORMAT::RGBA8:
                            destPx0[3] = 0xff;

                            destPx1[3] = 0xff;

                            destPx2[3] = 0xff;

                            destPx3[3] = 0xff;
                        case PIXEL_FORMAT::RGB8:
                            destPx0[0] = srcPx[0];
                            destPx0[1] = srcPx[1];
                            destPx0[2] = srcPx[2];

                            destPx1[0] = srcPx[0];
                            destPx1[1] = srcPx[1];
                            destPx1[2] = srcPx[2];

                            destPx2[0] = srcPx[0];
                            destPx2[1] = srcPx[1];
                            destPx2[2] = srcPx[2];

                            destPx3[0] = srcPx[0];
                            destPx3[1] = srcPx[1];
                            destPx3[2] = srcPx[2];

                            break;
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                case PIXEL_FORMAT::RGBA8:
                    switch(destFormat){
                        case PIXEL_FORMAT::R8: {
                            uint8_t avrg = (uint8_t) (((uint16_t) srcPx[0] + srcPx[1] + srcPx[2]) / 3);

                            destPx0[0] = avrg;

                            destPx1[0] = avrg;

                            destPx2[0] = avrg;

                            destPx3[0] = avrg;

                            break;
                        }
                        case PIXEL_FORMAT::RGBA8:
                            destPx0[3] = srcPx[3];

                            destPx1[3] = srcPx[3];

                            destPx2[3] = srcPx[3];

                            destPx3[3] = srcPx[3];
                        case PIXEL_FORMAT::RGB8:
                            destPx0[0] = srcPx[0];
                            destPx0[1] = srcPx[1];
                            destPx0[2] = srcPx[2];

                            destPx1[0] = srcPx[0];
                            destPx1[1] = srcPx[1];
                            destPx1[2] = srcPx[2];

                            destPx2[0] = srcPx[0];
                            destPx2[1] = srcPx[1];
                            destPx2[2] = srcPx[2];

                            destPx3[0] = srcPx[0];
                            destPx3[1] = srcPx[1];
                            destPx3[2] = srcPx[2];

                            break;
                        default:
                            throw std::runtime_error("No Conversion between given Pixel Formats.");
                    }

                    break;
                default:
                    throw std::runtime_error("Unknown Pixel Format.");
            }
        }

        size_t Bitmap::pixelSize(PIXEL_FORMAT pixelFormat){
            switch(pixelFormat){
                case PIXEL_FORMAT::R8:
                    return 1;
                case PIXEL_FORMAT::RGB8:
                    return 3;
                case PIXEL_FORMAT::RGBA8:
                    return 4;
                case PIXEL_FORMAT::LAB:
                    return 12;
                default:
                    throw std::runtime_error("Unknown Pixel Format.");
            }
        }

        Bitmap::Bitmap(size_t width, size_t height, PIXEL_FORMAT pixelFormat, uint8_t *data){
            _width = width;
            _height = height;
            _byteSize = width * height * pixelSize(pixelFormat);
            _format = pixelFormat;
            _externData = data != nullptr;

            if(!_externData){
                data = new uint8_t[_byteSize];
            }

            _data = data;
        }

        Bitmap::~Bitmap() {
            if(!_externData) {
                delete[] _data;
            }
        }

        size_t Bitmap::getWidth() const {
            return _width;
        }

        size_t Bitmap::getHeight() const {
            return _height;
        }

        size_t Bitmap::getByteSize() const {
            return _byteSize;
        }

        void Bitmap::copyRectFrom(const Bitmap &src, size_t srcX, size_t srcY, size_t destX, size_t destY, size_t cpyWidth, size_t cpyHeight){
#ifdef BITMAP_ENABLE_SAFETY_CHECKS
            if((srcX + cpyWidth) > src._width || (srcY + cpyHeight) > src._height){
        throw std::runtime_error("Trying to copy Rect outside of Source Boundaries.");
    }

    if((destX + cpyWidth) > _width || (destY + cpyHeight) > _height){
        throw std::runtime_error("Trying to copy Rect outside of Destination Boundaries.");
    }
#endif

            size_t srcPixelSize = pixelSize(src._format);
            size_t destPixelSize = pixelSize(_format);

            for(size_t y = 0; y < cpyHeight; ++y) {
                for (size_t x = 0; x < cpyWidth; ++x) {
                    _copyPixel(&src._data[((srcY + y) * src._width + srcX + x) * srcPixelSize], src._format, &_data[((destY + y) * _width + destX + x) * destPixelSize], _format);
                }
            }
        }

        void Bitmap::deflateRectFrom(const Bitmap &src, size_t srcX, size_t srcY, size_t destX, size_t destY, size_t cpyWidth, size_t cpyHeight){
#ifdef BITMAP_ENABLE_SAFETY_CHECKS
            if((srcX + cpyWidth) > src._width || (srcY + cpyHeight) > src._height){
        throw std::runtime_error("Trying to copy Rect outside of Source Boundaries.");
    }

    if((destX + ((cpyWidth + 1) >> 1)) > _width || (destY + ((cpyHeight + 1) >> 1)) > _height){
        throw std::runtime_error("Trying to copy Rect outside of Destination Boundaries.");
    }
#endif

            size_t srcPixelSize = pixelSize(src._format);
            size_t destPixelSize = pixelSize(_format);

            for(size_t y = 0; y < cpyHeight; y += 2) {
                for (size_t x = 0; x < cpyWidth; x += 2) {
                    _deflatePixels(&src._data[((srcY + y) * src._width + srcX + x) * srcPixelSize], &src._data[((srcY + y) * src._width + srcX + x + 1) * srcPixelSize],
                                   &src._data[((srcY + y + 1) * src._width + srcX + x) * srcPixelSize], &src._data[((srcY + y + 1) * src._width + srcX + x + 1) * srcPixelSize],
                                   src._format,
                                   &_data[((destY + (y >> 1)) * _width + destX + (x >> 1)) * destPixelSize],
                                   _format);
                }
            }
        }

        void Bitmap::inflateRectFrom(const Bitmap &src, size_t srcX, size_t srcY, size_t destX, size_t destY, size_t cpyWidth, size_t cpyHeight){
#ifdef BITMAP_ENABLE_SAFETY_CHECKS
            if((srcX + cpyWidth) > src._width || (srcY + cpyHeight) > src._height){
        throw std::runtime_error("Trying to copy Rect outside of Source Boundaries.");
    }

    if((destX + (cpyWidth << 1)) > _width || (destY + (cpyHeight << 1)) > _height){
        throw std::runtime_error("Trying to copy Rect outside of Destination Boundaries.");
    }
#endif

            size_t srcPixelSize = pixelSize(src._format);
            size_t destPixelSize = pixelSize(_format);

            for(size_t y = 0; y < cpyHeight; y += 2) {
                for (size_t x = 0; x < cpyWidth; x += 2) {
                    _inflatePixel(&src._data[((srcY + y) * src._width + srcX + x) * srcPixelSize], src._format,
                                  &_data[((destY + (y << 1)) * _width + destX + (x << 1)) * destPixelSize],
                                  &_data[((destY + (y << 1)) * _width + destX + (x << 1) + 1) * destPixelSize],
                                  &_data[((destY + (y << 1) + 1) * _width + destX + (x << 1)) * destPixelSize],
                                  &_data[((destY + (y << 1) + 1) * _width + destX + (x << 1) + 1) * destPixelSize],
                                  _format);
                }
            }
        }

        void Bitmap::smearHorizontal(size_t srcX, size_t srcY, size_t destX, size_t destY, size_t width, size_t height){
#ifdef BITMAP_ENABLE_SAFETY_CHECKS
            if(srcX >= _width || (srcY + height) > _height || (destX + width) > _width || (destY + height) > _height){
        throw std::runtime_error("Trying to smear outside of Boundaries.");
    }
#endif

            size_t pxSize = pixelSize(_format);

            for(size_t y = 0; y < height; ++y){
                uint8_t *srcPx = &_data[((srcY + y) * _width + srcX) * pxSize];

                for(size_t x = 0; x < width; ++x){
                    _copyPixel(srcPx, _format, &_data[((destY + y) * _width + destX + x) * pxSize], _format);
                }
            }
        }

        void Bitmap::smearVertical(size_t srcX, size_t srcY, size_t destX, size_t destY, size_t width, size_t height){
#ifdef BITMAP_ENABLE_SAFETY_CHECKS
            if(srcX >= _width || (srcY + height) > _height || (destX + width) > _width || (destY + height) > _height){
        throw std::runtime_error("Trying to smear outside of Boundaries.");
    }
#endif

            size_t pxSize = pixelSize(_format);

            for(size_t x = 0; x < width; ++x){
                uint8_t *srcPx = &_data[(srcY * _width + srcX + x) * pxSize];

                for(size_t y = 0; y < height; ++y){
                    _copyPixel(srcPx, _format, &_data[((destY + y) * _width + destX + x) * pxSize], _format);
                }
            }
        }

        void Bitmap::fillRect(const uint8_t * const px, PIXEL_FORMAT format, size_t x, size_t y, size_t width, size_t height){
#ifdef BITMAP_ENABLE_SAFETY_CHECKS
            if(x + width > _width || y + height > _height){
        throw std::runtime_error("Trying to fill outside of Boundaries.");
    }
#endif

            size_t pxSize = pixelSize(_format);

            for(size_t yPos = 0; yPos < height; ++yPos){
                for(size_t xPos = 0; xPos < width; ++xPos){
                    _copyPixel(px, format, &_data[((y + yPos) * _width + x + xPos) * pxSize], _format);
                }
            }
        }

        void Bitmap::setData(uint8_t *data){
            _data = data;
        }

        uint8_t *Bitmap::getData() const {
            return _data;
        }
    }
}
