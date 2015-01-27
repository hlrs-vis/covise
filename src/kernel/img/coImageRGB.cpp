/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImageRGB
//
// Class Template for new image formats
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImageRGB.h"
#include "coImage.h"
#include <covise/covise.h>
#include <assert.h>

/// add all suffixes this class might have, NULL-terminated
static const char *suffixes[] = { "RGB", "Rgb", "rgb", NULL };

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Factory methods: Initialization and static cTor
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// static initializer
static coImageImpl *createRGB(const char *filename)
{
    return new coImageRGB(filename);
}

/// Registration at factory
static bool registered = coImage::registerImageType(suffixes, &createRGB);

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void coImageRGB::readBW(IMAGE *image)
{
#ifndef WIN32
    int x, y;
    pixmap_ = new unsigned char[width_ * height_];
    unsigned char *imgPtr = pixmap_;
    short *line = new short[width_];
    for (y = 0; y < height_; y++)
    {
        getrow(image, line, height_ - y - 1, 0);

        for (x = 0; x < width_; x++)
        {
            *imgPtr++ = line[x];
        }
    }
#endif
}

void coImageRGB::readRGB(IMAGE *image)
{
#ifndef WIN32
    short *redBuf = new short[width_];
    short *greenBuf = new short[width_];
    short *blueBuf = new short[width_];
    pixmap_ = new unsigned char[3 * width_ * height_];
    int y;
    short *redP, *greenP, *blueP;
    unsigned char *imgPtr = pixmap_;
    for (y = 0; y < height_; y++)
    {
        //We read one line per channel
        getrow(image, redBuf, height_ - y - 1, 0);
        getrow(image, greenBuf, height_ - y - 1, 1);
        getrow(image, blueBuf, height_ - y - 1, 2);

        //and copy them into the result ;
        int x;
        redP = redBuf;
        greenP = greenBuf;
        blueP = blueBuf;
        for (x = 0; x < width_; x++)
        {
            *imgPtr++ = (char)*redP++;
            *imgPtr++ = (char)*greenP++;
            *imgPtr++ = (char)*blueP++;
        }
    }
    delete redBuf;
    delete greenBuf;
    delete blueBuf;
#endif
}

void coImageRGB::readRGBA(IMAGE *image)
{

#ifndef WIN32
    short *redBuf = new short[width_];
    short *greenBuf = new short[width_];
    short *blueBuf = new short[width_];
    short *alphaBuf = new short[width_];
    pixmap_ = new unsigned char[4 * width_ * height_];
    int y;
    unsigned char *imgPtr = pixmap_;
    short *redP, *greenP, *blueP, *alphaP;
    for (y = 0; y < height_; y++)
    {
        //We read one line per channel
        getrow(image, redBuf, height_ - y - 1, 0);
        getrow(image, greenBuf, height_ - y - 1, 1);
        getrow(image, blueBuf, height_ - y - 1, 2);
        getrow(image, alphaBuf, height_ - y - 1, 2);

        //and copy them into the result ;
        int x;
        redP = redBuf;
        greenP = greenBuf;
        blueP = blueBuf;
        alphaP = alphaBuf;
        for (x = 0; x < width_; x++)
        {
            *imgPtr++ = (unsigned char)*redP++;
            *imgPtr++ = (unsigned char)*greenP++;
            *imgPtr++ = (unsigned char)*blueP++;
            *imgPtr++ = (unsigned char)*alphaP++;
        }
    }
    delete redBuf;
    delete greenBuf;
    delete blueBuf;
    delete alphaBuf;
#endif
}

coImageRGB::coImageRGB(const char *filename)
{
#ifndef WIN32
    IMAGE *image = NULL;
    image = iopen(filename, "r");
    if (NULL == image)
    {
        setError(strerror(errno));
        return;
    }
    width_ = image->xsize;
    height_ = image->ysize;
    dimension_ = image->dim;
    switch (dimension_)
    {
    case 1:
        numChannels_ = 1;
        height_ = 1;
        setError("Sorry one-dimensional pictures unsupported");
        return;
    case 2:
        numChannels_ = 1;
        readBW(image);
        break;
    case 3:
        numChannels_ = image->zsize;
        switch (numChannels_)
        {
        case 3:
            readRGB(image);
            break;
        case 4:
            readRGBA(image);
            break;
        }
        break;
    }
#endif
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coImageRGB::~coImageRGB()
{
    cerr << "Add coImageRGB::~coImageRGB c'tore here" << endl;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get pointer to internal data
unsigned char *
coImageRGB::getBitmap(int frameno)
{
    (void)frameno;
    return pixmap_;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
coImageRGB::coImageRGB(const coImageRGB &)
    : coImageImpl()
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
coImageRGB &coImageRGB::operator=(const coImageRGB &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
coImageRGB::coImageRGB()
{
    assert(0);
}
