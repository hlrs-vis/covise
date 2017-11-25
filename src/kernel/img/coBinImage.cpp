/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coBinImage
//
// Class Template for new image formats
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coBinImage.h"
#include "coImage.h"
#include <covise/covise.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

using namespace covise;

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coBinImage::coBinImage(const char *filename)
{
    cerr << "coBinImage::coBinImage(filename=" << filename << ") called" << endl;
    assert(0);
}

coBinImage::coBinImage(int width, int height, int numChannels, int numFrames, void *buffer)
    : coImageImpl(width, height, numChannels, numFrames)
{
    unsigned char *chBuffer = (unsigned char *)buffer;

    pixbuf = new unsigned char *[numFrames];
    int i;
    int numBytes = width * height * numChannels;
    for (i = 0; i < numFrames; i++)
    {
        pixbuf[i] = new unsigned char[numBytes];
        if (buffer)
            memmove(pixbuf[i], chBuffer + i * numBytes, numBytes);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coBinImage::coBinImage(int width, int height, int numChannels, int numFrames,
                       const char *filename)
    : coImageImpl(width, height, numChannels, numFrames)
{
    // Check input file
    FILE *fi = fopen(filename, "r");
    if (!fi)
    {
        char buffer[1024];
        sprintf(buffer, "%s: %s", filename, strerror(errno));
        setError(buffer);
        return;
    }

    pixbuf = new unsigned char *[numFrames];
    int i;
    int numBytes = width * height * numChannels;
    for (i = 0; i < numFrames; i++)
    {
        size_t itemsRead = 0;
        pixbuf[i] = new unsigned char[numBytes];
        if (pixbuf[i])
            itemsRead = fread(pixbuf[i], numChannels, width * height, fi);
        else
        {
            setError("coBinImage::coBinImage cannot allocate memory");
            fclose(fi);
            return;
        }
        if (itemsRead < width * height)
        {
            setError("coBinImage::coBinImage incomplete read");
            fclose(fi);
            return;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coBinImage::~coBinImage()
{
    int i;
    for (i = 0; i < numFrames_; i++)
        delete[] pixbuf[i];
    delete[] pixbuf;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get pointer to internal data
unsigned char *
coBinImage::getBitmap(int frameno)
{
    if (frameno >= 0 || frameno < numFrames_)
        return pixbuf[frameno];
    else
        return NULL;
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
coBinImage::coBinImage(const coBinImage &)
    : coImageImpl()
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
coBinImage &coBinImage::operator=(const coBinImage &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
coBinImage::coBinImage()
{
    assert(0);
}
