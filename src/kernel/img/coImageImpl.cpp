/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImage
//
// This class @@@
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImageImpl.h"
#include "coBadImage.h"
#include <assert.h>

using namespace covise;

coImageImpl::coImageImpl()
{
    width_ = 0;
    height_ = 0;
    numChannels_ = 0;
    numFrames_ = 1;
    isBad_ = false;
    errString_ = new char[1];
    *errString_ = '\0';
}

/// Construct from given dimensions
coImageImpl::coImageImpl(int width, int height, int numChannels, int numFrames)
{
    width_ = width;
    height_ = height;
    numChannels_ = numChannels;
    numFrames_ = numFrames;
    isBad_ = false;
    errString_ = new char[1];
    *errString_ = '\0';
}

/// width of image
int coImageImpl::getWidth()
{
    return width_;
}

/// height of image;
int coImageImpl::getHeight()
{
    return height_;
}

/// get number of Color Channels
int coImageImpl::getNumChannels()
{
    return numChannels_;
}

/// get number of frames
int coImageImpl::getNumFrames()
{
    return numFrames_;
}

/// Did we have errors on reading
bool coImageImpl::isBad()
{
    return isBad_;
}

/// Error message if problems occurred: empty if ok
const char *coImageImpl::errorMessage()
{
    return errString_;
}

// set error flag to true and copy error string
void coImageImpl::setError(const char *errorString)
{
    isBad_ = true;
    delete[] errString_;
    errString_ = new char[strlen(errorString) + 1];
    strcpy(errString_, errorString);
}
