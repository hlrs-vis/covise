/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_IMAGEIMPL_H_
#define _CO_IMAGEIMPL_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImage
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include <covise/covise.h>

/**
 * Base Class: common things for image class implementations
 *
 */
namespace covise
{

class coImageImpl
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Construct without setting anything - derived classes will do later
    coImageImpl();
    virtual ~coImageImpl()
    {
    }

    /// Construct from given dimensions
    coImageImpl(int width, int height, int numChannels, int numFrames);

    /// width of image
    virtual int getWidth();

    /// height of image;
    virtual int getHeight();

    /// get number of Color Channels
    virtual int getNumChannels();

    /// get number of frames
    virtual int getNumFrames();

    /// get pointer to internal data
    virtual unsigned char *getBitmap(int frameno = 0) = 0;

    /// Did we have errors on reading
    virtual bool isBad();

    /// Error message if problems occurred: empty if ok
    virtual const char *errorMessage();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute setting
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // set error flag to true and copy error string
    void setError(const char *errorString);

protected:
    // the values for these variables HAVE to be set by the implementation
    int width_;
    int height_;
    int numChannels_;
    int numFrames_;

private:
    char *errString_;
    bool isBad_;
};
}
#endif
