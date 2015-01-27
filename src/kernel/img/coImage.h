/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_IMAGE_H_
#define _CO_IMAGE_H_
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

namespace covise
{
class coImageImpl;
class coBinImage;

/**
 * Base Class for all kinds of images
 *
 */
class COIMAGEEXPORT coImage
{
public:
    /// type for virtual constructor creating from a file name
    typedef coImageImpl *vCtor(const char *filename);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Constructor: build from filename
    coImage(const char *filename);

    /* Constructor: build from binary image: Takes over control about
         given coBinImage and its memor and deletes it in dTor or after
         scaling operations
       */
    coImage(coBinImage *binImage);

    /// Default c'tor: does nothing
    coImage(){};

    /// Destructor: build from filename
    ~coImage();

    /// register a derived class for usage with the Factory
    static bool registerImageType(const char *suffixes[], vCtor *cTor);

    /// create a scaled image: implemented, can be overridden by subclasses
    /// @@@@@ currently only shrink
    void scale(int newWidth, int newHeight);

    /// scale to next lower 2^x in both dimensions
    void scaleExp2();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// width of image
    int getWidth();

    /// height of image;
    int getHeight();

    /// get number of Color Channels
    int getNumChannels();

    /// get number of frames
    int getNumFrames();

    /// get pointer to internal data
    const unsigned char *getBitmap(int frameno = 0);

    /// Did we have errors on reading
    bool isBad();

    /// Error message if problems occurred: empty if ok
    const char *errorMessage();

private:
    /// virtual c'tors for given file endings
    static map<string, vCtor *> *vCtorList_;

    /// the real internal image
    coImageImpl *image_;
};
}
#endif
