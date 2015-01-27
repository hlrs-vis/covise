/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_BIN_IMAGE_H_
#define _CO_BIN_IMAGE_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coBinImage
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImageImpl.h"

/**
 * Class Template for new image formats
 *
 */

namespace covise
{
class coBinImage : public coImageImpl
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Construct from given filename : asserted to 0 here !
    coBinImage(const char *filename);

    /// Construct sizes, eventually fill from given field
    coBinImage(int width, int height, int numChannels, int numFrames,
               void *buffer = NULL);

    /// Construct from given input file
    coBinImage(int width, int height, int numChannels, int numFrames,
               const char *filename);

    /// Destructor : virtual in case we derive objects
    virtual ~coBinImage();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// get pointer to internal data
    virtual unsigned char *getBitmap(int frameno = 0);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // my pixel buffers
    unsigned char **pixbuf;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy roct lsutines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    coBinImage(const coBinImage &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    coBinImage &operator=(const coBinImage &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    coBinImage();
};
}
#endif
