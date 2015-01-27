/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_IMAGETIFF_H_
#define _CO_IMAGETIFF_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImageTIFF
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

class coImageTIFF : public coImageImpl
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Construct from given filename
    coImageTIFF(const char *filename);

    /// Destructor : virtual in case we derive objects
    virtual ~coImageTIFF();

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

    // my pixel map
    unsigned char *pixmap_;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    coImageTIFF(const coImageTIFF &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    coImageTIFF &operator=(const coImageTIFF &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    coImageTIFF();
};
}
#endif
