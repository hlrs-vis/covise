/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_IMAGETEMPL_H_
#define _CO_IMAGETEMPL_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImageTEMPL
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

class coImageTEMPL : public coImageImpl
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Construct from given filename
    coImageTEMPL(const char *filename);

    /// Destructor : virtual in case we derive objects
    virtual ~coImageTEMPL();

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

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    coImageTEMPL(const coImageTEMPL &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    coImageTEMPL &operator=(const coImageTEMPL &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    coImageTEMPL();
};
}
#endif
