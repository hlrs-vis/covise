/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImageTEMPL
//
// Class Template for new image formats
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImageTEMPL.h"
#include "coImage.h"
#include <covise/covise.h>
#include <assert.h>

/// add all suffixes this class might have, NULL-terminated
namespace covise
{
static const char *suffixes[] = { "TEMPL", NULL };

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Factory methods: Initialization and static cTor
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// static initializer
static coImageImpl *createTEMPL(const char *filename)
{
    return new coImageTEMPL(filename);
}

/// Registration at factory
static bool registered = coImage::registerImageType(suffixes, &createTEMPL);
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
using namespace covise;

coImageTEMPL::coImageTEMPL(const char *filename)
{
    (void)filename;
    cerr << "Construct from file " << filename << endl;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coImageTEMPL::~coImageTEMPL()
{
    cerr << "Add coImageTEMPL::~coImageTEMPL c'tore here" << endl;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get pointer to internal data
unsigned char *
coImageTEMPL::getBitmap(int frameno)
{
    (void)frameno;
    cerr << "get bitmap of frame " << frameno << endl;
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
coImageTEMPL::coImageTEMPL(const coImageTEMPL &)
    : coImageImpl()
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
coImageTEMPL &coImageTEMPL::operator=(const coImageTEMPL &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
coImageTEMPL::coImageTEMPL()
{
    assert(0);
}
