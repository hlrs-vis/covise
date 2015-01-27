/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _SCRIPTOBJECT_
#define _SCRIPTOBJECT_
//
//  Abstract Script Object class
//
#include "vrmlexport.h"

namespace vrml
{

class VrmlField;
class VrmlMFString;
class VrmlSFString;
class VrmlNodeScript;

class VRMLEXPORT ScriptObject
{

protected:
    ScriptObject(); // Use create()

public:
    virtual ~ScriptObject() = 0;

    // Script object factory.
    static ScriptObject *create(VrmlNodeScript *, VrmlMFString &url, VrmlSFString &relurl);

    // Invoke a named function
    virtual void activate(double timeStamp,
                          const char *fname,
                          int argc,
                          const VrmlField *argv[]) = 0;
};
}
#endif // _SCRIPTOBJECT_
