/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFSTRING_
#define _VRMLSFSTRING_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFString : public VrmlSField
{
public:
    VrmlSFString(const char *s = 0);
    VrmlSFString(const VrmlSFString &);
    ~VrmlSFString();

    // Assignment.
    void set(const char *s);
    VrmlSFString &operator=(const VrmlSFString &rhs);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFString *toSFString() const;
    virtual VrmlSFString *toSFString();

    const char *get(void) const
    {
        return d_s;
    }

private:
    char *d_s;
};
}
#endif // _VRMLSFSTRING_
