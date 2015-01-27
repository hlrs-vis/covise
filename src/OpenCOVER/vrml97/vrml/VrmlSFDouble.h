/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFDOUBLE_
#define _VRMLSFDOUBLE_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFDouble : public VrmlSField
{
public:
    VrmlSFDouble(double value = 0.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFDouble *toSFDouble() const;
    virtual VrmlSFDouble *toSFDouble();

    double get(void) const
    {
        return d_value;
    }
    void set(double value)
    {
        d_value = value;
    }

private:
    double d_value;
};
}
#endif //_VRMLSFDOUBLE_
