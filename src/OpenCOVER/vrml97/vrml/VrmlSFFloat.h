/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFFLOAT_
#define _VRMLSFFLOAT_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFFloat : public VrmlSField
{
public:
    VrmlSFFloat(float value = 0.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFFloat *toSFFloat() const;
    virtual VrmlSFFloat *toSFFloat();

    float get(void) const
    {
        return d_value;
    }
    void set(float value)
    {
        d_value = value;
    }

private:
    float d_value;
};
}
#endif //_VRMLSFFLOAT_
