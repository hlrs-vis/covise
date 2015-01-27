/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFINT_
#define _VRMLSFINT_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFInt : public VrmlSField
{
public:
    VrmlSFInt(int value = 0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFInt *toSFInt() const;
    virtual VrmlSFInt *toSFInt();

    int get(void) const
    {
        return d_value;
    }
    void set(int value)
    {
        d_value = value;
    }

private:
    int d_value;
};
}
#endif //_VRMLSFINT_
