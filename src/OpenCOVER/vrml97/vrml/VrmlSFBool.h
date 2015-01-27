/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFBOOL_
#define _VRMLSFBOOL_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFBool : public VrmlSField
{
public:
    VrmlSFBool(bool value = false);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFBool *toSFBool() const;
    virtual VrmlSFBool *toSFBool();

    bool get(void) const
    {
        return d_value;
    }
    void set(bool value)
    {
        d_value = value;
    }

private:
    bool d_value;
};
}
#endif //_VRMLSFBOOL_
