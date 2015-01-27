/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFNODE_
#define _VRMLSFNODE_

#include "VrmlField.h"

namespace vrml
{

class VrmlNode;

class VRMLEXPORT VrmlSFNode : public VrmlSField
{
public:
    VrmlSFNode(VrmlNode *value = 0);
    VrmlSFNode(const VrmlSFNode &n);
    ~VrmlSFNode();

    // Assignment.
    VrmlSFNode &operator=(const VrmlSFNode &rhs);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFNode *toSFNode() const;
    virtual VrmlSFNode *toSFNode();

    VrmlNode *get(void) const
    {
        return d_value;
    }

    void set(VrmlNode *value);

private:
    VrmlNode *d_value;
};
}
#endif //_VRMLSFNODE_
