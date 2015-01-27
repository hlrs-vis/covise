/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLMFNODE_
#define _VRMLMFNODE_

#include "VrmlField.h"
#include "string.h"

namespace vrml
{

class VrmlNode;

class VRMLEXPORT VrmlMFNode : public VrmlMField
{
public:
    VrmlMFNode();
    VrmlMFNode(VrmlNode *value);
    VrmlMFNode(int n, VrmlNode **v);
    VrmlMFNode(const VrmlMFNode &);

    ~VrmlMFNode();

    virtual std::ostream &print(std::ostream &os) const;

    // Assignment. Since the nodes themselves are ref counted,
    // I don't bother trying to share the NodeLists.
    VrmlMFNode &operator=(const VrmlMFNode &rhs);

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlMFNode *toMFNode() const;
    virtual VrmlMFNode *toMFNode();

    int size() const
    {
        return d_size;
    }
    VrmlNode **get()
    {
        return d_v;
    }
    VrmlNode *get(int index)
    {
        return d_v[index];
    }

    // can't use this as lhs for now.
    VrmlNode *operator[](int i) const
    {
        return d_v[i];
    };
    void set(int i, VrmlNode *val)
    {
        if (i < d_size)
            d_v[i] = val;
        else
        {
            VrmlNode **newdata = new VrmlNode *[i + 1];
            memcpy(newdata, d_v, d_size * sizeof(VrmlNode *));
            d_size = i + 1;
            d_v = newdata;
            d_v[i] = val;
        }
    }

    bool exists(VrmlNode *n);

    void addNode(VrmlNode *n);
    void removeNode(VrmlNode *n);

private:
    VrmlNode **d_v;
    int d_allocated;
    int d_size;
};
}
#endif //_VRMLMFNODE_
