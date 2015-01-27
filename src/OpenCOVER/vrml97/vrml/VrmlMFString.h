/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLMFSTRING_
#define _VRMLMFSTRING_

#include "VrmlField.h"
#include "string.h"

namespace vrml
{

class VRMLEXPORT VrmlMFString : public VrmlMField
{
public:
    VrmlMFString();
    VrmlMFString(const char *s);
    VrmlMFString(int n, const char **values = 0);
    VrmlMFString(const VrmlMFString &);

    ~VrmlMFString();

    // Assignment. Just reallocate for now...
    void set(int n, const char *v[]);
    VrmlMFString &operator=(const VrmlMFString &rhs);

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlMFString *toMFString() const;
    virtual VrmlMFString *toMFString();

    virtual std::ostream &print(std::ostream &os) const;

    int size() const
    {
        return d_size;
    }
    char **get()
    {
        return &d_v[0];
    }
    char *get(int index)
    {
        return d_v[index];
    }
    char *&operator[](int i)
    {
        return d_v[i];
    }
    void set(int i, const char *value)
    {
        if (i >= d_size)
        {
            char **newdata = new char *[i + 1];
            memcpy(newdata, d_v, d_size * sizeof(char *));
            d_size = i + 1;
            d_v = newdata;
        }
        d_v[i] = new char[strlen(value) + 1];
        strcpy(d_v[i], value);
    }

private:
    char **d_v;
    int d_allocated;
    int d_size;
};
}
#endif //_VRMLMFSTRING_
