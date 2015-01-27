/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLMFCOLOR_
#define _VRMLMFCOLOR_

#include "VrmlField.h"

//
// It would be nice to somehow incorporate the reference counting
// into a base class (VrmlMField) or make a VrmlMField template...
// There is no support for copy-on-write, so if you modify an element
// of the data vector, all objects that share that data will see the
// change.
//

namespace vrml
{

class VRMLEXPORT VrmlMFColor : public VrmlMField
{
private:
    class FData // reference counted float data
    {
    public:
        FData(int n = 0)
            : d_refs(1)
            , d_n(n)
            , d_v(n > 0 ? new float[n] : 0)
        {
        }
        ~FData()
        {
            delete[] d_v;
        }

        FData *ref()
        {
            ++d_refs;
            return this;
        }
        void deref()
        {
            if (--d_refs == 0)
                delete this;
        }

        int d_refs; // number of objects using this data
        int d_n; // size (in floats) of d_v
        float *d_v; // data vector
    };

    FData *d_data; // Color data in RGB triples

public:
    VrmlMFColor();
    VrmlMFColor(float r, float g, float b);
    VrmlMFColor(int n, float *values);
    VrmlMFColor(const VrmlMFColor &source);

    ~VrmlMFColor();

    virtual std::ostream &print(std::ostream &os) const;

    // Assignment.
    void set(int n, float *v);
    VrmlMFColor &operator=(const VrmlMFColor &rhs);

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlMFColor *toMFColor() const;
    virtual VrmlMFColor *toMFColor();

    int size() const // Number of colors
    {
        return d_data->d_n / 3;
    }
    float *get() const
    {
        return d_data->d_v;
    }
    float *operator[](int i) const
    {
        return &d_data->d_v[3 * i];
    }
};
}
#endif // _VRMLMFCOLOR_
