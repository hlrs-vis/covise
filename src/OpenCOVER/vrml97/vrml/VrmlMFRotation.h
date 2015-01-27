/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLMFROTATION_
#define _VRMLMFROTATION_

#include "VrmlField.h"

#include <cstring>

//
// It would be nice to somehow incorporate the reference counting
// into a base class (VrmlMField) or make a VrmlMField template...
// There is no support for copy-on-write, so if you modify an element
// of the data vector, all objects that share that data will see the
// change.
//

namespace vrml
{

class VRMLEXPORT VrmlMFRotation : public VrmlMField
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

        int d_refs; // number of MF* objects using this data
        int d_n; // size (in floats) of d_v
        float *d_v; // data vector
    };

    FData *d_data;

public:
    VrmlMFRotation();
    VrmlMFRotation(float x, float y, float z, float r);
    VrmlMFRotation(int n, float *v);
    VrmlMFRotation(const VrmlMFRotation &src);

    ~VrmlMFRotation();

    virtual std::ostream &print(std::ostream &os) const;

    // Assignment.
    void set(int n, float *v);
    VrmlMFRotation &operator=(const VrmlMFRotation &rhs);

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlMFRotation *toMFRotation() const;
    virtual VrmlMFRotation *toMFRotation();

    int size() const // # of rotations
    {
        return d_data->d_n / 4;
    }
    float *get() const
    {
        return d_data->d_v;
    }
    float *operator[](int index) const
    {
        return &d_data->d_v[4 * index];
    }
    void setSingle(int index, float *value)
    {
        if (4 * index >= d_data->d_n)
        {
            FData *newdata = new FData(4 * (index + 1));
            memcpy(newdata->d_v, d_data->d_v, d_data->d_n * sizeof(float));
            d_data->deref();
            d_data = newdata;
        }
        d_data->d_v[4 * index] = value[0];
        d_data->d_v[4 * index + 1] = value[1];
        d_data->d_v[4 * index + 2] = value[2];
        d_data->d_v[4 * index + 3] = value[3];
    }
};
}
#endif // _VRMLMFROTATION_
