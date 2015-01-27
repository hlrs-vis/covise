/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLMFVEC3D_
#define _VRMLMFVEC3D_

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

class VRMLEXPORT VrmlMFVec3d : public VrmlMField
{
private:
    class FData // reference counted double data
    {
    public:
        FData(int n = 0)
            : d_refs(1)
            , d_n(n)
            , d_v(n > 0 ? new double[n] : 0)
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
        int d_n; // size (in doubles) of d_v
        double *d_v; // data vector
    };

    FData *d_data; // Vec3d data

public:
    VrmlMFVec3d(int n = 0);
    VrmlMFVec3d(double x, double y, double z);
    VrmlMFVec3d(int n, double *v);
    VrmlMFVec3d(const VrmlMFVec3d &source);

    ~VrmlMFVec3d();

    // Assignment.
    void set(int n, double *v);
    VrmlMFVec3d &operator=(const VrmlMFVec3d &rhs);

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlMFVec3d *toMFVec3d() const;
    virtual VrmlMFVec3d *toMFVec3d();

    virtual std::ostream &print(std::ostream &os) const;

    int size() const // # of Vec3ds
    {
        return d_data->d_n / 3;
    }
    double *get() const
    {
        return d_data->d_v;
    }
    double *operator[](int index) const
    {
        return &d_data->d_v[3 * index];
    }
    void setSingle(int index, double *value)
    {
        if (3 * index >= d_data->d_n)
        {
            FData *newdata = new FData(3 * (index + 1));
            memcpy(newdata->d_v, d_data->d_v, d_data->d_n * sizeof(double));
            d_data->deref();
            d_data = newdata;
        }
        d_data->d_v[3 * index] = value[0];
        d_data->d_v[3 * index + 1] = value[1];
        d_data->d_v[3 * index + 2] = value[2];
    }
};
}
#endif // _VRMLMFVEC3D_
