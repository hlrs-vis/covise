/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLMFDOUBLE_
#define _VRMLMFDOUBLE_

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

class VRMLEXPORT VrmlMFDouble : public VrmlMField
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

        int d_refs; // number of MF* objects using this data
        int d_n; // size (in double) of d_v
        double *d_v; // data vector
    };

    FData *d_data;

public:
    VrmlMFDouble();
    VrmlMFDouble(double value);
    VrmlMFDouble(int n, double *v);
    VrmlMFDouble(const VrmlMFDouble &src);

    ~VrmlMFDouble();

    virtual std::ostream &print(std::ostream &os) const;

    // Assignment.
    void set(int n, double *v);
    VrmlMFDouble &operator=(const VrmlMFDouble &rhs);

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlMFDouble *toMFDouble() const;
    virtual VrmlMFDouble *toMFDouble();

    int size() const
    {
        return d_data->d_n;
    }
    double *get() const
    {
        return d_data->d_v;
    }
    double &operator[](int i)
    {
        return d_data->d_v[i];
    }
    void setSingle(int i, double val)
    {
        if (i < d_data->d_n)
            d_data->d_v[i] = val;
        else
        {
            FData *newdata = new FData(i + 1);
            memcpy(newdata->d_v, d_data->d_v, d_data->d_n * sizeof(double));
            d_data->deref();
            d_data = newdata;
            d_data->d_v[i] = val;
        }
    }
};
}
#endif // _VRMLMFDOUBLE_
