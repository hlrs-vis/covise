/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLMFBOOL_
#define _VRMLMFBOOL_

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

class VRMLEXPORT VrmlMFBool : public VrmlMField
{
private:
    class BData // reference counted bool data
    {
    public:
        BData(int n = 0)
            : d_refs(1)
            , d_n(n)
            , d_v((n > 0) ? new bool[n] : 0)
        {
        }
        ~BData()
        {
            delete[] d_v;
        }

        BData *ref()
        {
            ++d_refs;
            return this;
        }
        void deref()
        {
            if (--d_refs == 0)
                delete this;
        }

        int d_refs; // number of MFBool objects using this data
        int d_n; // size (in ints) of d_v
        bool *d_v; // data vector
    };

    BData *d_data;

public:
    VrmlMFBool();
    VrmlMFBool(bool value);
    VrmlMFBool(int n, bool *v);
    VrmlMFBool(const VrmlMFBool &src);

    ~VrmlMFBool();

    virtual std::ostream &print(std::ostream &os) const;

    void set(int n, bool *v);
    VrmlMFBool &operator=(const VrmlMFBool &rhs);

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlMFBool *toMFBool() const;
    virtual VrmlMFBool *toMFBool();

    int size() const
    {
        return d_data->d_n;
    }
    bool *get() const
    {
        return d_data->d_v;
    }
    bool &operator[](int i) const
    {
        return d_data->d_v[i];
    }
    void setSingle(int i, bool val)
    {
        if (i < d_data->d_n)
            d_data->d_v[i] = val;
        else
        {
            BData *newdata = new BData(i + 1);
            memcpy(newdata->d_v, d_data->d_v, d_data->d_n * sizeof(int));
            d_data->deref();
            d_data = newdata;
            d_data->d_v[i] = val;
        }
    }
};
}
#endif // _VRMLMFBOOL_
