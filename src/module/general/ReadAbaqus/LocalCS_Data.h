/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS LocalCS_Data
//
//  Local reference system representation.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_LOCAL_CS_DATA_H_
#define _ABAQUS_LOCAL_CS_DATA_H_

#include "Data.h"

class LocalCS_Data : public Data
{
public:
    LocalCS_Data(const vector<float> &ref_system);
    virtual ~LocalCS_Data();
    virtual Data *Copy() const;
    virtual Data *Average(const vector<Data *> &other_data) const;
    float operator[](int i) const;
    enum
    {
        XX = 0,
        XY = 1,
        XZ = 2,
        YX = 3,
        YY = 4,
        YZ = 5,
        ZX = 6,
        ZY = 7,
        ZZ = 8
    };

    class quaternion
    {
    public:
        quaternion(const float rot[9], const quaternion *);
        quaternion(const quaternion &);
        virtual ~quaternion();
        quaternion &operator+=(const quaternion &);
        void GetMatrix(vector<float> &mat) const;
        static float abs2(const quaternion &q1, const quaternion &q2);

    private:
        float _a, _b, _c, _d;
    };

protected:
    virtual coDistributedObject *GetNoDummy(const char *name,
                                            const vector<const Data *> &datalist) const;

private:
    float _ref_system[9];
};
#endif
