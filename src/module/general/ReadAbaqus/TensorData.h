/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS TensorData
//
//  Tensor data representation.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_TENSOR_H_
#define _ABAQUS_TENSOR_H_

#include "NonScalarData.h"

class TensorData : public NonScalarData
{
public:
    TensorData(const odb_FieldValue &f, const ComponentTranslator &ct, bool conj);
    TensorData(float xx, float yy, float zz, float xy, float yz, float zx);
    virtual ~TensorData();
    virtual void Globalise();
    virtual Data *Copy() const;
    virtual Data *Average(const vector<Data *> &other_data) const;

protected:
    TensorData();
    virtual coDistributedObject *GetNoDummy(const char *name,
                                            const vector<const Data *> &datalist) const;

private:
    enum
    {
        XX = 0,
        YY = 1,
        ZZ = 2,
        XY = 3,
        YZ = 4,
        ZX = 5
    };
};
#endif
