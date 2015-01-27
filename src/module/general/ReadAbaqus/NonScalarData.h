/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS NonScalarData
//
//  Abstract basis class for vector and tensor data
//
//  Initial version: 25.09.2003
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_NON_SCALAR_DATA_H_
#define _ABAQUS_NON_SCALAR_DATA_H_

#include "Data.h"
#include "ComponentTranslator.h"
typedef uint64_t uint64;
typedef int64_t int64;
#include "odb_FieldValue.h"

class LocalCS_Data;

class NonScalarData : public Data
{
public:
    NonScalarData(const odb_FieldValue &f, const ComponentTranslator &ct, int dim,
                  bool conj);
    virtual ~NonScalarData();
    virtual float GetComponent(int i) const;
    virtual void Globalise() = 0;

protected:
    NonScalarData(int dim);
    float *_Field;
    LocalCS_Data *_localRef;
    int _dim;

private:
};
#endif
