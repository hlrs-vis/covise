/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS VectorData
//
//  Vector data representation.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_VECTOR_DATA_H_
#define _ABAQUS_VECTOR_DATA_H_

#include "NonScalarData.h"

class VectorData : public NonScalarData
{
public:
    VectorData(const odb_FieldValue &f, const ComponentTranslator &ct, bool conj);
    VectorData(float x, float y, float z);
    virtual ~VectorData();
    virtual void Globalise();
    virtual Data *Copy() const;
    virtual Data *Average(const vector<Data *> &other_data) const;

protected:
    VectorData();
    virtual coDistributedObject *GetNoDummy(const char *name,
                                            const vector<const Data *> &datalist) const;

private:
};
#endif
