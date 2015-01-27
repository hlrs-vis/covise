/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VectorData.h"
#include "LocalCS_Data.h"

#include <do/coDoData.h>
VectorData::VectorData(const odb_FieldValue &f, const ComponentTranslator &ct,
                       bool conj)
    : NonScalarData(f, ct, 3, conj)
{
}

VectorData::VectorData(float x, float y, float z)
    : NonScalarData(3)
{
    _Field[0] = x;
    _Field[1] = y;
    _Field[2] = z;
}

VectorData::~VectorData()
{
}

void
VectorData::Globalise()
{
    if (!_localRef)
    {
        return;
    }
    float glob[3];
    glob[0] = _Field[0] * (*_localRef)[LocalCS_Data::XX] + _Field[1] * (*_localRef)[LocalCS_Data::XY] + _Field[2] * (*_localRef)[LocalCS_Data::XZ];
    glob[1] = _Field[0] * (*_localRef)[LocalCS_Data::YX] + _Field[1] * (*_localRef)[LocalCS_Data::YY] + _Field[2] * (*_localRef)[LocalCS_Data::YZ];
    glob[2] = _Field[0] * (*_localRef)[LocalCS_Data::ZX] + _Field[1] * (*_localRef)[LocalCS_Data::ZY] + _Field[2] * (*_localRef)[LocalCS_Data::ZZ];
    _Field[0] = glob[0];
    _Field[1] = glob[1];
    _Field[2] = glob[2];
    delete _localRef;
    _localRef = NULL;
}

VectorData::VectorData()
    : NonScalarData(3)
{
}

Data *
VectorData::Copy() const
{
    VectorData *ret = new VectorData();
    if (_localRef)
    {
        ret->_localRef = (LocalCS_Data *)(_localRef->Copy());
    }
    else
    {
        ret->_localRef = NULL;
    }
    ret->_dim = _dim;
    ret->_Field = new float[_dim];
    std::copy(_Field, _Field + 3, ret->_Field);
    return ret;
}

Data *
VectorData::Average(const vector<Data *> &other_data) const
{
    int i;
    for (i = 0; i < other_data.size(); ++i)
    {
        ((VectorData *)other_data[i])->Globalise();
    }
    VectorData *ret = new VectorData();
    if (other_data.size() == 0)
    {
        return ret;
    }
    for (i = 0; i < other_data.size(); ++i)
    {
        ret->_Field[0] += ((VectorData *)other_data[i])->_Field[0];
        ret->_Field[1] += ((VectorData *)other_data[i])->_Field[1];
        ret->_Field[2] += ((VectorData *)other_data[i])->_Field[2];
    }
    ret->_Field[0] /= other_data.size();
    ret->_Field[1] /= other_data.size();
    ret->_Field[2] /= other_data.size();
    return ret;
}

coDistributedObject *
VectorData::GetNoDummy(const char *name,
                       const vector<const Data *> &datalist) const
{
    coDoVec3 *ret = new coDoVec3(name, datalist.size());
    float *u, *v, *w;
    ret->getAddresses(&u, &v, &w);
    int i;
    for (i = 0; i < datalist.size(); ++i)
    {
        const VectorData *ThisRef = (const VectorData *)datalist[i];
        u[i] = ThisRef->_Field[0];
        v[i] = ThisRef->_Field[1];
        w[i] = ThisRef->_Field[2];
    }
    return ret;
}
