/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TensorData.h"
#include "LocalCS_Data.h"

#include <do/coDoData.h>
TensorData::TensorData(const odb_FieldValue &f, const ComponentTranslator &ct,
                       bool conj)
    : NonScalarData(f, ct, 6, conj)
{
}

TensorData::TensorData()
    : NonScalarData(6)
{
}

TensorData::TensorData(float xx, float yy, float zz, float xy, float yz, float zx)
    : NonScalarData(6)
{
    _Field[0] = xx;
    _Field[1] = yy;
    _Field[2] = zz;
    _Field[3] = xy;
    _Field[4] = yz;
    _Field[5] = zx;
}

TensorData::~TensorData()
{
}

void
TensorData::Globalise()
{
    if (!_localRef)
    {
        return;
    }
    float glob[6];
    glob[XX] = _Field[XX] * (*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::XX] + 2.0 * _Field[XY] * (*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::XY] + 2.0 * _Field[ZX] * (*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::XZ] + _Field[YY] * (*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::XY] + 2.0 * _Field[YZ] * (*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::XZ] + _Field[ZZ] * (*_localRef)[LocalCS_Data::XZ] * (*_localRef)[LocalCS_Data::XZ];
    glob[XY] = _Field[XX] * (*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::YX] + _Field[XY] * ((*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::YY] + (*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::YX]) + _Field[ZX] * ((*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::YZ] + (*_localRef)[LocalCS_Data::XZ] * (*_localRef)[LocalCS_Data::YX]) + _Field[YY] * (*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::YY] + _Field[YZ] * ((*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::YZ] + (*_localRef)[LocalCS_Data::XZ] * (*_localRef)[LocalCS_Data::YY]) + _Field[ZZ] * (*_localRef)[LocalCS_Data::XZ] * (*_localRef)[LocalCS_Data::YZ];
    glob[ZX] = _Field[XX] * (*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::ZX] + _Field[XY] * ((*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::ZY] + (*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::ZX]) + _Field[ZX] * ((*_localRef)[LocalCS_Data::XX] * (*_localRef)[LocalCS_Data::ZZ] + (*_localRef)[LocalCS_Data::XZ] * (*_localRef)[LocalCS_Data::ZX]) + _Field[YY] * (*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::ZY] + _Field[YZ] * ((*_localRef)[LocalCS_Data::XY] * (*_localRef)[LocalCS_Data::ZZ] + (*_localRef)[LocalCS_Data::XZ] * (*_localRef)[LocalCS_Data::ZY]) + _Field[ZZ] * (*_localRef)[LocalCS_Data::XZ] * (*_localRef)[LocalCS_Data::ZZ];
    glob[YY] = _Field[XX] * (*_localRef)[LocalCS_Data::YX] * (*_localRef)[LocalCS_Data::YX] + 2.0 * _Field[XY] * (*_localRef)[LocalCS_Data::YX] * (*_localRef)[LocalCS_Data::YY] + 2.0 * _Field[ZX] * (*_localRef)[LocalCS_Data::YX] * (*_localRef)[LocalCS_Data::YZ] + _Field[YY] * (*_localRef)[LocalCS_Data::YY] * (*_localRef)[LocalCS_Data::YY] + 2.0 * _Field[YZ] * (*_localRef)[LocalCS_Data::YY] * (*_localRef)[LocalCS_Data::YZ] + _Field[ZZ] * (*_localRef)[LocalCS_Data::YZ] * (*_localRef)[LocalCS_Data::YZ];
    glob[YZ] = _Field[XX] * (*_localRef)[LocalCS_Data::YX] * (*_localRef)[LocalCS_Data::ZX] + _Field[XY] * ((*_localRef)[LocalCS_Data::YX] * (*_localRef)[LocalCS_Data::ZY] + (*_localRef)[LocalCS_Data::YY] * (*_localRef)[LocalCS_Data::ZX]) + _Field[ZX] * ((*_localRef)[LocalCS_Data::YX] * (*_localRef)[LocalCS_Data::ZZ] + (*_localRef)[LocalCS_Data::YZ] * (*_localRef)[LocalCS_Data::ZX]) + _Field[YY] * (*_localRef)[LocalCS_Data::YY] * (*_localRef)[LocalCS_Data::ZY] + _Field[YZ] * ((*_localRef)[LocalCS_Data::YY] * (*_localRef)[LocalCS_Data::ZZ] + (*_localRef)[LocalCS_Data::YZ] * (*_localRef)[LocalCS_Data::ZY]) + _Field[ZZ] * (*_localRef)[LocalCS_Data::YZ] * (*_localRef)[LocalCS_Data::ZZ];
    glob[ZZ] = _Field[XX] * (*_localRef)[LocalCS_Data::ZX] * (*_localRef)[LocalCS_Data::ZX] + 2.0 * _Field[XY] * (*_localRef)[LocalCS_Data::ZX] * (*_localRef)[LocalCS_Data::ZY] + 2.0 * _Field[ZX] * (*_localRef)[LocalCS_Data::ZX] * (*_localRef)[LocalCS_Data::ZZ] + _Field[YY] * (*_localRef)[LocalCS_Data::ZY] * (*_localRef)[LocalCS_Data::ZY] + 2.0 * _Field[YZ] * (*_localRef)[LocalCS_Data::ZY] * (*_localRef)[LocalCS_Data::ZZ] + _Field[ZZ] * (*_localRef)[LocalCS_Data::ZZ] * (*_localRef)[LocalCS_Data::ZZ];

    _Field[0] = glob[0];
    _Field[1] = glob[1];
    _Field[2] = glob[2];
    _Field[3] = glob[3];
    _Field[4] = glob[4];
    _Field[5] = glob[5];
    delete _localRef;
    _localRef = NULL;
}

Data *
TensorData::Copy() const
{
    TensorData *ret = new TensorData();
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
    std::copy(_Field, _Field + 6, ret->_Field);
    return ret;
}

Data *
TensorData::Average(const vector<Data *> &other_data) const
{
    int i;
    for (i = 0; i < other_data.size(); ++i)
    {
        ((TensorData *)other_data[i])->Globalise();
    }
    TensorData *ret = new TensorData();
    if (other_data.size() == 0)
    {
        return ret;
    }
    for (i = 0; i < other_data.size(); ++i)
    {
        ret->_Field[0] += ((TensorData *)other_data[i])->_Field[0];
        ret->_Field[1] += ((TensorData *)other_data[i])->_Field[1];
        ret->_Field[2] += ((TensorData *)other_data[i])->_Field[2];
        ret->_Field[3] += ((TensorData *)other_data[i])->_Field[3];
        ret->_Field[4] += ((TensorData *)other_data[i])->_Field[4];
        ret->_Field[5] += ((TensorData *)other_data[i])->_Field[5];
    }
    ret->_Field[0] /= other_data.size();
    ret->_Field[1] /= other_data.size();
    ret->_Field[2] /= other_data.size();
    ret->_Field[3] /= other_data.size();
    ret->_Field[4] /= other_data.size();
    ret->_Field[5] /= other_data.size();
    return ret;
}

coDistributedObject *
TensorData::GetNoDummy(const char *name,
                       const vector<const Data *> &datalist) const
{
    coDoTensor *ret = new coDoTensor(name,
                                     datalist.size(), coDoTensor::S3D);
    float *ref;
    ret->getAddress(&ref);
    int i;
    for (i = 0; i < datalist.size(); ++i)
    {
        const TensorData *ThisRef = (const TensorData *)datalist[i];
        *ref = ThisRef->_Field[0];
        ++ref;
        *ref = ThisRef->_Field[1];
        ++ref;
        *ref = ThisRef->_Field[2];
        ++ref;
        *ref = ThisRef->_Field[3];
        ++ref;
        *ref = ThisRef->_Field[4];
        ++ref;
        *ref = ThisRef->_Field[5];
        ++ref;
    }
    return ret;
}
