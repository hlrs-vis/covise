/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NonScalarData.h"
#include "LocalCS_Data.h"

NonScalarData::NonScalarData(const odb_FieldValue &f,
                             const ComponentTranslator &ct,
                             int dim, bool conj)
    : _dim(dim)
{
    int i;
    odb_SequenceFloat data;
    if (conj)
    {
        if (f.precision() == odb_Enum::DOUBLE_PRECISION)
        {
            data = f.conjugateDataDouble();
            if (data.size() == 0)
            {
                data = f.dataDouble();
            }
        }
        else
        {
            data = f.conjugateData();
            if (data.size() == 0)
            {
                data = f.data();
            }
        }
    }
    else
    {
        if (f.precision() == odb_Enum::DOUBLE_PRECISION)
        {
            data = f.dataDouble();
        }
        else
        {
            data = f.data();
        }
    }
    _Field = new float[dim];
    for (i = 0; i < dim; ++i)
    {
        _Field[i] = 0.0;
    }
    for (i = 0; i < dim; ++i)
    {
        int pos = ct[i];
        if (pos >= 0 && pos < data.size())
        {
            _Field[pos] = data.constGet(i);
        }
    }
    // now check for the presence of a local reference system
    odb_SequenceSequenceFloat lcs = f.localCoordSystem();
    vector<float> refsys;
    for (i = 0; i < lcs.size(); i++)
    {
        odb_SequenceFloat d = lcs.constGet(i);
        int j;
        for (j = 0; j < d.size(); j++)
        {
            refsys.push_back(d.constGet(j));
        }
    }
    _localRef = new LocalCS_Data(refsys);
}

NonScalarData::~NonScalarData()
{
    delete _Field;
    delete _localRef;
}

float
NonScalarData::GetComponent(int i) const
{
    if (i < 0 || i >= _dim)
    {
        return 0.0;
    }
    return _Field[i];
}

NonScalarData::NonScalarData(int dim)
    : _Field(NULL)
    , _localRef(NULL)
    , _dim(dim)
{
    _Field = new float[dim];
    int i;
    for (i = 0; i < dim; ++i)
    {
        _Field[i] = 0.0;
    }
}
