/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#
#include "ScalarData.h"
#include "VectorData.h"
#include "TensorData.h"
#include <do/coDoData.h>

using namespace covise;

Data::MyValueType Data::TYPE = Data::UNDEFINED_TYPE;
string Data::SPECIES("None");
string Data::REALTIME("None");

Data::Data()
    : _position(odb_Enum::UNDEFINED_POSITION)
    , _node(-1)
{
}

Data::~Data()
{
}

void
Data::SetPosition(odb_Enum::odb_ResultPositionEnum position)
{
    _position = position;
}

void
Data::SetNode(int node)
{
    _node = node;
}

coDistributedObject *
Data::GetObject(const char *name, const vector<const Data *> &datalist)
{
    if (datalist.size() == 0)
    {
        return GetDummy(name);
    }
    return datalist[0]->GetNoDummy(name, datalist);
}

// incomplete: only scalar case
coDistributedObject *
Data::GetDummy(const char *name)
{
    switch (TYPE)
    {
    case SCALAR:
        return new coDoFloat(name, 0);
    case VECTOR:
        return new coDoVec3(name, 0);
    case REFERENCE_SYSTEM:
        return new coDoMat3(name, 0);
    case TENSOR:
        return new coDoTensor(name, 0, coDoTensor::S3D);
    }
    return NULL;
}

// well, this data is quite visisble, but... what else? kak djelat'?
Data *
Data::Invisible()
{
    switch (TYPE)
    {
    case VECTOR:
        return new VectorData(0.0, 0.0, 0.0);
        break;
    case TENSOR:
        return new TensorData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        break;
    case SCALAR:
    default:
        return new ScalarData(0.0);
        break;
    }
    return NULL;
}
