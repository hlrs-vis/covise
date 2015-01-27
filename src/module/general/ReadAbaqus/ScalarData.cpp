/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScalarData.h"
#include <do/coDoData.h>

ScalarData::ScalarData(float scalar)
    : _scalar(scalar)
{
}

ScalarData::~ScalarData()
{
}

Data *
ScalarData::Copy() const
{
    ScalarData *ret = new ScalarData(_scalar);
    ret->_position = _position;
    ret->_node = _node;
    return ret;
}

Data *
ScalarData::Average(const vector<Data *> &other_data) const
{
    float aver = 0.0;
    int i;
    for (i = 0; i < other_data.size(); ++i)
    {
        aver += ((ScalarData *)other_data[i])->_scalar;
    }
    aver /= other_data.size();
    Data *ret = new ScalarData(aver);
    ret->SetPosition(_position);
    return ret;
}

coDistributedObject *
ScalarData::GetNoDummy(const char *name,
                       const vector<const Data *> &datalist) const
{
    vector<float> dataArray;
    int i;
    for (i = 0; i < datalist.size(); ++i)
    {
        dataArray.push_back(((ScalarData *)datalist[i])->_scalar);
    }
    coDoFloat *ret = new coDoFloat(name, dataArray.size());
    float *data_adr;
    ret->getAddress(&data_adr);
    copy(dataArray.begin(), dataArray.end(), data_adr);
    return ret;
}
