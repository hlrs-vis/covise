/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Node.h"
#include "Data.h"
#include <util/coviseCompat.h>
#include "FieldLabel.h"

float Node::DISP_SCALE = 1.0;

Node::Node(int label, const vector<float> &coordinates)
    : _label(label)
    , _result(NULL)
{
    assert(coordinates.size() == 3);
    _coordinates[0] = coordinates[0];
    _coordinates[1] = coordinates[1];
    _coordinates[2] = coordinates[2];
    _displacement[0] = 0.0;
    _displacement[1] = 0.0;
    _displacement[2] = 0.0;
}

Node::~Node()
{
    int i;
    for (i = 0; i < _data.size(); ++i)
    {
        delete _data[i];
    }
    delete _result;
}

Node::Node(const Node &rhs)
    : _label(rhs._label)
{
    _coordinates[0] = rhs._coordinates[0];
    _coordinates[1] = rhs._coordinates[1];
    _coordinates[2] = rhs._coordinates[2];
    _displacement[0] = rhs._displacement[0];
    _displacement[1] = rhs._displacement[1];
    _displacement[2] = rhs._displacement[2];
    int i;
    for (i = 0; i < rhs._data.size(); ++i)
    {
        _data.push_back(rhs._data[i]->Copy());
    }
    if (rhs._result)
    {
        _result = rhs._result->Copy();
    }
    else
    {
        _result = NULL;
    }
}

Node &
    Node::
    operator=(const Node &rhs)
{
    if (this == &rhs)
    {
        return *this;
    }
    _label = rhs._label;
    _coordinates[0] = rhs._coordinates[0];
    _coordinates[1] = rhs._coordinates[1];
    _coordinates[2] = rhs._coordinates[2];
    _displacement[0] = rhs._displacement[0];
    _displacement[1] = rhs._displacement[1];
    _displacement[2] = rhs._displacement[2];
    int i;
    for (i = 0; i < _data.size(); ++i)
    {
        delete _data[i];
    }
    _data.clear();
    for (i = 0; i < rhs._data.size(); ++i)
    {
        _data.push_back(rhs._data[i]->Copy());
    }

    delete _result;
    if (rhs._result)
    {
        _result = rhs._result->Copy();
    }
    else
    {
        _result = NULL;
    }
    return *this;
}

bool
    Node::
    operator==(const Node &rhs) const
{
    return (_label == rhs._label);
}

bool
    Node::
    operator==(int rhs) const
{
    return (_label == rhs);
}

bool
    Node::
    operator<(const Node &rhs) const
{
    return (_label < rhs._label);
}

bool
    Node::
    operator<(int rhs) const
{
    return (_label < rhs);
}

void
Node::SetDisplacement(const vector<int> &order,
                      const odb_SequenceFloat &data)
{
    int j;
    for (j = 0; j < data.size(); ++j)
    {
        int coord = order[j];
        if (coord >= 0)
        {
            _displacement[coord] = data.constGet(j);
        }
    }
}

void
Node::AddData(Data *data)
{
    _data.push_back(data);
}

void
Node::Result()
{
    if (_data.size() > 0)
    {
        assert(_result == NULL);
        _result = _data[0]->Average(_data);
    }
}

void
Node::Result(Data *someData)
{
    delete _result;
    _result = someData;
}

const Data *
Node::GetData() const
{
    return _result;
}

int
Node::label() const
{
    return _label;
}

void
Node::Coordinates(float &x, float &y, float &z) const
{
    x = _coordinates[0] + DISP_SCALE * _displacement[0];
    y = _coordinates[1] + DISP_SCALE * _displacement[1];
    z = _coordinates[2] + DISP_SCALE * _displacement[2];
}

void
Node::Displacements(float &x, float &y, float &z) const
{
    x = _displacement[0];
    y = _displacement[1];
    z = _displacement[2];
}

bool
operator<(int lhs, const Node &node)
{
    return (lhs < node._label);
}

void
Node::NodeData(vector<int> &dataLabels) const
{
    if (_result)
    {
        dataLabels.push_back(_label);
    }
}
