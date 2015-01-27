/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Element.h"
#include "Data.h"
#include <ctype.h>
#include <algorithm>
#include <iterator> // std::insert_iterator

using std::binary_search;

#ifdef _STANDARD_C_PLUS_PLUS
#include <iterator.h> // HACK to get insert_iterator
#endif

Element::Element(int label, const char *type, const vector<int> &connectivity)
    : _label(label)
    , _type(type)
    , _shape(TYPE_NONE)
    , _connectivity(connectivity)
    , _result(NULL)
{
    // work out _shape and _reduced_connectivity here
    int nv = -1;
    int nint = -1;
    if (Continuum1D(type, nv))
    {
        int num_nodes = _connectivity.size();
        if (num_nodes >= 2)
        {
            _shape = TYPE_BAR;
            _reduced_connectivity.push_back(_connectivity[0]);
            _reduced_connectivity.push_back(_connectivity[num_nodes - 1]);
        }
    }
    else if (Continuum2D(type, nv))
    {
        switch (nv)
        {
        case 3:
        case 6:
            _shape = TYPE_TRIANGLE;
            CopyConnectivity(3);
            break;
        case 4:
        case 8:
            _shape = TYPE_QUAD;
            CopyConnectivity(4);
            break;
        default:
            break;
        }
    }
    else if (Continuum3D(type, nv))
    {
        switch (nv)
        {
        case 4:
        case 10:
            _shape = TYPE_TETRAHEDER;
            CopyConnectivity(4);
            break;
        case 6:
        case 15:
            _shape = TYPE_PRISM;
            CopyConnectivity(6);
            break;
        case 8:
        case 20:
        case 27:
            _shape = TYPE_HEXAEDER;
            CopyConnectivity(8);
            break;
        default:
            break;
        }
    }
    else if (ContinuumAX(type, nv))
    {
        switch (nv)
        {
        case 2:
            _shape = TYPE_BAR;
            CopyConnectivity(2);
            break;
        case 3:
        case 6:
            _shape = TYPE_TRIANGLE;
            CopyConnectivity(3);
            break;
        case 4:
        case 8:
            _shape = TYPE_QUAD;
            CopyConnectivity(4);
            break;
        default:
            break;
        }
    }
    else if (ContinuumCyl(type, nv))
    {
        switch (nv)
        {
        case 9:
            _shape = TYPE_PRISM;
            CopyConnectivity(6);
            break;
        case 12:
        case 18:
        case 24:
            _shape = TYPE_HEXAEDER;
            CopyConnectivity(8);
            break;
        default:
            break;
        }
    }
    else if (Membrane3D(type, nv))
    {
        switch (nv)
        {
        case 3:
        case 6:
            _shape = TYPE_TRIANGLE;
            CopyConnectivity(3);
            break;
        case 4:
        case 8:
        case 9:
            _shape = TYPE_QUAD;
            CopyConnectivity(4);
            break;
        default:
            break;
        }
    }
    else if (MembraneAX(type, nint)) // optimise
    {
        int num_nodes = _connectivity.size();
        if (num_nodes >= 2)
        {
            _shape = TYPE_BAR;
            _reduced_connectivity.push_back(_connectivity[0]);
            _reduced_connectivity.push_back(_connectivity[num_nodes - 1]);
        }
    }
    else if (MembraneCyl(type, nv))
    {
        _shape = TYPE_QUAD;
        CopyConnectivity(4);
    }
    else if (Truss(type, nv)) // optimise
    {
        int num_nodes = _connectivity.size();
        if (num_nodes >= 2)
        {
            _shape = TYPE_BAR;
            _reduced_connectivity.push_back(_connectivity[0]);
            _reduced_connectivity.push_back(_connectivity[num_nodes - 1]);
        }
    }
    else if (Beam(type, nint)) // optimise
    {
        int num_nodes = _connectivity.size();
        if (num_nodes >= 2)
        {
            _shape = TYPE_BAR;
            _reduced_connectivity.push_back(_connectivity[0]);
            _reduced_connectivity.push_back(_connectivity[num_nodes - 1]);
        }
    }
    else if (Frame(type))
    {
        _shape = TYPE_BAR;
        CopyConnectivity(2);
    }
    else if (Shell3D(type, nv))
    {
        switch (nv)
        {
        case 3:
        case 6:
            _shape = TYPE_TRIANGLE;
            CopyConnectivity(3);
            break;
        case 4:
        case 8:
        case 9:
            _shape = TYPE_QUAD;
            CopyConnectivity(4);
            break;
        default:
            break;
        }
    }
    else if (Rigid(type, nv))
    {
        switch (nv)
        {
        case 2:
            _shape = TYPE_BAR;
            CopyConnectivity(2);
            break;
        case 3:
            _shape = TYPE_TRIANGLE;
            CopyConnectivity(3);
            break;
        case 4:
            _shape = TYPE_QUAD;
            CopyConnectivity(4);
            break;
        default:
            break;
        }
    }
    else if (ShellAX(type, nint)) // optimise
    {
        int num_nodes = _connectivity.size();
        if (num_nodes >= 2)
        {
            _shape = TYPE_BAR;
            _reduced_connectivity.push_back(_connectivity[0]);
            _reduced_connectivity.push_back(_connectivity[num_nodes - 1]);
        }
    }
}

Element::~Element()
{
    int i;
    for (i = 0; i < _data.size(); ++i)
    {
        delete _data[i];
    }
    delete _result;
}

Element::Element(const Element &rhs)
    : _label(rhs._label)
    , _type(rhs._type)
    , _shape(rhs._shape)
    , _connectivity(rhs._connectivity)
    , _reduced_connectivity(rhs._reduced_connectivity)
    , _sub_elements(rhs._sub_elements)
{
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

Element &
    Element::
    operator=(const Element &rhs)
{
    if (this == &rhs)
    {
        return *this;
    }
    _label = rhs._label;
    _type = rhs._type;
    _shape = rhs._shape;
    _connectivity = rhs._connectivity;
    _reduced_connectivity = rhs._reduced_connectivity;
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
    _sub_elements = rhs._sub_elements;

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
    Element::
    operator==(const Element &rhs) const
{
    return (_label == rhs._label);
}

bool
    Element::
    operator<(const Element &rhs) const
{
    return (_label < rhs._label);
}

bool
    Element::
    operator==(int rhs) const
{
    return (_label == rhs);
}

bool
    Element::
    operator<(int rhs) const
{
    return (_label < rhs);
}

void
Element::AddData(Data *data)
{
    _data.push_back(data);
}

void
Element::Result(vector<Node> &nodes) // incomplete: assuming data per element
{
    if (_data.size() == 0)
    {
        return;
    }
    assert(_result == NULL);
    _result = _data[0]->Average(_data);
}

void
Element::ElementDataConnectivity(vector<int> &elem_data,
                                 vector<int> &elem_no_data,
                                 vector<int> &conn_data,
                                 vector<int> &conn_no_data,
                                 vector<int> &type_data,
                                 vector<int> &type_no_data,
                                 vector<const Data *> &data_per_element) const
{
    if (_shape != TYPE_NONE)
    {
        if (_result)
        {
            elem_data.push_back(conn_data.size());
            type_data.push_back(_shape);
            int i;
            for (i = 0; i < _reduced_connectivity.size(); ++i)
            {
                conn_data.push_back(_reduced_connectivity[i]);
            }
            data_per_element.push_back(_result);
        }
        else
        {
            elem_no_data.push_back(conn_no_data.size());
            type_no_data.push_back(_shape);
            int i;
            for (i = 0; i < _reduced_connectivity.size(); ++i)
            {
                conn_no_data.push_back(_reduced_connectivity[i]);
            }
        }
    }
    int subelem;
    for (subelem = 0; subelem < _sub_elements.size(); ++subelem)
    {
        _sub_elements[subelem].ElementDataConnectivity(elem_data, elem_no_data,
                                                       conn_data, conn_no_data,
                                                       type_data, type_no_data,
                                                       data_per_element);
    }
}

void
Element::NodalDataConnectivity(vector<int> &elem,
                               vector<int> &conn,
                               vector<int> &type) const
{
    elem.push_back(conn.size());
    type.push_back(_shape);
    int i;
    for (i = 0; i < _reduced_connectivity.size(); ++i)
    {
        conn.push_back(_reduced_connectivity[i]);
    }
}

bool
Element::AllNodesHaveData(const vector<int> &dataNodeLabels) const
{
    int rednode;
    for (rednode = 0; rednode < _reduced_connectivity.size(); ++rednode)
    {
        if (!binary_search(dataNodeLabels.begin(), dataNodeLabels.end(),
                           _reduced_connectivity[rednode]))
        {
            return false;
        }
    }
    return true;
}

bool
Element::SomeNodesHaveData(const vector<int> &dataNodeLabels,
                           vector<int> &elemNodesWithoutData) const
{
    int rednode;
    bool ret = false;
    vector<int> temporal;
    for (rednode = 0; rednode < _reduced_connectivity.size(); ++rednode)
    {
        if (!binary_search(dataNodeLabels.begin(), dataNodeLabels.end(),
                           _reduced_connectivity[rednode]))
        {
            temporal.push_back(_reduced_connectivity[rednode]);
        }
        else
        {
            ret = true;
        }
    }
    if (ret) // there are some nodes with data... but possibly not all nodes
    {
        std::insert_iterator<vector<int> > ii(elemNodesWithoutData,
                                              elemNodesWithoutData.end());
        std::copy(temporal.begin(), temporal.end(), ii);
    }
    return ret;
}

bool
Element::Continuum1D(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "DC1D", 4) == 0
        || strncmp(type, "AC1D", 4) == 0)
    {
        jump = 4;
    }
    else if (strncmp(type, "DCC1D", 5) == 0)
    {
        jump = 5;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::Continuum2D(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "CPEG", 4) == 0
        || strncmp(type, "DC2D", 4) == 0
        || strncmp(type, "AC2D", 4) == 0)
    {
        jump = 4;
    }
    else if (strncmp(type, "CPE", 3) == 0
             || strncmp(type, "CPS", 3) == 0)
    {
        jump = 3;
    }
    else if (strncmp(type, "DCC2D", 5) == 0)
    {
        jump = 5;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::Continuum3D(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "C3D", 3) == 0)
    {
        jump = 3;
    }
    else if (strncmp(type, "DC3D", 4) == 0
             || strncmp(type, "AC3D", 4) == 0)
    {
        jump = 4;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::ContinuumAX(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "CGAX", 4) == 0
        || strncmp(type, "DCAX", 4) == 0
        || strncmp(type, "ACAX", 4) == 0
        || strncmp(type, "CAXA", 4) == 0)
    {
        jump = 4;
    }
    else if (strncmp(type, "CAX", 3) == 0)
    {
        jump = 3;
    }
    else if (strncmp(type, "DCCAX", 5) == 0)
    {
        jump = 5;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::ContinuumCyl(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "CCL", 3) == 0)
    {
        jump = 3;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::Membrane3D(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "M3D", 3) == 0)
    {
        jump = 3;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::MembraneAX(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "MAX", 3) == 0)
    {
        jump = 3;
    }
    else if (strncmp(type, "MGAX", 4) == 0)
    {
        jump = 4;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::MembraneCyl(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "MCL", 3) == 0)
    {
        jump = 3;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::Truss(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "T3D", 3) == 0
        || strncmp(type, "T2D", 3) == 0)
    {
        jump = 3;
    }
    if (jump == -1 || sscanf(type + jump, "%d", &num) == 0)
    {
        return false;
    }
    return true;
}

bool
Element::Beam(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "B", 1) == 0)
    {
        jump = 2;
    }
    else if (strncmp(type, "PIPE", 4) == 0)
    {
        jump = 5;
    }
    if (jump == -1 || !isdigit(type[jump]))
    {
        return false;
    }
    int val = type[jump];
    int base = '0';
    num = val - base;
    return true;
}

bool
Element::Frame(const char *type)
{
    if (strncmp(type, "FRAME", 5) == 0)
    {
        return true;
    }
    return false;
}

bool
Element::Rigid(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "R2D2", 4) == 0 || strncmp(type, "RAX2", 4) == 0)
    {
        jump = 3;
    }
    else if (strncmp(type, "RB2D2", 5) == 0 || strncmp(type, "RB3D2", 5) == 0)
    {
        jump = 4;
    }
    else if (strncmp(type, "R3D", 3) == 0)
    {
        jump = 3;
    }

    if (jump == -1 || !isdigit(type[jump]))
    {
        return false;
    }
    int val = type[jump];
    int base = '0';
    num = val - base;
    return true;
}

bool
Element::Shell3D(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "STRI", 4) == 0)
    {
        jump = 4;
    }
    else if (strncmp(type, "SC", 2) == 0
             || strncmp(type, "DS", 2) == 0)
    {
        jump = 2;
    }
    else if (strncmp(type, "S", 1) == 0)
    {
        jump = 1;
    }
    if (jump == -1 || !isdigit(type[jump]))
    {
        return false;
    }
    int val = type[jump];
    int base = '0';
    num = val - base;
    return true;
}

bool
Element::ShellAX(const char *type, int &num)
{
    int jump = -1;
    if (strncmp(type, "DSAXA", 5) == 0)
    {
        jump = 5;
    }
    else if (strncmp(type, "SAXA", 4) == 0
             || strncmp(type, "DSAX", 4) == 0)
    {
        jump = 4;
    }
    else if (strncmp(type, "SAX", 3) == 0)
    {
        jump = 3;
    }
    if (jump == -1 || !isdigit(type[jump]))
    {
        return false;
    }
    int val = type[jump];
    int base = '0';
    num = val - base;
    return true;
}

void
Element::CopyConnectivity(int num)
{
    assert(_connectivity.size() >= num);
    int i;
    for (i = 0; i < num; ++i)
    {
        _reduced_connectivity.push_back(_connectivity[i]);
    }
}

bool
operator<(int lhs, const Element &elem)
{
    return (lhs < elem._label);
}
