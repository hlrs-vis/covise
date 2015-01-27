/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "InstanceMesh.h"
#include "ScalarData.h"
#include "LocalCS_Data.h"
#include "VectorData.h"
#include "TensorData.h"
#include "ResultMesh.h"
#include <algorithm>

using std::lower_bound;
using std::sort;

InstanceMesh::InstanceMesh()
{
}

InstanceMesh::~InstanceMesh()
{
}

InstanceMesh::InstanceMesh(const InstanceMesh &rhs)
    : _nodes(rhs._nodes)
    , _elements(rhs._elements)
{
}

InstanceMesh &
    InstanceMesh::
    operator=(const InstanceMesh &rhs)
{
    if (this == &rhs)
    {
        return *this;
    }
    _nodes = rhs._nodes;
    _elements = rhs._elements;
    return *this;
}

void
InstanceMesh::AddNode(int label, const vector<float> &coordinates)
{
    _nodes.push_back(Node(label, coordinates));
}

void
InstanceMesh::OrderNodesAndElements()
{
    sort(_nodes.begin(), _nodes.end());
    sort(_elements.begin(), _elements.end());
}

void
InstanceMesh::AddElement(int label,
                         const char *type,
                         const vector<int> &connectivity)
{
    _elements.push_back(Element(label, type, connectivity));
}

void
InstanceMesh::ReadDisplacement(int node,
                               const vector<int> &order,
                               const odb_SequenceFloat &data)
{
    vector<Node>::iterator it = lower_bound(_nodes.begin(), _nodes.end(), node);
    it->SetDisplacement(order, data);
}

void
InstanceMesh::ReadStatus(int elementLabel, const odb_SequenceFloat &data)
{
    if (data.constGet(0) < 1)
    {
        vector<Element>::iterator it = lower_bound(_elements.begin(), _elements.end(), elementLabel);
        if (*it == elementLabel)
        {
            _elements.erase(it);
        }
    }
}

void
InstanceMesh::ReadInvariant(const odb_FieldValue &f, INV_FUNC inv_func)
{
    const odb_FieldValue *pf = &f;
    float invariant = (pf->*inv_func)();
    Data *scalarData = new ScalarData(invariant);
    ReadData(scalarData, f);
}

void
InstanceMesh::ReadComponent(const odb_FieldValue &f, int dataposition,
                            bool conjugate)
{
    odb_SequenceFloat data;
    if (conjugate)
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
    if (dataposition >= 0 && dataposition < data.size())
    {
        Data *scalarData = new ScalarData(data.constGet(dataposition));
        ReadData(scalarData, f);
    }
}

void
InstanceMesh::ReadGlobal(const odb_FieldValue &f, int dataposition,
                         const ComponentTranslator &ct, bool conjugate)
{
    NonScalarData *globalVar = NULL;

    switch (Data::TYPE)
    {
    case Data::VECTOR:
        globalVar = new VectorData(f, ct, conjugate);
        break;
    case Data::TENSOR:
        globalVar = new TensorData(f, ct, conjugate);
        break;
    }
    globalVar->Globalise();

    if (dataposition >= 0)
    {
        dataposition = ct[dataposition];
        Data *scalarData = new ScalarData(globalVar->GetComponent(dataposition));
        delete globalVar;
        ReadData(scalarData, f);
    }
    else
    {
        ReadData(globalVar, f);
    }
}

void
InstanceMesh::ReadLocalReferenceSystem(const odb_FieldValue &f)
{
    odb_SequenceSequenceFloat lcs = f.localCoordSystem();
    int lcs_size = lcs.size();
    vector<float> ref;
    int i;
    for (i = 0; i < lcs.size(); ++i)
    {
        odb_SequenceFloat d = lcs.constGet(i);
        int j;
        for (j = 0; j < d.size(); ++j)
        {
            ref.push_back(d.constGet(j));
        }
    }
    if (ref.size() == 9 || ref.size() == 4)
    {
        Data *lCSData = new LocalCS_Data(ref);
        ReadData(lCSData, f);
    }
}

void
InstanceMesh::ReadData(Data *data, const odb_FieldValue &f)
{
    odb_Enum::odb_ResultPositionEnum position = f.position();
    data->SetPosition(position);

    switch (position)
    {
    case odb_Enum::NODAL:
    {
        int node = f.nodeLabel();
        vector<Node>::iterator it = lower_bound(_nodes.begin(), _nodes.end(), node);
        data->SetNode(node);
        it->AddData(data);
        return;
    }
    break;
    case odb_Enum::ELEMENT_NODAL:
    {
        int node = f.nodeLabel();
        int elem = f.elementLabel();
        vector<Element>::iterator it = lower_bound(_elements.begin(), _elements.end(), elem);
        data->SetNode(node);
        if (*it == elem) // element might have been removed
        {
            it->AddData(data);
        }
        return;
    }
    break;
    case odb_Enum::INTEGRATION_POINT:
    case odb_Enum::CENTROID:
    case odb_Enum::WHOLE_ELEMENT:
    {
        int elem = f.elementLabel();
        vector<Element>::iterator it = lower_bound(_elements.begin(), _elements.end(), elem);
        if (*it == elem) // element might have been removed
        {
            it->AddData(data);
        }
        return;
    }
    break;
    case odb_Enum::ELEMENT_FACE:
        break;
    }
    cerr << "Location not supported" << endl;
    delete data;
}

ResultMesh *
InstanceMesh::Result()
{
    int node, elem;
    for (node = 0; node < _nodes.size(); ++node)
    {
        _nodes[node].Result();
    }
    for (elem = 0; elem < _elements.size(); ++elem)
    {
        _elements[elem].Result(_nodes);
    }
    {
        vector<int> elem_data;
        vector<int> elem_no_data;
        vector<int> conn_data;
        vector<int> conn_no_data;
        vector<int> type_data;
        vector<int> type_no_data;
        vector<const Data *> data_per_element;
        int elem;
        for (elem = 0; elem < _elements.size(); ++elem)
        {
            _elements[elem].ElementDataConnectivity(elem_data, elem_no_data,
                                                    conn_data, conn_no_data, type_data, type_no_data,
                                                    data_per_element);
        }
        // node labels are not consecutive!!!
        vector<int> nodal_label_map;
        vector<int> nodes_with_data;
        int node;
        for (node = 0; node < _nodes.size(); ++node)
        {
            nodal_label_map.push_back(_nodes[node].label());
            // get node data in data_per_node
            // and node label in nodes_with_data (growing sequence)
            _nodes[node].NodeData(nodes_with_data);
        }

        vector<float> node_data_x;
        vector<float> node_data_y;
        vector<float> node_data_z;
        vector<float> node_no_data_x;
        vector<float> node_no_data_y;
        vector<float> node_no_data_z;
        vector<float> node_data_disp_x;
        vector<float> node_data_disp_y;
        vector<float> node_data_disp_z;
        vector<float> node_no_data_disp_x;
        vector<float> node_no_data_disp_y;
        vector<float> node_no_data_disp_z;
        if (nodes_with_data.size() == 0) // data per element
        {

            // conn_data and conn_no_data have to be reduced
            Reduce(nodal_label_map, conn_data, NULL,
                   node_data_x, node_data_y, node_data_z,
                   node_data_disp_x, node_data_disp_y, node_data_disp_z);
            Reduce(nodal_label_map, conn_no_data, NULL,
                   node_no_data_x, node_no_data_y, node_no_data_z,
                   node_no_data_disp_x, node_no_data_disp_y, node_no_data_disp_z);

            ResultMesh *ret = new ResultMesh(elem_data, data_per_element, conn_data,
                                             type_data,
                                             node_data_x, node_data_y, node_data_z,
                                             node_data_disp_x, node_data_disp_y, node_data_disp_z,
                                             elem_no_data, conn_no_data,
                                             type_no_data,
                                             node_no_data_x, node_no_data_y, node_no_data_z,
                                             node_no_data_disp_x, node_no_data_disp_y, node_no_data_disp_z);
            return ret;
        }
        else // nodal data may be a bit trickier...
        {
            assert(elem_data.size() == 0 && conn_data.size() == 0 && type_data.size() == 0);
            // we have to distinguish between elements for which
            // all nodes have data and the rest
            int elem;
            vector<int> elem_nodal_data;
            vector<int> elem_no_nodal_data;
            vector<int> conn_nodal_data;
            vector<int> conn_no_nodal_data;
            vector<int> type_nodal_data;
            vector<int> type_no_nodal_data;

            vector<int> added_nodes_with_data;

            for (elem = 0; elem < elem_no_data.size(); ++elem)
            {
                if (_elements[elem].SomeNodesHaveData(nodes_with_data, added_nodes_with_data))
                {
                    _elements[elem].NodalDataConnectivity(elem_nodal_data, conn_nodal_data,
                                                          type_nodal_data);
                }
                else
                {
                    _elements[elem].NodalDataConnectivity(elem_no_nodal_data, conn_no_nodal_data,
                                                          type_no_nodal_data);
                }
            }
            // make sure that nodes in added_nodes_with_data get "invisible" data
            int new_data_node;
            for (new_data_node = 0; new_data_node < added_nodes_with_data.size(); ++new_data_node)
            {
                // find those nodes and give them an invisible value
                int label = added_nodes_with_data[new_data_node];
                Data *invisible = Data::Invisible();
                vector<Node>::iterator it = lower_bound(_nodes.begin(), _nodes.end(), label);
                it->Result(invisible);
            }

            vector<const Data *> nodal_data;
            Reduce(nodal_label_map, conn_nodal_data,
                   &nodal_data, node_data_x, node_data_y, node_data_z,
                   node_data_disp_x, node_data_disp_y, node_data_disp_z);
            Reduce(nodal_label_map, conn_no_nodal_data, NULL,
                   node_no_data_x, node_no_data_y, node_no_data_z,
                   node_no_data_disp_x, node_no_data_disp_y, node_no_data_disp_z);

            ResultMesh *ret = new ResultMesh(elem_nodal_data, nodal_data, conn_nodal_data,
                                             type_nodal_data,
                                             node_data_x, node_data_y, node_data_z,
                                             node_data_disp_x, node_data_disp_y, node_data_disp_z,
                                             elem_no_nodal_data, conn_no_nodal_data,
                                             type_no_nodal_data,
                                             node_no_data_x, node_no_data_y, node_no_data_z,
                                             node_no_data_disp_x, node_no_data_disp_y, node_no_data_disp_z);
            return ret;
        }
    }
    return NULL;
}

void
InstanceMesh::Reduce(const vector<int> &nodal_label_map,
                     vector<int> &conn_info,
                     vector<const Data *> *nodal_data,
                     vector<float> &node_x,
                     vector<float> &node_y,
                     vector<float> &node_z,
                     vector<float> &node_disp_x,
                     vector<float> &node_disp_y,
                     vector<float> &node_disp_z) const
{
    // mark_node: array as long as the number of nodes
    // indicating which nodes are used
    vector<int> mark_node;
    int node;
    for (node = 0; node < _nodes.size(); ++node)
    {
        mark_node.push_back(-1);
    }
    int conn;
    for (conn = 0; conn < conn_info.size(); ++conn)
    {
        int not_consec_lab = conn_info[conn];
        vector<int>::const_iterator it = lower_bound(nodal_label_map.begin(),
                                                     nodal_label_map.end(), not_consec_lab);
        int consec_lab = it - nodal_label_map.begin();
        mark_node[consec_lab] = 0;
    }
    // now we mark used nodes with the final, reduced label
    int consec_lab = 0;
    for (node = 0; node < _nodes.size(); ++node)
    {
        if (mark_node[node] == 0)
        {
            mark_node[node] = consec_lab;
            ++consec_lab;
        }
    }
    // now we may reduce:
    // nodal_label_map maps from original labels to
    // the space of consecutive unreduced labels
    // and mark_node from this space into the space
    // of consecutive, reduced labels.
    for (conn = 0; conn < conn_info.size(); ++conn)
    {
        int not_consec_lab = conn_info[conn];
        vector<int>::const_iterator it = lower_bound(nodal_label_map.begin(),
                                                     nodal_label_map.end(), not_consec_lab);
        int consec_lab = it - nodal_label_map.begin();
        conn_info[conn] = mark_node[consec_lab];
    }
    // getting coordinates:
    // only nodes surviving reduction are taken
    // into account
    for (node = 0; node < _nodes.size(); ++node)
    {
        if (mark_node[node] >= 0)
        {
            float x, y, z;
            _nodes[node].Coordinates(x, y, z);
            float xd, yd, zd;
            _nodes[node].Displacements(xd, yd, zd);
            node_x.push_back(x);
            node_y.push_back(y);
            node_z.push_back(z);
            node_disp_x.push_back(xd);
            node_disp_y.push_back(yd);
            node_disp_z.push_back(zd);
        }
    }
    if (nodal_data)
    {
        for (node = 0; node < _nodes.size(); ++node)
        {
            if (mark_node[node] >= 0)
            {
                const Data *nodalData = _nodes[node].GetData();
                assert(nodalData != NULL);
                nodal_data->push_back(nodalData);
            }
        }
    }
}
