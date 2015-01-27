/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS InstanceMesh
//
//  Abstraction of a 'simple' mesh with non-consecutive element and node labels.
//  It describes mesh and results for an ABAQUS instance part.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _INSTANCE_MESH_H_
#define _INSTANCE_MESH_H_

#include "Node.h"
#include "Element.h"
#include "FieldLabel.h"
#include "ComponentTranslator.h"

class ResultMesh;

class InstanceMesh
{
public:
    InstanceMesh();
    virtual ~InstanceMesh();
    InstanceMesh(const InstanceMesh &rhs);
    InstanceMesh &operator=(const InstanceMesh &rhs);
    void AddNode(int label, const vector<float> &coordinates);
    void OrderNodesAndElements();
    void AddElement(int label, const char *type, const vector<int> &connectivity);
    void ReadDisplacement(int node,
                          const vector<int> &order,
                          const odb_SequenceFloat &data);

    void ReadStatus(int label, const odb_SequenceFloat &data);
    void ReadInvariant(const odb_FieldValue &f, INV_FUNC inv_func);
    void ReadComponent(const odb_FieldValue &f, int dataposition, bool conjugate);
    void ReadGlobal(const odb_FieldValue &f, int dataposition,
                    const ComponentTranslator &ct, bool conjugate);
    void ReadLocalReferenceSystem(const odb_FieldValue &f);
    ResultMesh *Result();
    void ReadData(Data *data, const odb_FieldValue &f);

protected:
private:
    void Reduce(const vector<int> &nodal_label_map,
                vector<int> &conn_info,
                vector<const Data *> *nodal_data,
                vector<float> &node_data_x,
                vector<float> &node_data_y,
                vector<float> &node_data_z,
                vector<float> &node_data_disp_x,
                vector<float> &node_data_disp_y,
                vector<float> &node_data_disp_z) const;

    vector<Node> _nodes;
    vector<Element> _elements;
};
#endif
