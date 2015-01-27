/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS ResultMesh
//
//  This abstraction is used by way of trasition between
//  InstanceMesh and COVISE objects. The key difference is that
//  in ResultMesh labels are consecutive.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _RESULT_MESH_H_
#define _RESULT_MESH_H_

#include <util/coviseCompat.h>
#include <api/coModule.h>
using namespace covise;

class Data;

class ResultMesh
{
public:
    ResultMesh(const vector<int> &elem_data,
               const vector<const Data *> &data_per_element_or_node,
               const vector<int> &conn_data,
               const vector<int> &type_data,
               const vector<float> &node_data_x,
               const vector<float> &node_data_y,
               const vector<float> &node_data_z,
               const vector<float> &node_data_disp_x,
               const vector<float> &node_data_disp_y,
               const vector<float> &node_data_disp_z,
               const vector<int> &elem_no_data,
               const vector<int> &conn_no_data,
               const vector<int> &type_no_data,
               const vector<float> &node_no_data_x,
               const vector<float> &node_no_data_y,
               const vector<float> &node_no_data_z,
               const vector<float> &node_no_data_disp_x,
               const vector<float> &node_no_data_disp_y,
               const vector<float> &node_no_data_disp_z);
    virtual ~ResultMesh();
    static void GetObjects(const vector<ResultMesh *> &results,
                           vector<coDistributedObject *> &mesh,
                           vector<coDistributedObject *> &data,
                           vector<coDistributedObject *> &part_indices,
                           vector<coDistributedObject *> &disp,
                           const string &mesh_name,
                           const string &data_name,
                           const string &part_indices_name,
                           const string &disp_name,
                           const vector<int> &instance_labels);

protected:
private:
    // data
    vector<int> _elem_data;
    vector<int> _conn_data;
    vector<int> _type_data;
    vector<float> _node_data_x;
    vector<float> _node_data_y;
    vector<float> _node_data_z;
    vector<float> _node_data_disp_x;
    vector<float> _node_data_disp_y;
    vector<float> _node_data_disp_z;

    vector<const Data *> _data;

    // no data
    vector<int> _elem_no_data;
    vector<int> _conn_no_data;
    vector<int> _type_no_data;
    vector<float> _node_no_data_x;
    vector<float> _node_no_data_y;
    vector<float> _node_no_data_z;
    vector<float> _node_no_data_disp_x;
    vector<float> _node_no_data_disp_y;
    vector<float> _node_no_data_disp_z;

    enum
    {
        DATA_PER_ELEMENT,
        DATA_PER_NODE
    } _data_per;
};
#endif
