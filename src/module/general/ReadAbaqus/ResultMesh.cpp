/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "ResultMesh.h"
#include "Data.h"
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
ResultMesh::ResultMesh(const vector<int> &elem_data,
                       const vector<const Data *> &data_per_element,
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
                       const vector<float> &node_no_data_disp_z)
    : _elem_data(elem_data)
    , _conn_data(conn_data)
    , _type_data(type_data)
    , _node_data_x(node_data_x)
    , _node_data_y(node_data_y)
    , _node_data_z(node_data_z)
    , _node_data_disp_x(node_data_disp_x)
    , _node_data_disp_y(node_data_disp_y)
    , _node_data_disp_z(node_data_disp_z)
    , _data(data_per_element)
    , _elem_no_data(elem_no_data)
    , _conn_no_data(conn_no_data)
    , _type_no_data(type_no_data)
    , _node_no_data_x(node_no_data_x)
    , _node_no_data_y(node_no_data_y)
    , _node_no_data_z(node_no_data_z)
    , _node_no_data_disp_x(node_no_data_disp_x)
    , _node_no_data_disp_y(node_no_data_disp_y)
    , _node_no_data_disp_z(node_no_data_disp_z)
    , _data_per(DATA_PER_ELEMENT)
{
}

ResultMesh::~ResultMesh()
{
}

void
ResultMesh::GetObjects(const vector<ResultMesh *> &results,
                       vector<coDistributedObject *> &mesh,
                       vector<coDistributedObject *> &data,
                       vector<coDistributedObject *> &part,
                       vector<coDistributedObject *> &disp,
                       const string &mesh_name,
                       const string &data_name,
                       const string &part_name, const string &disp_name, const vector<int> &instance_labels)
{
    coDistributedObject *meshList[3];
    coDistributedObject *dataList[3];
    coDistributedObject *partList[3];
    coDistributedObject *dispList[3];
    meshList[0] = NULL;
    dataList[0] = NULL;
    partList[0] = NULL;
    dispList[0] = NULL;
    meshList[1] = NULL;
    dataList[1] = NULL;
    partList[1] = NULL;
    dispList[1] = NULL;
    meshList[2] = NULL;
    dataList[2] = NULL;
    partList[2] = NULL;
    dispList[2] = NULL;
    // mesh with data and parts
    {
        vector<int> elements;
        vector<int> connectivities;
        vector<int> elemtypes;
        vector<float> coordX;
        vector<float> coordY;
        vector<float> coordZ;
        vector<float> dispX;
        vector<float> dispY;
        vector<float> dispZ;
        vector<int> partinds;

        int res;
        for (res = 0; res < results.size(); ++res)
        {
            ResultMesh &ThisResult = *results[res];
            int elem;
            int base_conn = connectivities.size();
            for (elem = 0; elem < ThisResult._elem_data.size(); ++elem)
            {
                elements.push_back(base_conn + ThisResult._elem_data[elem]);
                elemtypes.push_back(ThisResult._type_data[elem]);
                partinds.push_back(instance_labels[res]);
            }
            int base_node = coordX.size();
            int conn;
            for (conn = 0; conn < ThisResult._conn_data.size(); ++conn)
            {
                connectivities.push_back(base_node + ThisResult._conn_data[conn]);
            }
            int coord;
            for (coord = 0; coord < ThisResult._node_data_x.size(); ++coord)
            {
                coordX.push_back(ThisResult._node_data_x[coord]);
                coordY.push_back(ThisResult._node_data_y[coord]);
                coordZ.push_back(ThisResult._node_data_z[coord]);
                dispX.push_back(ThisResult._node_data_disp_x[coord]);
                dispY.push_back(ThisResult._node_data_disp_y[coord]);
                dispZ.push_back(ThisResult._node_data_disp_z[coord]);
            }
        }
        string disp_name_data = disp_name;
        disp_name_data += "_Data";
        coDoVec3 *DisplacementsObject = new coDoVec3(disp_name_data.c_str(), coordX.size());
        float *x_disp, *y_disp, *z_disp;
        DisplacementsObject->getAddresses(&x_disp, &y_disp, &z_disp);
        copy(dispX.begin(), dispX.end(), x_disp);
        copy(dispY.begin(), dispY.end(), y_disp);
        copy(dispZ.begin(), dispZ.end(), z_disp);
        dispList[0] = DisplacementsObject;

        string mesh_name_data = mesh_name;
        mesh_name_data += "_Data";
        coDoUnstructuredGrid *DataMesh = new coDoUnstructuredGrid(mesh_name_data.c_str(),
                                                                  elements.size(), connectivities.size(), coordX.size(), 1);
        int *elem_adr, *conn_adr, *type_adr;
        float *x_adr, *y_adr, *z_adr;
        DataMesh->getAddresses(&elem_adr, &conn_adr, &x_adr, &y_adr, &z_adr);
        DataMesh->getTypeList(&type_adr);
        copy(elements.begin(), elements.end(), elem_adr);
        copy(connectivities.begin(), connectivities.end(), conn_adr);
        copy(coordX.begin(), coordX.end(), x_adr);
        copy(coordY.begin(), coordY.end(), y_adr);
        copy(coordZ.begin(), coordZ.end(), z_adr);
        copy(elemtypes.begin(), elemtypes.end(), type_adr);
        meshList[0] = DataMesh;
        meshList[0]->addAttribute("REALTIME", Data::REALTIME.c_str());
        // part
        string part_name_data = part_name;
        part_name_data += "_Data";
        int dimension = partinds.size();
        coDoIntArr *DataPart = new coDoIntArr(part_name_data.c_str(), 1, &dimension);
        int *partArray = DataPart->getAddress();
        copy(partinds.begin(), partinds.end(), partArray);
        partList[0] = DataPart;
    }
    // mesh without data
    {
        vector<int> elements;
        vector<int> connectivities;
        vector<int> elemtypes;
        vector<float> coordX;
        vector<float> coordY;
        vector<float> coordZ;
        vector<float> dispX;
        vector<float> dispY;
        vector<float> dispZ;
        vector<int> partinds;

        int res;
        for (res = 0; res < results.size(); ++res)
        {
            ResultMesh &ThisResult = *results[res];

            int elem;
            int base_conn = connectivities.size();
            for (elem = 0; elem < ThisResult._elem_no_data.size(); ++elem)
            {
                elements.push_back(base_conn + ThisResult._elem_no_data[elem]);
                elemtypes.push_back(ThisResult._type_no_data[elem]);
                partinds.push_back(instance_labels[res]);
            }
            int base_node = coordX.size();
            int conn;
            for (conn = 0; conn < ThisResult._conn_no_data.size(); ++conn)
            {
                connectivities.push_back(base_node + ThisResult._conn_no_data[conn]);
            }
            int coord;
            for (coord = 0; coord < ThisResult._node_no_data_x.size(); ++coord)
            {
                coordX.push_back(ThisResult._node_no_data_x[coord]);
                coordY.push_back(ThisResult._node_no_data_y[coord]);
                coordZ.push_back(ThisResult._node_no_data_z[coord]);
                dispX.push_back(ThisResult._node_no_data_disp_x[coord]);
                dispY.push_back(ThisResult._node_no_data_disp_y[coord]);
                dispZ.push_back(ThisResult._node_no_data_disp_z[coord]);
            }
        }

        string disp_name_nodata = disp_name;
        disp_name_nodata += "_NoData";
        coDoVec3 *DisplacementsObject = new coDoVec3(disp_name_nodata.c_str(), coordX.size());
        float *x_disp, *y_disp, *z_disp;
        DisplacementsObject->getAddresses(&x_disp, &y_disp, &z_disp);
        copy(dispX.begin(), dispX.end(), x_disp);
        copy(dispY.begin(), dispY.end(), y_disp);
        copy(dispZ.begin(), dispZ.end(), z_disp);
        dispList[1] = DisplacementsObject;

        string mesh_name_no_data = mesh_name;
        mesh_name_no_data += "_NoData";
        coDoUnstructuredGrid *NoDataMesh = new coDoUnstructuredGrid(mesh_name_no_data.c_str(),
                                                                    elements.size(), connectivities.size(), coordX.size(), 1);
        int *elem_adr, *conn_adr, *type_adr;
        float *x_adr, *y_adr, *z_adr;
        NoDataMesh->getAddresses(&elem_adr, &conn_adr, &x_adr, &y_adr, &z_adr);
        NoDataMesh->getTypeList(&type_adr);
        copy(elements.begin(), elements.end(), elem_adr);
        copy(connectivities.begin(), connectivities.end(), conn_adr);
        copy(coordX.begin(), coordX.end(), x_adr);
        copy(coordY.begin(), coordY.end(), y_adr);
        copy(coordZ.begin(), coordZ.end(), z_adr);
        copy(elemtypes.begin(), elemtypes.end(), type_adr);
        meshList[1] = NoDataMesh;
        meshList[1]->addAttribute("REALTIME", Data::REALTIME.c_str());
        // part
        string part_name_no_data = part_name;
        part_name_no_data += "_NoData";
        int dimension = partinds.size();
        coDoIntArr *NoDataPart = new coDoIntArr(part_name_no_data.c_str(), 1, &dimension);
        int *partArray = NoDataPart->getAddress();
        copy(partinds.begin(), partinds.end(), partArray);
        partList[1] = NoDataPart;
    }
    coDoSet *MeshAll = new coDoSet(mesh_name.c_str(), meshList);
    delete meshList[0];
    delete meshList[1];
    mesh.push_back(MeshAll);
    coDoSet *PartAll = new coDoSet(part_name.c_str(), partList);
    delete partList[0];
    delete partList[1];
    part.push_back(PartAll);

    coDoSet *DispAll = new coDoSet(disp_name.c_str(), dispList);
    delete dispList[0];
    delete dispList[1];
    disp.push_back(DispAll);
    // data
    {
        vector<const Data *> AllData;
        for (int res = 0; res < results.size(); ++res)
        {
            ResultMesh &ThisResult = *results[res];
            for (int i = 0; i < ThisResult._data.size(); ++i)
            {
                AllData.push_back(ThisResult._data[i]);
            }
        }
        string data_data_name = data_name;
        data_data_name += "_Data";
        dataList[0] = Data::GetObject(data_data_name.c_str(), AllData);
        string no_data_name = data_name;
        no_data_name += "_NoData";
        dataList[1] = Data::GetDummy(no_data_name.c_str());
        if (dataList[0])
        {
            dataList[0]->addAttribute("SPECIES", Data::SPECIES.c_str());
            dataList[0]->addAttribute("REALTIME", Data::REALTIME.c_str());
        }
        if (dataList[1])
        {
            dataList[1]->addAttribute("SPECIES", Data::SPECIES.c_str());
            dataList[1]->addAttribute("REALTIME", Data::REALTIME.c_str());
        }
    }
    coDoSet *DataAll = new coDoSet(data_name.c_str(), dataList);
    delete dataList[0];
    delete dataList[1];
    data.push_back(DataAll);
}
