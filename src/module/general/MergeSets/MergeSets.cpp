/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                (C)2013 Stellba Hydro GmbH & Co. KG  ++
// ++                                                                     ++
// ++ Description:  MergeSets module                                  ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Martin Becker                            ++
// ++                     Stellba Hydro GmbH & Co. KG                     ++
// ++                            Eiffelstr. 4                             ++
// ++                        89542 Herbrechtingen                         ++
// ++                                                                     ++
// ++ Date:  09/2013  V1.0                                                ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "MergeSets.h"

MergeSets::MergeSets(int argc, char *argv[])
    : coSimpleModule(argc, argv, "merge polygons from set into one object")
{

    //parameters

    // none so far ...

    //ports
    p_inGeo = addInputPort("inMesh", "Polygons|UnstructuredGrid", "input mesh");
    p_inGeo->setRequired(1);

    p_inScalarData = addInputPort("inScalarData", "Float", "input data");
    p_inScalarData->setRequired(0);

    p_inVectorData = addInputPort("inVectorData", "Vec3", "input data");
    p_inVectorData->setRequired(0);

    p_outGeo = addOutputPort("outMesh", "Polygons", "output mesh");

    p_outScalarData = addOutputPort("outScalarData", "Float", "output data");

    p_outVectorData = addOutputPort("outVectorData", "Vec3", "output data");
}

MergeSets::~MergeSets()
{
    container_conn_list.clear();
    container_element_list.clear();
    container_type_list.clear();
    container_x.clear();
    container_y.clear();
    container_z.clear();

    container_scalarData.clear();

    container_vectorData_X.clear();
    container_vectorData_Y.clear();
    container_vectorData_Z.clear();
}

void MergeSets::preHandleObjects(coInputPort **InPorts)
{
    (void)InPorts;

    sendInfo("entering preHandleObjects");

    num_points_handled = 0;
    num_corners_handled = 0;
    num_elements_handled = 0;
    hasTypeList = 0;

    num_scalarData_handled = 0;
    num_vectorData_handled = 0;

    num_sets_handled = 0;

    haveScalarData = false;
    haveVectorData = false;
}

int MergeSets::compute(const char *)
{

    sendInfo("entering compute");

    // this is a coSimpleModule: compute is called for each set member
    // here, we extract the polygons from the sets to the objects generated in preHandleObjects
    // covise objects are constructed after compute, in postHandleObjects
    // note that preHandleObjects and postHandleObjects are called only once!

    //get input objects
    const coDistributedObject *geo_obj_in = p_inGeo->getCurrentObject();
    const coDistributedObject *scalar_data_obj_in = p_inScalarData->getCurrentObject();
    const coDistributedObject *vector_data_obj_in = p_inVectorData->getCurrentObject();

    if (!geo_obj_in)
    {
        sendError("Did not receive object at port '%s'", p_inGeo->getName());
        return FAIL;
    }

    // handle geometry (polygons or unstructured grid)
    int *conn_list, *element_list, *type_list;
    float *x, *y, *z;
    int num_points; // length of coordinate array
    int num_corners; // length of connectivity list
    int num_elements; // number of polygons / usg elements

    // handle polygons
    if (const coDoPolygons *polygons = dynamic_cast<const coDoPolygons *>(geo_obj_in))
    {
        havePolygons = true;
        polygons->getAddresses(&x, &y, &z, &conn_list, &element_list);
        num_points = polygons->getNumPoints(); // length of coordinate array
        num_corners = polygons->getNumVertices(); // length of connectivity list
        num_elements = polygons->getNumPolygons(); // number of polygons
    }

    // handle usg
    if (const coDoUnstructuredGrid *usg = dynamic_cast<const coDoUnstructuredGrid *>(geo_obj_in))
    {
        havePolygons = false;
        usg->getAddresses(&element_list, &conn_list, &x, &y, &z);
        usg->getGridSize(&num_elements, &num_corners, &num_points);
        hasTypeList = usg->hasTypeList();
        if (hasTypeList)
        {
            usg->getTypeList(&type_list);
        }
    }

    // add this to our container objects
    for (int i = 0; i < num_points; i++)
    {
        container_x.push_back(x[i]);
        container_y.push_back(y[i]);
        container_z.push_back(z[i]);
    }

    for (int i = 0; i < num_corners; i++)
    {
        container_conn_list.push_back(num_points_handled + conn_list[i]);
    }

    for (int i = 0; i < num_elements; i++)
    {
        container_element_list.push_back(num_corners_handled + element_list[i]);
    }
    if ((!havePolygons) && (hasTypeList))
    {
        for (int i = 0; i < num_elements; i++)
        {
            container_type_list.push_back(type_list[i]);
        }
    }

    num_points_handled += num_points;
    num_corners_handled += num_corners;
    num_elements_handled += num_elements;

    // handle scalar data
    if (scalar_data_obj_in)
    {
        if (const coDoFloat *fdata = dynamic_cast<const coDoFloat *>(scalar_data_obj_in))
        {
            haveScalarData = true;

            float *scalar;
            int num_data;
            num_data = fdata->getNumPoints();
            fdata->getAddress(&scalar);

            // add this to our container objects
            for (int i = 0; i < num_data; i++)
            {
                container_scalarData.push_back(scalar[i]);
            }

            num_scalarData_handled += num_data;
        }
        else
        {
            sendError("Received illegal type at port '%s'", p_inScalarData->getName());
            return FAIL;
        }
    }

    // handle vector data
    if (vector_data_obj_in)
    {
        if (const coDoVec3 *vecData = dynamic_cast<const coDoVec3 *>(vector_data_obj_in))
        {
            haveVectorData = true;

            float *vectorX;
            float *vectorY;
            float *vectorZ;
            int num_data;
            num_data = vecData->getNumPoints();
            vecData->getAddresses(&vectorX, &vectorY, &vectorZ);

            // add this to our container objects
            for (int i = 0; i < num_data; i++)
            {
                container_vectorData_X.push_back(vectorX[i]);
                container_vectorData_Y.push_back(vectorY[i]);
                container_vectorData_Z.push_back(vectorZ[i]);
            }

            num_vectorData_handled += num_data;
        }
        else
        {
            sendError("Received illegal type at port '%s'", p_inVectorData->getName());
            return FAIL;
        }
    }

    num_sets_handled++;

    return SUCCESS;
}

void MergeSets::postHandleObjects(coOutputPort **OutPorts)
{

    sendInfo("entering postHandleObjects");

    (void)OutPorts;

    coDoPolygons *polygons_out;
    coDoUnstructuredGrid *usg_out;
    float *x_merged, *y_merged, *z_merged;
    int *conn_list_merged, *element_list_merged, *type_list_merged;

    if (havePolygons)
    {
        polygons_out = new coDoPolygons(p_outGeo->getObjName(), num_points_handled, num_corners_handled, num_elements_handled);
        polygons_out->getAddresses(&x_merged, &y_merged, &z_merged, &conn_list_merged, &element_list_merged);
    }
    else
    {
        usg_out = new coDoUnstructuredGrid(p_outGeo->getObjName(), num_elements_handled, num_corners_handled, num_points_handled, hasTypeList);
        usg_out->getAddresses(&element_list_merged, &conn_list_merged, &x_merged, &y_merged, &z_merged);
        if (hasTypeList)
        {
            usg_out->getTypeList(&type_list_merged);
        }
    }

    // copy the data ...
    memcpy(x_merged, &container_x[0], num_points_handled * sizeof(float));
    memcpy(y_merged, &container_y[0], num_points_handled * sizeof(float));
    memcpy(z_merged, &container_z[0], num_points_handled * sizeof(float));
    memcpy(conn_list_merged, &container_conn_list[0], num_corners_handled * sizeof(int));
    memcpy(element_list_merged, &container_element_list[0], num_elements_handled * sizeof(int));
    if (hasTypeList)
    {
        memcpy(type_list_merged, &container_type_list[0], num_elements_handled * sizeof(int));
    }

    // handle data
    coDoFloat *scalar_data_out;
    coDoVec3 *vector_data_out;

    if (haveScalarData)
    {
        scalar_data_out = new coDoFloat(p_outScalarData->getObjName(), num_scalarData_handled);
        float *scalar_data_merged;
        scalar_data_out->getAddress(&scalar_data_merged);

        memcpy(scalar_data_merged, &container_scalarData[0], num_scalarData_handled * sizeof(float));
    }

    if (haveVectorData)
    {
        vector_data_out = new coDoVec3(p_outVectorData->getObjName(), num_vectorData_handled);
        float *vector_data_merged_X;
        float *vector_data_merged_Y;
        float *vector_data_merged_Z;
        vector_data_out->getAddresses(&vector_data_merged_X, &vector_data_merged_Y, &vector_data_merged_Z);

        memcpy(vector_data_merged_X, &container_vectorData_X[0], num_vectorData_handled * sizeof(float));
        memcpy(vector_data_merged_Y, &container_vectorData_Y[0], num_vectorData_handled * sizeof(float));
        memcpy(vector_data_merged_Z, &container_vectorData_Z[0], num_vectorData_handled * sizeof(float));
    }

    // clear container data
    container_conn_list.clear();
    container_element_list.clear();
    container_x.clear();
    container_y.clear();
    container_z.clear();

    container_scalarData.clear();

    container_vectorData_X.clear();
    container_vectorData_Y.clear();
    container_vectorData_Z.clear();

    if (havePolygons)
    {
        p_outGeo->setCurrentObject(polygons_out);
    }
    else
    {
        p_outGeo->setCurrentObject(usg_out);
    }

    if (haveScalarData)
    {
        p_outScalarData->setCurrentObject(scalar_data_out);
    }
    if (haveVectorData)
    {
        p_outVectorData->setCurrentObject(vector_data_out);
    }

    sendInfo("MergeSets: handled %d set elements containing %d nodes, %d vertices and %d polygons", num_sets_handled, num_points_handled, num_corners_handled, num_elements_handled);
}

MODULE_MAIN(Tools, MergeSets)
