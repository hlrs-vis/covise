/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/********************************************************************
 **                                                   (C)1997 RUS  **
 **                                                                **
 ** Description: Module to displace unstructured grids             **
 **                                                                **
 **                                                                **
 **                                                                **
 **                                                                **
 **                                                                **
 ** Author:                                                        **
 **                                                                **
 **                         Reiner Beller                          **
 **            Computer Center University of Stuttgart             **
 **                         Allmandring 30                         **
 **                         70550 Stuttgart                        **
 **                                                                **
 ** Date:  29.08.97  V0.1 					   **
 ** Modified:	06.02.2001 Boris Teplitski			   **
 **			converting to new API			   **
 ********************************************************************/

#include "DisplaceUsg.h"
#include <api/coFeedback.h>
#include <util/coviseCompat.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <stdlib.h>

// macros for error handling
#define ERR0(cond, text, action)     \
    {                                \
        if (cond)                    \
        {                            \
            Covise::sendError(text); \
            {                        \
                action               \
            }                        \
        }                            \
    }

#define ERR1(cond, text, arg1, action)     \
    {                                      \
        if (cond)                          \
        {                                  \
            Covise::sendError(text, arg1); \
            {                              \
                action                     \
            }                              \
        }                                  \
    }

#define ERR2(cond, text, arg1, arg2, action)     \
    {                                            \
        if (cond)                                \
        {                                        \
            Covise::sendError(text, arg1, arg2); \
            {                                    \
                action                           \
            }                                    \
        }                                        \
    }

// p_grid: pointer to object to be replicated
// objName: name of the resulting set
// p_vector: set whose structure we want to replicate
coDoSet *replicate(const coDistributedObject *p_grid,
                   const std::string &objName,
                   const coDoSet *p_vector)
{
    int no_elems;
    coDoSet *ret;
    const coDistributedObject *const *elems = p_vector->getAllElements(&no_elems);
    const coDistributedObject **r_elems = new const coDistributedObject *[no_elems + 1];
    r_elems[no_elems] = 0;
    int i;
    for (i = 0; i < no_elems; ++i)
    {
        if (strcmp(elems[i]->getType(), "SETELE") != 0)
        {
            r_elems[i] = p_grid;
            p_grid->incRefCount();
        }
        else
        {
            std::string elemObjName(objName);
            char buf[16];
            sprintf(buf, "_%d", i);
            elemObjName += buf;
            // it is hardly thinkable that we have sets in sets if the
            // input mesh is a UNSGRD and the data is a set, but OK let us go on...
            r_elems[i] = replicate(p_grid, elemObjName,
                                   (coDoSet *)(elems[i]));
        }
    }
    ret = new coDoSet(objName, r_elems);

// if the set elements are deleted the module crashes sometimes...
#if 0
   for(i=0;i<no_elems;++i)
   {
      delete r_elems[i];
   }
#endif

    delete[] r_elems;
    return ret;
}

void DisplaceUSG::postHandleObjects(coOutputPort **)
{
    const coDistributedObject *p_grid;
    if (p_original_grid_)
    {
        p_grid = inMeshPort->getCurrentObject();
        //      p_grid->destroy();  // is this slippery ground !?
        delete p_grid;
        inMeshPort->setCurrentObject(p_original_grid_);
    }
}

void DisplaceUSG::preHandleObjects(coInputPort **inPorts)
{
    // we want to consider one very special case that is not
    // properly handled by the coSimpleModule class: we have a
    // grid and a set of unstructured vector fields.
    // The motivation for this is that this functionality was
    // already present in the module before coSimpleModule was
    // used and we do not want to downgrade the module. Nevetheless
    // further generalisation could be envisaged: for instance the
    // input grid is a static set of grids, and the data is a dynamic
    // set with sets of data... These situations are left for the future...
    std::string objName;
    coDoSet *p_setGrid;
    const coDistributedObject *p_grid, *p_vector;
    const char *attr;

    p_original_grid_ = 0;
    ++run_count;
    if (inPorts[0] && inPorts[1])
    {
        p_grid = inPorts[0]->getCurrentObject();
        p_vector = inPorts[1]->getCurrentObject();
        if (p_grid && p_vector && p_grid->objectOk() && p_vector->objectOk())
        {
            if ((strcmp(p_grid->getType(), "UNSGRD") == 0
                 || strcmp(p_grid->getType(), "POLYGN") == 0
                 || strcmp(p_grid->getType(), "LINES") == 0)
                && strcmp(p_vector->getType(), "SETELE") == 0)
            {
                p_original_grid_ = p_grid;
                objName = p_grid->getName();
                char buf[16];
                sprintf(buf, "_%d_", run_count);
                objName += buf;
                objName += Covise::get_host();
                objName += "_";
                objName += Covise::get_module();
                objName += "_";
                objName += Covise::get_instance();
                p_setGrid = replicate(p_grid, objName,
                                      (coDoSet *)(p_vector));
                attr = p_vector->getAttribute("TIMESTEP");
                if (attr)
                {
                    p_setGrid->addAttribute("TIMESTEP", attr);
                }
                inPorts[0]->setCurrentObject(p_setGrid);
            }
        }
    }
}

DisplaceUSG::DisplaceUSG(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Displace USG")
{
    inMeshPort = addInputPort("GridIn0", "UnstructuredGrid|Polygons|Lines", "Mesh Input");
    inMeshPort->setRequired(1);

    inDataPort = addInputPort("DataIn0", "Vec3", "Data Input");
    inDataPort->setRequired(1);

    paramScale = addFloatParam("scale", "Scaling factor");
    paramScale->setValue(1.0);
    paramAbsolute = addBooleanParam("absolute", "Absolute coordinates");
    paramAbsolute->setValue(false);

    outMeshPort = addOutputPort("GridOut0", "UnstructuredGrid|Polygons|Lines", "Mesh Output");

    run_count = 0;
}

//void Application::compute(void *)
int DisplaceUSG::compute(const char *)
{
    ////////// Get the grid

    const coDistributedObject *mesh_obj;
    if ((mesh_obj = inMeshPort->getCurrentObject()) == NULL)
    {
        sendError("Error receiving mesh");
        return FAIL;
    }
    if (!dynamic_cast<const coDoUnstructuredGrid *>(mesh_obj)
        && !dynamic_cast<const coDoPolygons *>(mesh_obj)
        && !dynamic_cast<const coDoLines *>(mesh_obj))
    {
        sendError("Input grids may only be UNSGRD or POLYGN or LINES");
        return FAIL;
    }

    ////////// Get the data

    const coDistributedObject *data_obj;
    if ((data_obj = inDataPort->getCurrentObject()) == NULL)
    {
        Covise::sendError("Error receiving dataIn");
        return 0;
    }
    if (!dynamic_cast<const coDoVec3 *>(data_obj))
    {
        sendError("Input data may only be USTVDT");
        return FAIL;
    }

    ////////// Get the scaling factor
    float scale = paramScale->getValue();
    absolute = paramAbsolute->getValue();

    //// create output object name
    const char *out_mesh_name = outMeshPort->getObjName();
    ERR0((out_mesh_name == NULL),
         "Cannot create output name for port 'meshOut'",
         return FAIL;)

    coDistributedObject *res_Mesh = displaceNodes(mesh_obj, data_obj,
                                                  scale, out_mesh_name);
    if (!res_Mesh)
        return FAIL;

    outMeshPort->setCurrentObject(res_Mesh);

    return SUCCESS;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

coDistributedObject *
DisplaceUSG::displaceNodes(const coDistributedObject *m,
                           const coDistributedObject *d,
                           float s,
                           const char *meshName)
{
    if (!m)
        return NULL;

    int i;

    int num_coord;
    float *vert_x, *vert_y, *vert_z;
    float *o_vert_x, *o_vert_y, *o_vert_z;
    float *d_x, *d_y, *d_z;

    coDistributedObject *o_mesh = NULL;

    int num_displ = 0;

    const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(m);
    const coDoLines *lines = dynamic_cast<const coDoLines *>(m);

    if (const coDoUnstructuredGrid *mesh = dynamic_cast<const coDoUnstructuredGrid *>(m))
    {
        const coDoVec3 *displ_data = (const coDoVec3 *)d;

        ////////////////////////////////////////////////////////////////
        /////// Get the mesh

        int *elem_list, *conn_list, *type_list;
        int num_elem, num_conn;
        mesh->getGridSize(&num_elem, &num_conn, &num_coord);
        mesh->getAddresses(&elem_list, &conn_list, &vert_x, &vert_y, &vert_z);
        mesh->getTypeList(&type_list);

        ////////////////////////////////////////////////////////////////
        /////// Get the data
        displ_data->getAddresses(&d_x, &d_y, &d_z);
        num_displ = displ_data->getNumPoints();

        ////////////////////////////////////////////////////////////////
        //////// Check consistency
        // for the moment treat dummy data as error...
        ERR2((num_displ != 0 && num_coord != num_displ),
             "Mesh and Data file not consistent: Nodes %i vs. %i",
             num_coord, num_displ,
             return 0;);

        ////////////////////////////////////////////////////////////////
        ///////  create mesh output
        o_mesh = new coDoUnstructuredGrid(meshName, num_elem, num_conn, num_coord,
                                          mesh->hasTypeList());

        ERR0((o_mesh == NULL),
             "could not create mesh output object", return 0;);

        int *o_elem_list, *o_conn_list, *o_type_list;
        ((coDoUnstructuredGrid *)o_mesh)->getAddresses(&o_elem_list, &o_conn_list, &o_vert_x, &o_vert_y, &o_vert_z);
        if (mesh->hasTypeList())
        {
            ((coDoUnstructuredGrid *)o_mesh)->getTypeList(&o_type_list);
            memcpy(o_type_list, type_list, num_elem * sizeof(int));
        }

        memcpy(o_elem_list, elem_list, num_elem * sizeof(int));
        memcpy(o_conn_list, conn_list, num_conn * sizeof(int));
    }
    else if (poly || lines)
    {
        coDoVec3 *displ_data = (coDoVec3 *)d;

        ////////////////////////////////////////////////////////////////
        /////// Get the mesh

        int nPoints = poly ? poly->getNumPoints() : lines->getNumPoints();
        int nCorners = poly ? poly->getNumVertices() : lines->getNumVertices();
        int nPolygons = poly ? poly->getNumPolygons() : lines->getNumLines();

        num_coord = nPoints;
        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;

        poly ? poly->getAddresses(&vert_x, &vert_y, &vert_z, &cl, &pl) : lines->getAddresses(&vert_x, &vert_y, &vert_z, &cl, &pl);

        ////////////////////////////////////////////////////////////////
        /////// Get the data
        displ_data->getAddresses(&d_x, &d_y, &d_z);
        num_displ = displ_data->getNumPoints();

        ////////////////////////////////////////////////////////////////
        //////// Check consistency
        // for the moment treat dummy data as error...
        ERR2((num_displ != 0 && num_coord != num_displ),
             "Mesh and Data file not consistent: Nodes %i vs. %i",
             num_coord, num_displ,
             return 0;);

        ////////////////////////////////////////////////////////////////
        ///////  create mesh output
        if (poly)
            o_mesh = new coDoPolygons(meshName, nPoints, nCorners, nPolygons);
        else
            o_mesh = new coDoLines(meshName, nPoints, nCorners, nPolygons);

        ERR0((o_mesh == NULL),
             "could not create mesh output object", return 0;);

        int *o_cl, *o_pl;
        poly ? ((coDoPolygons *)o_mesh)->getAddresses(&o_vert_x, &o_vert_y, &o_vert_z, &o_cl, &o_pl) : ((coDoLines *)o_mesh)->getAddresses(&o_vert_x, &o_vert_y, &o_vert_z, &o_cl, &o_pl);

        memcpy(o_cl, cl, nCorners * sizeof(int));
        memcpy(o_pl, pl, nPolygons * sizeof(int));
    }
    else
    {
        cerr << "DisplaceUsg::displaceNodes: unhandled type " << endl;
        return NULL;
    }

    // displace coordinates
    if (num_displ != 0)
    {
        if (absolute)
        {
            for (i = 0; i < num_coord; i++)
            {
                o_vert_x[i] = d_x[i];
                o_vert_y[i] = d_y[i];
                o_vert_z[i] = d_z[i];
            }
        }
        else
        {
            for (i = 0; i < num_coord; i++)
            {
                o_vert_x[i] = vert_x[i] + s * d_x[i];
                o_vert_y[i] = vert_y[i] + s * d_y[i];
                o_vert_z[i] = vert_z[i] + s * d_z[i];
            }
        }
    }
    else
    {
        sendWarning("No displacements available: returning input grid");
        memcpy(o_vert_x, vert_x, num_coord * sizeof(float));
        memcpy(o_vert_y, vert_y, num_coord * sizeof(float));
        memcpy(o_vert_z, vert_z, num_coord * sizeof(float));
    }

    return o_mesh;
}

MODULE_MAIN(Tools, DisplaceUSG)
