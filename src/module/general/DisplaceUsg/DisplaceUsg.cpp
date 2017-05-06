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
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoData.h>
#include <stdlib.h>

enum Operations
{
    OpIdentity,
    OpSqrt,
    OpLog
};

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
    : coSimpleModule(argc, argv, "Displace vertices of grids")
{
    inMeshPort = addInputPort("GridIn0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid|Polygons|Lines", "Mesh Input");
    inMeshPort->setRequired(1);

    inDataPort = addInputPort("DataIn0", "Float|Vec3", "Data Input");
    inDataPort->setRequired(1);

    paramScale = addFloatParam("scale", "Scaling factor");
    paramScale->setValue(1.0);
    paramAbsolute = addBooleanParam("absolute", "Absolute coordinates");
    paramAbsolute->setValue(false);

    const char *DirChoice[] = { "x-axis", "y-axis", "z-axis" };
    p_direction = addChoiceParam("Direction", "displacement direction for Scalar data");
    p_direction->setValue(3, DirChoice, 0);

    // keep in sync with Operations enum
    const char *OpChoice[] = { "Identity", "Square root", "Log" };
    p_operation = addChoiceParam("Operation", "operation to apply to input data");
    p_operation->setValue(3, OpChoice, 0);

    outMeshPort = addOutputPort("GridOut0", "StructuredGrid|UnstructuredGrid|Polygons|Lines", "Mesh Output");

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
        && !dynamic_cast<const coDoLines *>(mesh_obj)
        && !dynamic_cast<const coDoUniformGrid *>(mesh_obj)
        && !dynamic_cast<const coDoStructuredGrid *>(mesh_obj)
        && !dynamic_cast<const coDoRectilinearGrid *>(mesh_obj))
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
    if (!dynamic_cast<const coDoVec3 *>(data_obj) && !dynamic_cast<const coDoFloat *>(data_obj))
    {
        sendError("Input data may only be USTVDT or USTSDT");
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

    int num_coord=0;
    float *vert_x=NULL, *vert_y=NULL, *vert_z=NULL;
    float *o_vert_x=NULL, *o_vert_y=NULL, *o_vert_z=NULL;
    float *d_x=NULL, *d_y=NULL, *d_z=NULL;

    coDistributedObject *o_mesh = NULL;

    int num_displ = 0;

    const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(m);
    const coDoLines *lines = dynamic_cast<const coDoLines *>(m);
    const coDoAbstractData *displ_data = dynamic_cast<const coDoAbstractData *>(d);
    const coDoVec3 *displ_vec = dynamic_cast<const coDoVec3 *>(d);
    const coDoFloat *displ_scal = dynamic_cast<const coDoFloat *>(d);

    const int operation = p_operation->getValue();

    ////////////////////////////////////////////////////////////////
    /////// Get the data
    if (displ_vec)
    {
        displ_vec->getAddresses(&d_x, &d_y, &d_z);
        num_displ = displ_data->getNumPoints();
    }
    else if (displ_scal)
    {
        const int axis = p_direction->getValue();

        if (axis == 0)
            d_x = displ_scal->getAddress();
        else if (axis == 1)
            d_y = displ_scal->getAddress();
        else if (axis == 2)
            d_z = displ_scal->getAddress();
        num_displ = displ_data->getNumPoints();
    }

    if (const coDoUnstructuredGrid *mesh = dynamic_cast<const coDoUnstructuredGrid *>(m))
    {
        ////////////////////////////////////////////////////////////////
        /////// Get the mesh

        int *elem_list, *conn_list, *type_list;
        int num_elem, num_conn;
        mesh->getGridSize(&num_elem, &num_conn, &num_coord);
        mesh->getAddresses(&elem_list, &conn_list, &vert_x, &vert_y, &vert_z);
        mesh->getTypeList(&type_list);


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
    else if (const coDoAbstractStructuredGrid *gr = dynamic_cast<const coDoAbstractStructuredGrid *>(m))
    {
        const coDoUniformGrid *uni = dynamic_cast<const coDoUniformGrid *>(m);
        const coDoRectilinearGrid *rect = dynamic_cast<const coDoRectilinearGrid *>(m);
        const coDoStructuredGrid *str = dynamic_cast<const coDoStructuredGrid *>(m);

        int nx, ny, nz;
        gr->getGridSize(&nx, &ny, &nz);
        o_mesh = new coDoStructuredGrid(meshName, nx, ny, nz);
        ((coDoStructuredGrid *)o_mesh)->getAddresses(&vert_x, &vert_y, &vert_z);
        num_coord = nx*ny*nz;

        if (uni)
        {
            float dx, dy, dz;
            uni->getDelta(&dx, &dy, &dz);
            float minX, maxX, minY, maxY, minZ, maxZ;
            uni->getMinMax(&minX, &maxX, &minY, &maxY, &minZ, &maxZ);

            for (int ix=0; ix<nx; ++ix)
            {
                for (int iy=0; iy<ny; ++iy)
                {
                    for (int iz=0; iz<nz; ++iz)
                    {
                        int idx = coIndex(ix, iy, iz, nx, ny, nz);
                        vert_x[idx] = minX + ix*dx;
                        vert_y[idx] = minY + iy*dy;
                        vert_z[idx] = minZ + iz*dz;
                    }
                }
            }
            o_vert_x = vert_x;
            o_vert_y = vert_y;
            o_vert_z = vert_z;
        }
        else if (rect)
        {
            float *xc, *yc, *zc;
            rect->getAddresses(&xc, &yc, &zc);
            for (int ix=0; ix<nx; ++ix)
            {
                for (int iy=0; iy<ny; ++iy)
                {
                    for (int iz=0; iz<nz; ++iz)
                    {
                        int idx = coIndex(ix, iy, iz, nx, ny, nz);
                        vert_x[idx] = xc[ix];
                        vert_y[idx] = yc[iy];
                        vert_z[idx] = zc[iz];
                    }
                }
            }
            o_vert_x = vert_x;
            o_vert_y = vert_y;
            o_vert_z = vert_z;
        }
        else if (str)
        {
            str->getAddresses(&o_vert_x, &o_vert_y, &o_vert_z);
        }
    }
    else if (poly || lines)
    {
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
            for (int i = 0; i < num_coord; i++)
            {
                float dx = d_x ? d_x[i] : 0.;
                float dy = d_y ? d_y[i] : 0.;
                float dz = d_z ? d_z[i] : 0.;
                if (operation == OpLog)
                {
                    dx = log(dx);
                    dy = log(dy);
                    dz = log(dz);
                }
                else if (operation == OpSqrt)
                {
                    dx = sqrt(dx);
                    dy = sqrt(dy);
                    dz = sqrt(dz);
                }
                o_vert_x[i] = d_x ? dx : vert_x[i];
                o_vert_y[i] = d_y ? dy : vert_y[i];
                o_vert_z[i] = d_z ? dz : vert_z[i];
            }
        }
        else
        {
            for (int i = 0; i < num_coord; i++)
            {
                float dx = d_x ? d_x[i] : 0.;
                float dy = d_y ? d_y[i] : 0.;
                float dz = d_z ? d_z[i] : 0.;
                if (operation == OpLog)
                {
                    if (dx > 0.)
                        dx = log(fabs(dx));
                    if (dy > 0.)
                        dy = log(fabs(dy));
                    if (dz > 0.)
                        dz = log(fabs(dz));
                }
                else if (operation == OpSqrt)
                {
                    dx = dx >= 0. ? sqrt(dx) : -sqrt(-dx);
                    dy = dy >= 0. ? sqrt(dy) : -sqrt(-dy);
                    dz = dz >= 0. ? sqrt(dz) : -sqrt(-dz);
                }

                o_vert_x[i] = vert_x[i] + s * dx;
                o_vert_y[i] = vert_y[i] + s * dy;
                o_vert_z[i] = vert_z[i] + s * dz;
            }
        }
    }
    else
    {
        sendWarning("No displacements available: returning input grid");
        if (o_vert_x != vert_x)
            memcpy(o_vert_x, vert_x, num_coord * sizeof(float));
        if (o_vert_y != vert_y)
            memcpy(o_vert_y, vert_y, num_coord * sizeof(float));
        if (o_vert_z != vert_z)
            memcpy(o_vert_z, vert_z, num_coord * sizeof(float));
    }

    return o_mesh;
}

MODULE_MAIN(Tools, DisplaceUSG)
