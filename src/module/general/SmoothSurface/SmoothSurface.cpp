/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface smoothing application module              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1998                                   **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 ** Date:  August 1998  V1.0                                               **
\**************************************************************************/

#include <api/coSimpleModule.h>
#include "SmoothSurface.h"
#include <util/coviseCompat.h>
#include <do/coDoTriangleStrips.h>

int *vl_in, *pl_in;
int set_num_elem = 0;

int method;
int iterations;
float lambda;
float mu;

float *x_in;
float *y_in;
float *z_in;

char *dtype, *ntype;
char *MeshIn, *MeshOut;
char *colorn;
const char *color[10] = {
    "yellow", "green", "blue", "red", "violet", "chocolat",
    "linen", "pink", "crimson", "indigo"
};

//  Shared memory data
const coDoPolygons *mesh_in = NULL;
coDoPolygons *mesh_out = NULL;
const coDoTriangleStrips *tmesh_in = NULL;

SmoothSurfaceModule::SmoothSurfaceModule(int argc, char **argv)
    : coSimpleModule(argc, argv, "Smoothing of a polygonal surface")
{
    pMeshIn = addInputPort("meshIn", "Polygons|TriangleStrips", "Geometry");
    pMeshOut = addOutputPort("meshOut", "Polygons", "The reduced geometry");

    pMethod = addChoiceParam("method", "Which method to use");
    const char *methods[] = { "Gaussian", "Taubin", "Uwe" };
    pMethod->setValue(3, methods, 0);
    pIterations = addInt32Param("iterations", "Number of filtering iterations");
    pIterations->setValue(10);
    pScale1 = addFloatParam("scale_1", "Scale factor for smoothing (between 0 and 1)");
    pScale1->setValue(1.0f / 3.0f);
    pScale2 = addFloatParam("scale_2", "Scale factor for Taubin's second pass smoothing (between -1 and 0)");
    pScale2->setValue(-0.35f);
}

//======================================================================
// Computation routine (called when START message arrives)
//======================================================================
int SmoothSurfaceModule::compute(const char *)
{
    const char *gtype = "";
    //	get parameter
    method = pMethod->getValue();
    iterations = pIterations->getValue();
    lambda = pScale1->getValue();
    mu = pScale2->getValue();

    if ((lambda >= 1.0) || (lambda <= 0))
    {
        sendWarning("Parameter 'scale_1' out of range, set to default");
        lambda = 0.35f;
    }
    if (method == 1)
    {
        if ((mu >= -lambda) || (mu <= -1))
        {
            sendWarning("Parameter 'scale_2' out of range, set to default");
            mu = -lambda - 0.01f;
        }
    }
    if (iterations <= 0)
    {
        sendWarning("Parameter 'iterations' out of range, set to default");
        iterations = 10;
    }

    int n_pnt, n_vert, n_poly;
    SmoothSurface *surf = NULL;

    const coDistributedObject *mesh_object = pMeshIn->getCurrentObject();
    if (mesh_object == NULL)
        return STOP_PIPELINE;

    if (mesh_object->objectOk())
    {
        if ((mesh_in = dynamic_cast<const coDoPolygons *>(mesh_object)))
        {
            gtype = "POLYGN";
            n_pnt = mesh_in->getNumPoints();
            n_vert = mesh_in->getNumVertices();
            n_poly = mesh_in->getNumPolygons();
            mesh_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            if ((colorn = (char *)mesh_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "yellow");
            }
        }
        else if ((tmesh_in = dynamic_cast<const coDoTriangleStrips *>(mesh_object)))
        {
            gtype = "TRIANG";
            n_pnt = tmesh_in->getNumPoints();
            n_vert = tmesh_in->getNumVertices();
            n_poly = tmesh_in->getNumStrips();
            tmesh_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            if ((colorn = (char *)tmesh_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "yellow");
            }
        }
        else
        {
            sendError("ERROR: Data object 'meshIn' has wrong data type");
            return STOP_PIPELINE;
        }
    }
    else
    {
        sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return STOP_PIPELINE;
    }

    if (n_pnt == 0 || n_vert == 0 || n_poly == 0)
    {
        sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    surf = new SmoothSurface(n_pnt, n_vert, n_poly, gtype, pl_in, vl_in, x_in, y_in, z_in, NULL, NULL, NULL);
    surf->Set_Iterations(iterations);
    surf->Set_Scale_1(lambda);
    surf->Set_Scale_2(mu);
    switch (method)
    {
    case 0:
        surf->Smooth_Gaussian();
        break;

    case 1:
        surf->Smooth_Taubin();
        break;
    case 2:
        surf->Smooth_Uwe();
        break;

    default:
        sendError("ERROR: No smoothing method indicated!");
        delete surf;
        return STOP_PIPELINE;
    };

    coDistributedObject *meshOut = surf->createDistributedObjects(pMeshOut->getNewObjectInfo());

    if (!meshOut)
    {
        sendError("ERROR: Creation of output object failed!");

        return STOP_PIPELINE;
    }

    pMeshOut->setCurrentObject(meshOut);

    return CONTINUE_PIPELINE;
}

///////////////////////////////////////////////////////////////////////////
// Derived class SmoothSurface                                           //
///////////////////////////////////////////////////////////////////////////

coDistributedObject *SmoothSurface::createDistributedObjects(coObjInfo Triangle_name)
{
    coDistributedObject *polygons_out = new coDoPolygons(Triangle_name, num_points, coords_x, coords_y, coords_z, num_vertices, vertex_list, num_triangles, tri_list);

    if (!polygons_out->objectOk())
    {
        return NULL;
    }

    return polygons_out;
}

void SmoothSurface::initialize_neighborlist()
{
    int i, j;
    pair edges[MAXTRI];
    int points[MAXTRI];
    int num_edges;

    for (i = 0; i < num_points; i++)
    {
        make_link(i, edges, num_edges);
        extract_points(i, num_edges, edges, link[i].num, points);
        link[i].pnt = new int[link[i].num];
        for (j = 0; j < link[i].num; j++)
            link[i].pnt[j] = points[j];

        delete[] stars[i].tri;
        stars[i].tri = NULL;
        stars[i].num_tri = 0;
    }
}

void SmoothSurface::compute_parameters()
{
}

void SmoothSurface::Gaussian(float lambda)
{
    int i, j;
    float *delta_x, *delta_y, *delta_z;
    float weight;

    delta_x = new float[num_points];
    delta_y = new float[num_points];
    delta_z = new float[num_points];

    for (i = 0; i < num_points; i++)
    {
        weight = 1.0f / (float)link[i].num;
        delta_x[i] = 0.0;
        delta_y[i] = 0.0;
        delta_z[i] = 0.0;
        // if (!stars[i].boundary && stars[i].manifold)
        if (!stars[i].boundary)
        {
            for (j = 0; j < link[i].num; j++)
            {
                delta_x[i] += coords_x[link[i].pnt[j]];
                delta_y[i] += coords_y[link[i].pnt[j]];
                delta_z[i] += coords_z[link[i].pnt[j]];
            }
            delta_x[i] *= weight;
            delta_x[i] -= coords_x[i];
            delta_y[i] *= weight;
            delta_y[i] -= coords_y[i];
            delta_z[i] *= weight;
            delta_z[i] -= coords_z[i];
        }
    }
    for (i = 0; i < num_points; i++)
    {
        coords_x[i] += lambda * delta_x[i];
        coords_y[i] += lambda * delta_y[i];
        coords_z[i] += lambda * delta_z[i];
    }

    delete[] delta_x;
    delete[] delta_y;
    delete[] delta_z;
}

#define LEN_VEC(a, b, c) sqrt(a *a + b * b + c * c)
void SmoothSurface::Uwe(float lambda)
{
    int i, j;
    int p1, p2;
    float *delta_x, *delta_y, *delta_z;
    float weight;
    float a, b;
    float area, areaSum;
    float dx, dy, dz;

    delta_x = new float[num_points];
    delta_y = new float[num_points];
    delta_z = new float[num_points];

    for (i = 0; i < num_points; i++)
    {
        delta_x[i] = 0.0;
        delta_y[i] = 0.0;
        delta_z[i] = 0.0;
        areaSum = 0;
        //if (!stars[i].boundary )
        if (!stars[i].boundary && stars[i].manifold)
        {
            for (j = 0; j < (link[i].num) - 1; j++)
            {
                p1 = link[i].pnt[j];
                p2 = link[i].pnt[j + 1];
                dx = coords_x[i] - coords_x[p1];
                dy = coords_y[i] - coords_y[p1];
                dz = coords_z[i] - coords_z[p1];
                a = LEN_VEC(dx, dy, dz);
                dx = coords_x[p2] - coords_x[p1];
                dy = coords_y[p2] - coords_y[p1];
                dz = coords_z[p2] - coords_z[p1];
                b = LEN_VEC(dx, dy, dz);
                area = a * b / 2.0f;
                areaSum += area;
                delta_x[i] += area * (coords_x[i] + coords_x[p1] + coords_x[p2]) / 3.0f;
                delta_y[i] += area * (coords_y[i] + coords_y[p1] + coords_y[p2]) / 3.0f;
                delta_z[i] += area * (coords_z[i] + coords_z[p1] + coords_z[p2]) / 3.0f;
            }
            weight = 1.0f / areaSum;
            delta_x[i] *= weight;
            delta_x[i] -= coords_x[i];
            delta_y[i] *= weight;
            delta_y[i] -= coords_y[i];
            delta_z[i] *= weight;
            delta_z[i] -= coords_z[i];
        }
    }
    for (i = 0; i < num_points; i++)
    {
        coords_x[i] += lambda * delta_x[i];
        coords_y[i] += lambda * delta_y[i];
        coords_z[i] += lambda * delta_z[i];
    }

    delete[] delta_x;
    delete[] delta_y;
    delete[] delta_z;
}

void SmoothSurface::Smooth_Gaussian()
{
    int i;

    link = new Neighbor[num_points];

    initialize_connectivity();
    for (i = 0; i < num_points; i++)
        initialize_topology(i);
    initialize_neighborlist();

    for (i = 0; i < iterations; i++)
        Gaussian(scale_1);

    for (i = 0; i < num_points; i++)
        delete[] link[i].pnt;
    delete[] link;
}

void SmoothSurface::Smooth_Uwe()
{
    int i;

    link = new Neighbor[num_points];

    initialize_connectivity();
    for (i = 0; i < num_points; i++)
        initialize_topology(i);
    initialize_neighborlist();

    for (i = 0; i < iterations; i++)
        Uwe(scale_1);

    for (i = 0; i < num_points; i++)
        delete[] link[i].pnt;
    delete[] link;
}

void SmoothSurface::Smooth_Taubin()
{
    int i;

    link = new Neighbor[num_points];

    initialize_connectivity();
    for (i = 0; i < num_points; i++)
        initialize_topology(i);
    initialize_neighborlist();
    compute_parameters();

    for (i = 0; i < iterations; i++)
    {
        Gaussian(scale_1);
        Gaussian(scale_2);
    }

    for (i = 0; i < num_points; i++)
        delete[] link[i].pnt;
    delete[] link;
}

MODULE_MAIN(Filter, SmoothSurfaceModule)
