/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 SGI  **
 **                                                                        **
 ** Description:  COVISE Surface reduction application module  using       **
 **                               Optimizer                                **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author:  Monika Wierse                                                 **
 **                                                                        **
 ** Date:  October 1997  V1.0                                              **
\**************************************************************************/
#define _IEEE 1
#define NUMVERTS 64

#include <appl/ApplInterface.h>
#include "OptimizerReducer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//  Shared memory data
coDoPolygons *mesh_poly_in = NULL;
coDoPolygons *mesh_poly_out = NULL;
coDoTriangleStrips *mesh_tristrip_in = NULL;
coDoTriangleStrips *mesh_tristrip_out = NULL;
coDoFloat *data_in = NULL;
coDoFloat *data_out = NULL;
coDoVec3 *vdata_in = NULL;
coDoVec3 *vdata_out = NULL;

//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//======================================================================
// Computation routine (called when START message arrives)
//======================================================================
void Application::compute(void *)
{

    char *inportnames[] = { "meshIn", "dataIn", NULL };
    char *outportnames[] = { "meshOut", "dataOut", NULL };

    //	get parameter
    Covise::get_scalar_param("percent", &percent);
    Covise::get_scalar_param("weight0", &weight0);
    Covise::get_scalar_param("weight1", &weight1);
    Covise::get_scalar_param("weight2", &weight2);
    Covise::get_scalar_param("alpha", &alpha);
    alpha_1 = 1. - alpha;

    /* The weights are used in the evaluation
        function at each vertex according to:
        weight0*val0+weight1*val1+weight2*val2. Weight0
        is an approximation to the Hausdorff distance from the vertex
        to the mesh obtained after
        the removal of the vertex. Weight1 is an approximation of the
        first derivative of the mesh
        at the vertex. Weight2 is an approximation of the curvature
        of the mesh at the vertex.
        fAngle does not influence the simplification anymore,
        thought its set and getfunctions are
   left in the api for backward compatibility. */
    /* the value alpha is to build a linear combination of the geometry
   criterion and the data criterion. This is necessary for an isosurface
   for instance. Alpha=0 for cutting planes. */

    cerr << "Weight0 " << weight0 << endl;
    cerr << "Weight1 " << weight1 << endl;
    cerr << "Weight2 " << weight2 << endl;

    if ((percent > 100.) || (percent < 0.))
    {
        Covise::sendWarning("Parameter 'percent' out of range, set to default");
        cerr << "Percent: " << percent << endl;
        percent = 30.0;
    }

    mesh_poly_in = NULL;
    mesh_poly_out = NULL;
    mesh_tristrip_in = NULL;
    mesh_tristrip_out = NULL;

    Compute(2, inportnames, outportnames);
}

coDistributedObject **Application::ComputeObject(coDistributedObject **data_in, char **out_name, int num)
{
    coDistributedObject **r;

    r = HandleObjects(data_in[0], data_in[1], out_name[0], out_name[1]);

    return (r);
}

opReal opSRASimplify_vector_data::calculateVtxEval(int index)
{
    int count, i, li;
    opDVector<int> loop;
    opReal vtxvals[4], x, y, z, u, v, w, max, value;

    x = coords_stat[index].vec[0];
    y = coords_stat[index].vec[1];
    z = coords_stat[index].vec[2];
    u = u_in[index];
    v = v_in[index];
    w = w_in[index];

    getVtxEvals(index, vtxvals);
    if (fabs(vtxvals[3] - SRA_EVAL_FAR) < 0.001)
        return SRA_EVAL_FAR; // don't treat boundary points

    if (getOneRing(index, &count, loop))
    {
        max = -1000.;
        for (i = 0; i < count; i++)
        {
            li = loop[i];
            value = (fabs(u - u_in[li]) + fabs(v - v_in[li]) + fabs(w - w_in[li])) / (fabs(x - coords_stat[li].vec[0]) + fabs(y - coords_stat[li].vec[1]) + fabs(z - coords_stat[li].vec[2]));
            if (max < value)
                max = value;
        }
        return (alpha_1 * max + alpha * (weight0 * vtxvals[0] + weight1 * vtxvals[1] + weight2 * vtxvals[2]));
    }
    else
        return SRA_EVAL_FAR;
}

opReal opSRASimplify_scalar_data::calculateVtxEval(int index)
{
    int count, i, li;
    opDVector<int> loop;
    opReal vtxvals[4], x, y, z, s, max, value, res[3];

    x = coords_stat[index].vec[0];
    y = coords_stat[index].vec[1];
    z = coords_stat[index].vec[2];

    s = s_in[index];

    getVtxEvals(index, vtxvals);
    if (fabs(vtxvals[3] - SRA_EVAL_FAR) < 0.001)
        return SRA_EVAL_FAR; // don't treat boundary points

    if (getOneRing(index, &count, loop))
    {
        max = -1000.;
        for (i = 0; i < count; i++)
        {
            li = loop[i];
            value = fabs(s - s_in[li]) / (fabs(x - coords_stat[li].vec[0]) + fabs(y - coords_stat[li].vec[1]) + fabs(z - coords_stat[li].vec[2]));
            if (max < value)
                max = value;
        }
        return (alpha_1 * max + alpha * (weight0 * vtxvals[0] + weight1 * vtxvals[1] + weight2 * vtxvals[2]));
    }
    else
        return SRA_EVAL_FAR;
}

coDistributedObject **Application::HandleObjects(coDistributedObject *mesh_object, coDistributedObject *data_object, char *Mesh_out_name, char *Data_out_name)
{
    int numpoints, numvertices, numprimitives;
    int red_tri, red_points;
    Surface *surf;
    int ndata;
    opSRASimplify *simplifier;
    clock_t starttime, endtime;
    double time;
    char buf[1024];
    int status;
    int new_no_triangles;
    coDistributedObject **DO_return;

    if (mesh_object == NULL)
        return (NULL);

    if (mesh_object->objectOk())
    {
        gtype = mesh_object->getType();

        cout << "GTYPE:" << gtype << endl;

        if (strcmp(gtype, "POLYGN") == 0)
        {
            mesh_poly_in = (coDoPolygons *)mesh_object;
            numpoints = mesh_poly_in->getNumPoints();
            numvertices = mesh_poly_in->get_no_of_vertices();
            numprimitives = mesh_poly_in->get_no_of_polygons();
            mesh_poly_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            if ((colorn = mesh_poly_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "yellow");
            }
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            mesh_tristrip_in = (coDoTriangleStrips *)mesh_object;
            numpoints = mesh_tristrip_in->getNumPoints();
            numvertices = mesh_tristrip_in->get_no_of_vertices();
            numprimitives = mesh_tristrip_in->get_no_of_strips();
            mesh_tristrip_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            if ((colorn = mesh_tristrip_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "yellow");
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return (NULL);
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return (NULL);
    }

    if (numpoints == 0 || numvertices == 0 || numprimitives == 0)
    {
        Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    if (data_object == NULL)
    {
        no_data = 1;
        simplifier = new opSRASimplify;
    }
    else
    {
        no_data = 0;

        if (data_object->objectOk())
        {
            dtype = data_object->getType();
            cout << "DTYPE:" << dtype << endl;
            if (strcmp(dtype, "USTSDT") == 0)
            {
                data_in = (coDoFloat *)data_object;
                ndata = data_in->getNumPoints();
                data_in->getAddress(&s_in);

                simplifier = new opSRASimplify_scalar_data;
            }
            else if (strcmp(dtype, "USTVDT") == 0)
            {
                vdata_in = (coDoVec3 *)data_object;
                ndata = vdata_in->getNumPoints();
                vdata_in->getAddresses(&u_in, &v_in, &w_in);
                simplifier = new opSRASimplify_vector_data;
            }
            else
            {
                Covise::sendError("ERROR: Data object 'DataIn' has wrong data type");
                return (NULL);
            }
            if (ndata != numpoints)
            {
                Covise::sendError("ERROR: Size of data object 'DataIn' does not match mesh size");
                return (NULL);
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'DataIn' can't be accessed in shared memory");
            return (NULL);
        }
    }

    simplifier->setPercent(percent);
    simplifier->setWeights(weight0, weight1, weight2);
    simplifier->setShareCoordSet(true);
    simplifier->setSplitVertices(true);

    surf = new Surface(numpoints, numvertices, numprimitives);
    new_no_triangles = surf->gset->getPrimCount();

    starttime = clock();
    if (!no_data)
        coords_stat = ((csCoordSet3f *)surf->gset->getCoordSet())->point()->edit();

    surf->gset = simplifier->decimateGeoSet(surf->gset, &status);
    endtime = clock();

    red_tri = surf->gset->getPrimCount();
    red_points = ((csCoordSet3f *)(surf->gset->getCoordSet()))->point()->getCount();

    time = (endtime - starttime) / (double)CLOCKS_PER_SEC;
    sprintf(buf, "Removed %d triangles of %d, i.e. %.2f %% are left (%d)",
            new_no_triangles - red_tri, new_no_triangles, 100. * red_tri / new_no_triangles, red_tri);
    Covise::sendInfo(buf);
    sprintf(buf, "#points old: %d, #points new %d", numpoints, red_points);
    Covise::sendInfo(buf);
    sprintf(buf, "Time needed %.2f seconds", time);
    Covise::sendInfo(buf);

    DO_return = surf->createcoDistributedObjects(red_tri, red_points, Mesh_out_name, Data_out_name);
    delete surf;
    delete simplifier;
    return (DO_return);
}

Surface::~Surface()
{
    cerr << "Delete gset" << endl;

    delete[] vertex_list;
    delete gset;
}

Surface::Surface(int n_points, int n_vert, int n_poly)
{
    int i;
    csCoordSet3f *coords;
    csIndexSet *indices;

    num_points = n_points;
    num_vertices = n_vert;
    num_triangles = n_poly;
    num_poly = n_poly;

    if (strcmp(gtype, "POLYGN") == 0)
        ToTriangulationForPolygons();
    if (strcmp(gtype, "TRIANG") == 0)
        ToTriangulationForTriStrips();

    csTriSet *pset = new csTriSet;
    gset = pset;
    gset->setPrimCount(num_triangles);

    coords = new csCoordSet3f(num_points);
    coords->point()->setCount(num_points);
    for (i = 0; i < n_points; i++)
        coords->point()->set(i, csVec3f(x_in[i], y_in[i], z_in[i]));
    coords->point()->editDone();
    gset->setCoordSet(coords);

    indices = new csIndexSet(num_vertices);
    for (i = 0; i < num_vertices; i++)
        indices->index()->set(i, csInt(vertex_list[i]));

    indices->index()->editDone();
    //  gset->setCoordIndexSet(vertex_list);
    gset->setCoordIndexSet(indices);
}

coDistributedObject **Surface::createcoDistributedObjects(int red_tri, int red_points, char *Triangle_name, char *Data_name)
{
    int *vl, *pl, count, j;
    float *co_x, *co_y, *co_z;
    float *dt, *dt_x, *dt_y, *dt_z;

    coDoPolygons *polygons_out = NULL;
    coDoTriangleStrips *tristrips_out = NULL;
    coDoFloat *s_out = NULL;
    coDoVec3 *v_out = NULL;
    coDistributedObject **DO_return = new coDistributedObject *[2];

    if (red_points == 0)
        return (NULL);

    co_x = new float[red_points];
    co_y = new float[red_points];
    co_z = new float[red_points];
    pl = new int[red_tri];
    if (!no_data)
        if (strcmp(dtype, "USTSDT") == 0)
            dt = new float[red_points];
        else if (strcmp(dtype, "USTVDT") == 0)
        {
            dt_x = new float[red_points];
            dt_y = new float[red_points];
            dt_z = new float[red_points];
        }
        else
            Covise::sendError("dtype must be USTSDT or USTVDT");

    count = ((csCoordSet3f *)(gset->getCoordSet()))->point()->getCount();
    if (count != red_points)
    {
        Covise::sendError("ERROR: non-consistent number of points!");
        return (NULL);
    }

    count = ((csIndexSet *)(gset->getCoordIndexSet()))->index()->getCount();
    if (count != (red_tri * 3))
    {
        Covise::sendError("ERROR: non-consistent number of triangles!");
        return (NULL);
    }

    csVec3f *coords = ((csCoordSet3f *)gset->getCoordSet())->point()->edit();

    vl = (int *)gset->getCoordIndexSet()->index()->edit();

    for (j = 0; j < red_tri; j++)
        pl[j] = j * 3;

    for (j = 0; j < red_points; j++)
    {
        co_x[j] = coords[j][0];
        co_y[j] = coords[j][1];
        co_z[j] = coords[j][2];
    }

    if ((strcmp(gtype, "POLYGN") != 0) && (strcmp(gtype, "TRIANG") != 0))
    {
        Covise::sendError("gtype must be POLYGN or TRIANG");
        return (NULL);
    }

    if (strcmp(gtype, "POLYGN") == 0)
    {
        polygons_out = new coDoPolygons(Triangle_name, red_points, co_x, co_y, co_z,
                                        count, vl, red_tri, pl);
        if (!polygons_out->objectOk())
        {
            Covise::sendError("ERROR: creation of geometry object 'polygonsOut' failed");
            return (NULL);
        }
        else
        {
            polygons_out->addAttribute("vertexOrder", "2");
            polygons_out->addAttribute("COLOR", colorn);
        }
    }
    else if (strcmp(gtype, "TRIANG") == 0)
    {
        tristrips_out = new coDoTriangleStrips(Triangle_name, red_points, co_x, co_y, co_z,
                                               count, vl, red_tri, pl);

        if (!tristrips_out->objectOk())
        {
            Covise::sendError("ERROR: creation of geometry object 'tristripsOut' failed");
            return (NULL);
        }
        else
        {
            tristrips_out->addAttribute("vertexOrder", "2");
            tristrips_out->addAttribute("COLOR", colorn);
        }
    }

    if (!no_data)
    {
        if (strcmp(dtype, "USTSDT") == 0)
        {
            for (j = 0; j < red_points; j++)
                dt[j] = s_in[j];

            s_out = new coDoFloat(Data_name, red_points, dt);
            if (!s_out->objectOk())
            {
                Covise::sendError("ERROR: creation of geometry object 'dataOut' failed");
                return (NULL);
            }
        }
        else
        {
            for (j = 0; j < red_points; j++)
            {
                dt_x[j] = u_in[j];
                dt_y[j] = v_in[j];
                dt_z[j] = w_in[j];
            }
            v_out = new coDoVec3(Data_name, red_points, dt_x, dt_y, dt_z);
            if (!v_out->objectOk())
            {
                Covise::sendError("ERROR: creation of geometry object 'dataOut' failed");
                return (NULL);
            }
        }
    }

    if (polygons_out != NULL)
        DO_return[0] = polygons_out;
    if (tristrips_out != NULL)
        DO_return[0] = tristrips_out;
    if (!no_data)
        if (strcmp(dtype, "USTSDT") == 0)
            DO_return[1] = s_out;
        else
            DO_return[1] = v_out;

    if (!no_data)
        if (strcmp(dtype, "USTSDT") == 0)
            delete[] dt;
        else
        {
            delete[] dt_x;
            delete[] dt_y;
            delete[] dt_z;
        }

    delete[] pl;
    delete[] co_x;
    delete[] co_y;
    delete[] co_z;

    return (DO_return);
}

void Surface::ToTriangulationForTriStrips()
{
    int i, j;
    int vert;
    int count;
    int tri_count;

    num_triangles = num_poly;

    //  PrintInMesh() ;

    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = pl_in[i + 1] - pl_in[i]) > 3)
            num_triangles += vert - 3;
    }
    j = 0;
    while (pl_in[num_poly - 1] + j < num_vertices)
        j++;

    num_triangles += j - 3;

    count = 0;
    vertex_list = new int[num_triangles * 3];

    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = pl_in[i + 1] - pl_in[i]) == 3)
        {
            tri_count = 3 * count;
            vertex_list[tri_count] = vl_in[pl_in[i]];
            vertex_list[tri_count + 1] = vl_in[pl_in[i] + 1];
            vertex_list[tri_count + 2] = vl_in[pl_in[i] + 2];

            count++;
        }
        else
        {

            for (j = 0; j < vert - 2; j++)
            {
                tri_count = count * 3;
                vertex_list[tri_count] = vl_in[pl_in[i] + j];
                vertex_list[tri_count + 1] = vl_in[pl_in[i] + j + 1];
                vertex_list[tri_count + 2] = vl_in[pl_in[i] + j + 2];
                count++;
            }
        }
    }
    j = 0;

    while (pl_in[num_poly - 1] + j + 1 < num_vertices - 1)
    {
        //new triangle j, j+1, j+2
        tri_count = count * 3;
        vertex_list[tri_count] = vl_in[pl_in[num_poly - 1] + j];
        vertex_list[tri_count + 1] = vl_in[pl_in[num_poly - 1] + j + 1];
        vertex_list[tri_count + 2] = vl_in[pl_in[num_poly - 1] + j + 2];
        count++;
        j++;
    }

    num_vertices = 3 * num_triangles;
    //    PrintOutMesh() ;
    if (num_triangles != count)
        Covise::sendInfo("Triangle list non-consistent!!!");
}

void Surface::ToTriangulationForPolygons()
{
    int i, j;
    int vert;
    int count;
    int tri_count;

    num_triangles = num_poly;

    //    PrintInMesh() ;

    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = pl_in[i + 1] - pl_in[i]) > 3)
            num_triangles += vert - 3;
    }
    j = 0;
    while (pl_in[num_poly - 1] + j < num_vertices)
        j++;

    num_triangles += j - 3;

    count = 0;
    vertex_list = new int[num_triangles * 3];
    //  vertex_list = new csIndexSet(num_vertices);

    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = pl_in[i + 1] - pl_in[i]) == 3)
        {
            tri_count = 3 * count;
            vertex_list[tri_count] = vl_in[pl_in[i]];
            vertex_list[tri_count + 1] = vl_in[pl_in[i] + 1];
            vertex_list[tri_count + 2] = vl_in[pl_in[i] + 2];

            count++;
        }
        else
        {
            for (j = 1; j < vert - 1; j++)
            {
                tri_count = count * 3;
                vertex_list[tri_count] = vl_in[pl_in[i]];
                vertex_list[tri_count + 1] = vl_in[pl_in[i] + j];
                vertex_list[tri_count + 2] = vl_in[pl_in[i] + j + 1];
                count++;
            }
        }
    }
    j = 1;
    while (pl_in[num_poly - 1] + j + 1 < num_vertices)
    {
        //new triangle 0, j, j+1
        tri_count = count * 3;
        vertex_list[tri_count] = vl_in[pl_in[num_poly - 1]];
        vertex_list[tri_count + 1] = vl_in[pl_in[num_poly - 1] + j];
        vertex_list[tri_count + 2] = vl_in[pl_in[num_poly - 1] + j + 1];

        count++;
        j++;
    }

    num_vertices = 3 * num_triangles;
    //    PrintOutMesh() ;
    if (num_triangles != count)
        Covise::sendInfo("Triangle list non-consistent!!!");
}

void Surface::PrintInMesh()

{
    int i;

    cout << "num_poly: " << num_poly << " num_vertices: " << num_vertices << endl;
    cout << "Polygon List: \n" << endl;

    for (i = 0; i < num_poly; i++)
        cout << i << " " << pl_in[i] << "\n" << endl;

    for (i = 0; i < num_vertices; i++)
        cout << i << " " << vl_in[i] << "\n" << endl;
}

void Surface::PrintOutMesh()

{
    int i, j, count;

    cout << "num_triangles: " << num_triangles << endl;
    cout << "Triangle List: \n" << endl;

    for (i = 0; i < num_triangles; i++)
    {
        cout << i << ": ";
        count = i * 3;
        for (j = 0; j < 3; j++)
            cout << vertex_list[count + j] << " ";
        cout << "\n" << endl;
    }
}

void Surface::determine_neighbours(int *neighbours,
                                   int *num_of_neighbours, int *number_of_edges)
{ // makes the connectivity list stars[0..num_points-1]
    // containing for each point a list of all adjacent triangles,
    // the number of those (num_tri), and topology relevant information

    int i, j, k, v, jj, other_point, triangle, other_triangle, kk, count;
    int *sl;
    Star *stars;

    sl = new int[num_points];
    stars = new Star[num_points];

    // initialization: in sl will be number of triangles containing the vertex

    for (i = 0; i < num_points; i++)
        sl[i] = 0;

    for (i = 0; i < num_triangles; i++)
    {
        count = i * 3;
        sl[vertex_list[count]]++;
        sl[vertex_list[count + 1]]++;
        sl[vertex_list[count + 2]]++;
    }

    for (i = 0; i < num_points; i++)
    {
        stars[i].tri = new int[sl[i]];
        stars[i].num_tri = 0;
    }

    for (i = 0; i < num_triangles; i++)
    {
        count = i * 3;
        v = vertex_list[count];
        stars[v].tri[stars[v].num_tri] = i;
        stars[v].num_tri++;

        v = vertex_list[count + 1];
        stars[v].tri[stars[v].num_tri] = i;
        stars[v].num_tri++;

        v = vertex_list[count + 2];
        stars[v].tri[stars[v].num_tri] = i;
        stars[v].num_tri++;
    }

    // determine neighbours out of the starlist information

    for (i = 0; i < num_triangles; i++)
        num_of_neighbours[i] = 0;

    *number_of_edges = 0;

    for (i = 0; i < num_points; i++)
    {
        if (sl[i] <= 1)
            continue; // this point is contained in more that one triangle

        for (j = 0; j < stars[i].num_tri; j++)
        {

            triangle = stars[i].tri[j];
            cout << "Vertex: " << i << "Triangle: " << triangle << endl;

            for (k = 0; k < 3; k++)
            {
                other_point = vertex_list[triangle * 3 + k];
                cout << "other point " << other_point << endl;

                if (i < other_point && sl[other_point] > 1)
                {

                    for (jj = 0; jj < stars[other_point].num_tri; jj++)
                        if (stars[other_point].tri[jj] != triangle)
                        {
                            other_triangle = stars[other_point].tri[jj];
                            cout << "other triangle " << other_triangle << endl;

                            for (kk = 0; kk < 3; kk++)
                                if (vertex_list[other_triangle * 3 + kk] == i)
                                {
                                    neighbours[triangle * 3 + num_of_neighbours[triangle]] = other_triangle;
                                    cout << "setting of neighbours " << num_of_neighbours[triangle] << " " << triangle << " " << other_triangle << endl;
                                    num_of_neighbours[triangle]++;
                                    if (num_of_neighbours[triangle] > 3)
                                    {
                                        printf(" ERROR num_of_neighbours[triangle] > 3: %d %d \n",
                                               triangle, other_triangle);
                                        {
                                            int jjj;
                                            for (jjj = 0; jjj < num_of_neighbours[triangle]; jjj++)
                                            {
                                                printf(" neighbours of %d: %d \n", triangle, neighbours[triangle * 3 + jjj]);
                                                printf(" %d %d %d \n", vertex_list[neighbours[triangle * 3 + jjj] * 3],
                                                       vertex_list[neighbours[triangle * 3 + jjj] * 3 + 1],
                                                       vertex_list[neighbours[triangle * 3 + jjj] * 3 + 2]);
                                            }
                                            exit(1);
                                        }
                                    }
                                    if (triangle > other_triangle)
                                        (*number_of_edges)++;
                                } // if(vertex...
                        } // if(stars
                } // if( i < other_point ...
            } // k = 0
        } // j
    } // i

    delete[] sl;
    delete[] stars;
}
