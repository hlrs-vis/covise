/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoPolygons.h"
#include "coDoOctTreeP.h"
#include "covise_gridmethods.h"

/*
 $Log: covise_unstr.C,v $
Revision 1.1  1993/09/25  20:52:52  zrhk0125
Initial revision

*/

/***********************************************************************\ 
 **                                                                     **
 **   Structured classes Routines                   Version: 1.1        **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of structured grids       **
 **                  in a distributed manner.                           **
 **                                                                     **
 **   Classes      : coDoPoints, coDoLines, coDoPolygons **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  26.05.93  Ver 1.1 shm-access restructured,         **
 **                                    recursive data-objects (simple   **
 **                                    version),                        **
 **                                    some new types added             **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

using namespace covise;

coDistributedObject *coDoPolygons::virtualCtor(coShmArray *arr)
{
    return new coDoPolygons(coObjInfo(), arr);
}

int coDoPolygons::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 1)
    {
        (*il)[0].description = "Lines";
        return 1;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoPolygons::~coDoPolygons()
{
    delete[] lnl;
    delete[] lnli;
}

int
coDoPolygons::testACell(float *v_interp, const float *point,
                        int cell, int no_arrays, int array_dim, float tolerance,
                        const float *const *velo) const
{
    int num_of_vert;
    if (cell < numelem - 1)
    {
        num_of_vert = el[cell + 1] - el[cell];
    }
    else
    {
        num_of_vert = numconn - el[cell];
    }
    if (num_of_vert < 3)
    {
        return -1;
    }
    // test point for num_of_vert-2 triangles
    int base = cl[el[cell]];
    int triangle;
    float p0[3];
    float p1[3];
    float p2[3];
    p0[0] = x_c_[base];
    p0[1] = y_c_[base];
    p0[2] = z_c_[base];
    for (triangle = 1; triangle <= num_of_vert - 2; ++triangle)
    {
        int second = cl[el[cell] + triangle];
        int third = cl[el[cell] + triangle + 1];
        p1[0] = x_c_[second];
        p1[1] = y_c_[second];
        p1[2] = z_c_[second];
        p2[0] = x_c_[third];
        p2[1] = y_c_[third];
        p2[2] = z_c_[third];
        if (grid_methods::isin_triangle(point, p0, p1, p2, tolerance) == 1)
        {
            grid_methods::interpolateInTriangle(v_interp, point, no_arrays, array_dim,
                                                velo, base, second, third, p0, p1, p2);
            return 0;
        }
    }
    return -1;
}

void
coDoPolygons::MakeOctTree(const char *octtreeSurname) const
{
    if (!oct_tree)
    {
        char fullOctname[256];
        sprintf(fullOctname, "%s_OctTree_%s", name, octtreeSurname);
        oct_tree = new coDoOctTreeP(coObjInfo(fullOctname), numelem,
                                    numconn, numpoints,
                                    el, cl, x_c_, y_c_, z_c_);
    }
}

const coDoOctTreeP *
coDoPolygons::GetOctTree(const coDistributedObject *reuseOct,
                         const char *OctTreeSurname) const
{
    if (reuseOct)
    {
        oct_tree = reuseOct;
    }
    else if (OctTreeSurname)
    {
        MakeOctTree(OctTreeSurname);
    }

    if (oct_tree)
    {
        ((coDoOctTreeP *)oct_tree)->setInfo(numelem, numconn, numpoints, el, cl, x_c_, y_c_, z_c_);
    }
    return (coDoOctTreeP *)oct_tree;
}

float
coDoPolygons::Distance(int cell, const float *point) const
{
    float cp_point[3];
    cp_point[0] = point[0];
    cp_point[1] = point[1];
    cp_point[2] = point[2];
    Project(cp_point, cell);
    return sqrt((point[0] - cp_point[0]) * (point[0] - cp_point[0]) + (point[1] - cp_point[1]) * (point[1] - cp_point[1]) + (point[2] - cp_point[2]) * (point[2] - cp_point[2]));
}

void
coDoPolygons::Project(float *point, int cell) const
{
    int num_of_vert;
    if (cell < numelem - 1)
    {
        num_of_vert = el[cell + 1] - el[cell];
    }
    else
    {
        num_of_vert = numconn - el[cell];
    }
    // compute an average normal vector...
    float proj_point[3];
    grid_methods::ProjectPoint(proj_point, point, cl, el[cell], num_of_vert,
                               x_c_, y_c_, z_c_);
    point[0] = proj_point[0];
    point[1] = proj_point[1];
    point[2] = proj_point[2];
}

int
coDoPolygons::interpolateField(float *v_interp,
                               float *point,
                               int *cell,
                               int no_arrays,
                               int array_dim,
                               float tolerance,
                               const float *const *velo,
                               int search_level) const
{
    coDoOctTreeP *cast_oct_tree = (coDoOctTreeP *)oct_tree;
    if (*cell >= 0
        && *cell < numelem
        && testACell(v_interp, point, *cell, no_arrays, array_dim, tolerance,
                     velo) == 0)
    {
        Project(point, *cell);
        return 0;
    }

    // if this test fails or *cell<0
    // use the oct-tree to get a list of candidate cells
    // and test cells...
    const int *cell_list = cast_oct_tree->search(point);
    if (*cell_list > 0)
    {
        int i;

        int old_cell = *cell;
        float distance = FLT_MAX;
        int theCell = -1;
        if (search_level >= 0)
        {
            for (i = 0; i < *cell_list; ++i)
            {
                *cell = cell_list[i + 1];
                if (testACell(v_interp, point, *cell, no_arrays, array_dim, tolerance,
                              velo) == 0)
                {
                    float thisDistance = Distance(*cell, point);
                    // keep success in memory
                    if (distance > thisDistance)
                    {
                        distance = thisDistance;
                        theCell = *cell;
                    }
                }
            }
        }
        else // assume there is only a unique solution
        {
            for (i = 0; i < *cell_list; ++i)
            {
                *cell = cell_list[i + 1];
                if (testACell(v_interp, point, *cell, no_arrays, array_dim, tolerance,
                              velo) == 0)
                {
                    Project(point, *cell);
                    return 0;
                }
            }
        }

        if (theCell >= 0)
        {
            testACell(v_interp, point, theCell, no_arrays, array_dim, tolerance, velo);
            *cell = theCell;
            Project(point, *cell);
            return 0;
        }
        // else  not found
        *cell = old_cell;
    }
    // ...if the test fails for all cells,
    // search from the root of up to (2*search_level+1)**3 octtrees
    if (search_level < 0)
    {
        return -1;
    }

    int no_max_oct_trees = (2 * search_level + 1) * (2 * search_level + 1) * (2 * search_level + 1);
    std::vector<int> **oct_trees_cells = new std::vector<int> *[no_max_oct_trees];
    cast_oct_tree->LoadCellPopulations(oct_trees_cells, point, search_level);
    int old_cell = *cell;
    float distance = FLT_MAX;
    int theCell = -1;
    int octree_count;
    for (octree_count = 0; octree_count < no_max_oct_trees; ++octree_count)
    {
        if (oct_trees_cells[octree_count] == NULL)
        {
            continue;
        }
        int i;
        for (i = 0; i < oct_trees_cells[octree_count]->size(); ++i)
        {
            *cell = oct_trees_cells[octree_count]->operator[](i);
            if (testACell(v_interp, point, *cell, no_arrays, array_dim, tolerance,
                          velo) == 0)
            {
                float thisDistance = Distance(*cell, point);
                // keep success in memory
                if (distance > thisDistance)
                {
                    distance = thisDistance;
                    theCell = *cell;
                }
            }
        }
    }
    delete[] oct_trees_cells;

    if (theCell >= 0)
    {
        testACell(v_interp, point, theCell, no_arrays, array_dim, tolerance, velo);
        *cell = theCell;
        Project(point, *cell);
        return 0;
    }
    *cell = old_cell;
    return -1;
}

coDoPolygons::coDoPolygons(const coObjInfo &info, coShmArray *arr)
    : coDoGrid(info, "POLYGN")
    , lnl(NULL)
    , lnli(NULL)
    , el(NULL)
    , cl(NULL)
    , x_c_(NULL)
    , y_c_(NULL)
    , z_c_(NULL)
    , oct_tree(NULL)
{
    lines = new coDoLines(coObjInfo());
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
    getAddresses(&x_c_, &y_c_, &z_c_, &cl, &el);
    numelem = getNumPolygons();
    numconn = getNumVertices();
    numpoints = getNumPoints();
}

coDoPolygons::coDoPolygons(const coObjInfo &info, int no_p, int no_v, int no_pol)
    : coDoGrid(info, "POLYGN")
    , lnl(NULL)
    , lnli(NULL)
    , el(NULL)
    , cl(NULL)
    , x_c_(NULL)
    , y_c_(NULL)
    , z_c_(NULL)
    , oct_tree(NULL)
{
    char *l_name;
    covise_data_list dl[1];

    l_name = new char[strlen(info.getName()) + 3];
    strcpy(l_name, info.getName());
    strcat(l_name, "_L");
    lines = new coDoLines(coObjInfo(l_name), no_p, no_v, no_pol);
    delete[] l_name;
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)lines;
    new_ok = store_shared_dl(1, dl) != 0;
    if (!new_ok)
        return;

    getAddresses(&x_c_, &y_c_, &z_c_, &cl, &el);
    numelem = getNumPolygons();
    numconn = getNumVertices();
    numpoints = getNumPoints();
}

coDoPolygons::coDoPolygons(const coObjInfo &info, int no_p,
                           float *x_c, float *y_c, float *z_c, int no_v, int *v_l,
                           int no_pol, int *pol_l)
    : coDoGrid(info, "POLYGN")
    , lnl(NULL)
    , lnli(NULL)
    , el(NULL)
    , cl(NULL)
    , x_c_(NULL)
    , y_c_(NULL)
    , z_c_(NULL)
    , oct_tree(NULL)
{
    char *l_name;
    covise_data_list dl[1];

    l_name = new char[strlen(info.getName()) + 3];
    strcpy(l_name, info.getName());
    strcat(l_name, "_L");
    lines = new coDoLines(coObjInfo(l_name), no_p, x_c, y_c, z_c,
                          no_v, v_l, no_pol, pol_l);
    delete[] l_name;
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)lines;
    new_ok = store_shared_dl(1, dl) != 0;
    if (!new_ok)
        return;

    getAddresses(&x_c_, &y_c_, &z_c_, &cl, &el);
    numelem = getNumPolygons();
    numconn = getNumVertices();
    numpoints = getNumPoints();
}

coDoPolygons *coDoPolygons::cloneObject(const coObjInfo &newinfo) const
{
    float *c[3];
    int *v_l, *p_l;
    getAddresses(&c[0], &c[1], &c[2], &v_l, &p_l);
    return new coDoPolygons(newinfo, getNumPoints(), c[0], c[1], c[2], getNumVertices(), v_l,
                            getNumPolygons(), p_l);
}

int coDoPolygons::rebuildFromShm()
{
    covise_data_list dl[1];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)lines;
    return restore_shared_dl(1, dl);
}

void coDoPolygons::computeNeighborList() const
{
    int ja, j, i, offset, *tmpl1, numcoord;
    float *d1, *d2, *d3;
    lines->getAddresses(&d1, &d2, &d3, &cl, &el);
    numcoord = lines->getNumPoints();
    numelem = lines->getNumLines();
    numconn = lines->getNumVertices();
    lnli = new int[(int)numcoord + 1];
    tmpl1 = new int[(int)numcoord];
    memset(lnli, 0, (int)numcoord * sizeof(int));
    for (i = 0; i < (int)numconn; i++)
    {
        lnli[cl[i]]++;
    }

    //create the position list for cuc list -> cuc_pos
    j = ja = 0;
    for (i = 0; i < (int)numcoord; i++)
    {
        j += lnli[i];
        lnli[i] = ja;
        tmpl1[i] = ja;
        ja = j;
    }
    lnli[i] = ja;
    lnl = new int[(int)numconn]; //size for array

    //fill the cuc list
    for (i = 0; i < (int)numelem; i++)
    {
        if (i == numelem - 1)
            offset = numconn - el[i];
        else
            offset = el[i + 1] - el[i];
        for (j = 0; j < offset; j++)
        {
            lnl[tmpl1[cl[el[i] + j]]] = i;
            tmpl1[cl[el[i] + j]]++;
        }
    }
    delete[] tmpl1;
}

int coDoPolygons::getNeighbor(int element, int n1, int n2)
{
    int f2, ce = -1;
    int i, n, numpt;

    f2 = 0;
    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            f2 = 0;
            if (ce == numelem - 1)
                numpt = numconn - el[ce];
            else
                numpt = el[ce + 1] - el[ce];
            for (n = 0; n < numpt; n++)
            {
                if (cl[el[ce] + n] == n2)
                    f2 = 1;
            }
            if (f2)
            {
                break;
            }
        }
    }
    if (f2)
        return (ce);
    else
        return (-1);
}

int coDoPolygons::getNeighbors(int element, int n1, int *neighbors)
{
    int ce = -1;
    int i, num = -1;

    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            num++;
            neighbors[i] = ce;
        }
    }
    return (num);
}

coDistributedObject *coDoTriangles::virtualCtor(coShmArray *arr)
{
    return new coDoTriangles(coObjInfo(), arr);
}

int coDoTriangles::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 5)
    {
        (*il)[0].description = "Points";
        (*il)[1].description = "Number of Vertices";
        (*il)[2].description = "Vertex List";
        return 5;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

int coDoTriangles::setNumVertices(int numElem)
{
    if (numElem > no_of_vertices)
        return -1;

    no_of_vertices = numElem;
    return 0;
}

coDoTriangles::coDoTriangles(const coObjInfo &info, coShmArray *arr)
    : coDoGrid(info, "TRITRI")
{
    points = new coDoPoints(coObjInfo());
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoTriangles::coDoTriangles(const coObjInfo &info,
                             int no_p, int no_v)
    : coDoGrid(info, "TRITRI")
{
    char *p_name;
    covise_data_list dl[3];

    p_name = new char[strlen(info.getName()) + 3];
    strcpy(p_name, info.getName());
    strcat(p_name, "_P");
    points = new coDoPoints(coObjInfo(p_name), no_p);
    delete[] p_name;
    vertex_list.set_length(no_v);
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_vertices = no_v;
}

coDoTriangles::coDoTriangles(const coObjInfo &info, int no_p,
                             float *x_c, float *y_c, float *z_c, int no_v, int *v_l)
    : coDoGrid(info, "TRITRI")
{
    char *p_name;
    int i;
    covise_data_list dl[3];

    p_name = new char[strlen(info.getName()) + 3];
    strcpy(p_name, info.getName());
    strcat(p_name, "_P");
    points = new coDoPoints(coObjInfo(p_name), no_p, x_c, y_c, z_c);
    delete[] p_name;
    vertex_list.set_length(no_v);
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_vertices = no_v;

    int *tmpv;
    float *tmpf;
    getAddresses(&tmpf, &tmpf, &tmpf, &tmpv);
    i = no_v * sizeof(int);
    memcpy(tmpv, v_l, i);
}

coDoTriangles *coDoTriangles::cloneObject(const coObjInfo &newinfo) const
{
    float *c[3];
    int *v_l;
    getAddresses(&c[0], &c[1], &c[2], &v_l);
    return new coDoTriangles(newinfo, getNumPoints(), c[0], c[1], c[2], getNumVertices(), v_l);
}

int coDoTriangles::rebuildFromShm()
{
    covise_data_list dl[3];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    return restore_shared_dl(3, dl);
}

coDistributedObject *coDoQuads::virtualCtor(coShmArray *arr)
{
    return new coDoQuads(coObjInfo(), arr);
}

int coDoQuads::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 5)
    {
        (*il)[0].description = "Points";
        (*il)[1].description = "Number of Vertices";
        (*il)[2].description = "Vertex List";
        return 5;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

int coDoQuads::setNumVertices(int numElem)
{
    if (numElem > no_of_vertices)
        return -1;

    no_of_vertices = numElem;
    return 0;
}

coDoQuads::coDoQuads(const coObjInfo &info, coShmArray *arr)
    : coDoGrid(info, "QUADS")
{
    points = new coDoPoints(coObjInfo());
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoQuads::coDoQuads(const coObjInfo &info,
                     int no_p, int no_v)
    : coDoGrid(info, "QUADS")
{
    char *p_name;
    covise_data_list dl[3];

    p_name = new char[strlen(info.getName()) + 3];
    strcpy(p_name, info.getName());
    strcat(p_name, "_P");
    points = new coDoPoints(coObjInfo(p_name), no_p);
    delete[] p_name;
    vertex_list.set_length(no_v);
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_vertices = no_v;
}

coDoQuads::coDoQuads(const coObjInfo &info, int no_p,
                     float *x_c, float *y_c, float *z_c, int no_v, int *v_l)
    : coDoGrid(info, "QUADS")
{
    char *p_name;
    int i;
    covise_data_list dl[3];

    p_name = new char[strlen(info.getName()) + 3];
    strcpy(p_name, info.getName());
    strcat(p_name, "_P");
    points = new coDoPoints(coObjInfo(p_name), no_p, x_c, y_c, z_c);
    delete[] p_name;
    vertex_list.set_length(no_v);
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_vertices = no_v;

    int *tmpv;
    float *tmpf;
    getAddresses(&tmpf, &tmpf, &tmpf, &tmpv);
    i = no_v * sizeof(int);
    memcpy(tmpv, v_l, i);
}

coDoQuads *coDoQuads::cloneObject(const coObjInfo &newinfo) const
{
    float *c[3];
    int *v_l;
    getAddresses(&c[0], &c[1], &c[2], &v_l);
    return new coDoQuads(newinfo, getNumPoints(), c[0], c[1], c[2], getNumVertices(), v_l);
}

int coDoQuads::rebuildFromShm()
{
    covise_data_list dl[3];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    return restore_shared_dl(3, dl);
}
