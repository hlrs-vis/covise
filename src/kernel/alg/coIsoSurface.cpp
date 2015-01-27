/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coIsoSurface.h"
#include <config/CoviseConfig.h>
#include "IsoCuttingTables.h"
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <api/coOutputPort.h>
#include <api/coModule.h>

using namespace covise;

#define ADDVERTEX                \
    if (n1 < n2)                 \
    {                            \
        if (!add_vertex(n1, n2)) \
            return false;        \
    }                            \
    else                         \
    {                            \
        if (!add_vertex(n2, n1)) \
            return false;        \
    }

#define ADDVERTEXX01                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX02                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 2)], jj + y_add[*(polygon_nodes + 2)], kk + z_add[*(polygon_nodes + 2)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 2)], jj + y_add[*(polygon_nodes + 2)], kk + z_add[*(polygon_nodes + 2)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX03                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 3)], jj + y_add[*(polygon_nodes + 3)], kk + z_add[*(polygon_nodes + 3)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 3)], jj + y_add[*(polygon_nodes + 3)], kk + z_add[*(polygon_nodes + 3)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX04                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 4)], jj + y_add[*(polygon_nodes + 4)], kk + z_add[*(polygon_nodes + 4)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 4)], jj + y_add[*(polygon_nodes + 4)], kk + z_add[*(polygon_nodes + 4)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX(a1, a2)                                                                                                                                                                                                                    \
    if (n1 < n2)                                                                                                                                                                                                                              \
        add_vertex(n1, n2, ii + x_add[*(polygon_nodes + a1)], jj + y_add[*(polygon_nodes + a1)], kk + z_add[*(polygon_nodes + a1)], ii + x_add[*(polygon_nodes + a2)], jj + y_add[*(polygon_nodes + a2)], kk + z_add[*(polygon_nodes + a2)]); \
    else                                                                                                                                                                                                                                      \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + a2)], jj + y_add[*(polygon_nodes + a2)], kk + z_add[*(polygon_nodes + a2)], ii + x_add[*(polygon_nodes + a1)], jj + y_add[*(polygon_nodes + a1)], kk + z_add[*(polygon_nodes + a1)]);
#define ADDVERTEXXX(a1, a2)                                                                                                 \
    if (n1 < n2)                                                                                                            \
        add_vertex(n1, n2, ii + x_add[a1], jj + y_add[a1], kk + z_add[a1], ii + x_add[a2], jj + y_add[a2], kk + z_add[a2]); \
    else                                                                                                                    \
        add_vertex(n2, n1, ii + x_add[a2], jj + y_add[a2], kk + z_add[a2], ii + x_add[a1], jj + y_add[a1], kk + z_add[a1]);

// lazy eval: set from covise.config upon 1st usage. default=17
int IsoPlane::maxTriPerVertex = -1;

namespace covise
{
// commodity function: read an it with a default value.
static int readConfig(const char *varName, int defaultValue)
{
    return coCoviseConfig::getInt(varName, defaultValue);
}
}

IsoPlane::IsoPlane()
    : vertice_list(NULL)
    , coords_x(NULL)
    , coords_y(NULL)
    , coords_z(NULL)
    , V_Data_U(NULL)
    , V_Data_V(NULL)
    , V_Data_W(NULL)
    , S_Data(NULL)
    , node_table(NULL)
{
    if (maxTriPerVertex < 0)
        maxTriPerVertex = readConfig("Module.IsoSurface.MaxTrianglesPerVertex", 17);
}

IsoPlane::IsoPlane(int n_elem, int n_nodes, int Type, float cutVertexRatio,
                   const int *ell, const int *cll, const int *tll,
                   const float *xin, const float *yin, const float *zin,
                   const float *sin, const float *iin,
                   const float *uin, const float *vin, const float *win, float isovalue,
                   bool isConnected, char *ib)
    :

    el(ell)
    , cl(cll)
    , tl(tll)
    , x_in(xin)
    , y_in(yin)
    , z_in(zin)
    , s_in(sin)
    , i_in(iin)
    , u_in(uin)
    , v_in(vin)
    , w_in(win)
    , vertice_list(NULL)
    , coords_x(NULL)
    , coords_y(NULL)
    , coords_z(NULL)
    , V_Data_U(NULL)
    , V_Data_V(NULL)
    , V_Data_W(NULL)
    , S_Data(NULL)
    , node_table(NULL)
    , _isovalue(isovalue)
    , _isConnected(isConnected)
{
    iblank = ib;
    if (maxTriPerVertex < 0)
        maxTriPerVertex = readConfig("Module.IsoSurface.MaxTrianglesPerVertex", 17);

    NodeInfo *node;
    int i;
    Datatype = Type;
    num_nodes = n_nodes;
    num_elem = n_elem;
    //node_table   = (NodeInfo *)malloc(n_nodes*sizeof(NodeInfo));
    node_table = new NodeInfo[n_nodes];
    node = node_table;
    for (i = 0; i < n_nodes; i++)
    {
        node->targets[0] = 0;
        // Calculate the distance of each node
        // to the Isovalue
        node->dist = (i_in[i] - isovalue);
        node->side = (node->dist >= 0 ? 1 : 0);
        node++;
    }
    num_triangles = num_vertices = num_coords = 0;

    /// Calculate somewhat reasonable size for fields in USGs
    // leave on old values for STR so far
    if (cutVertexRatio > 0)
        max_coords = (int)(n_nodes * 0.01 * cutVertexRatio);
    else
        max_coords = 3 * n_nodes;

    // leave on old values for STR so far
    if (cutVertexRatio > 0)
        vertice_list = new int[max_coords * 6];
    else
        vertice_list = new int[n_elem * (size_t)12];

    vertex = vertice_list;
    coords_x = new float[max_coords];
    coords_y = new float[max_coords];
    coords_z = new float[max_coords];
    coord_x = coords_x;
    coord_y = coords_y;
    coord_z = coords_z;
    S_Data = V_Data_U = NULL;
    if (Datatype) // (Scalar Data)
    {
        S_Data_p = S_Data = new float[max_coords];
    }
    else
    {
        V_Data_U_p = V_Data_U = new float[max_coords];
        V_Data_V_p = V_Data_V = new float[max_coords];
        V_Data_W_p = V_Data_W = new float[max_coords];
    }
}

IsoPlane::IsoPlane(int n_elem, int n_nodes, int Type, /*float cutVertexRatio,*/
                   const int *ell, const int *cll, const int *tll,
                   const float *xin, const float *yin, const float *zin,
                   const float *sin, const float *iin,
                   const float *uin, const float *vin, const float *win, float isovalue,
                   bool isConnected, char *ib)
    :

    el(ell)
    , cl(cll)
    , tl(tll)
    , x_in(xin)
    , y_in(yin)
    , z_in(zin)
    , s_in(sin)
    , i_in(iin)
    , u_in(uin)
    , v_in(vin)
    , w_in(win)
    , vertice_list(NULL)
    , coords_x(NULL)
    , coords_y(NULL)
    , coords_z(NULL)
    , V_Data_U(NULL)
    , V_Data_V(NULL)
    , V_Data_W(NULL)
    , S_Data(NULL)
    , node_table(NULL)
    , _isovalue(isovalue)
    , _isConnected(isConnected)
{
    iblank = ib;

    if (x_in == NULL)
    {
        return;
    }

    Datatype = Type;
    num_nodes = n_nodes;
    num_elem = n_elem;
}

IsoPlane::~IsoPlane()
{
    delete[] node_table;
    node_table = NULL;
    delete[] coords_x;
    delete[] coords_y;
    delete[] coords_z;
    delete[] vertice_list;
    if (S_Data)
    {
        delete[] S_Data;
    }
    if (V_Data_U)
    {
        delete[] V_Data_U;
        delete[] V_Data_V;
        delete[] V_Data_W;
    }
}

POLYHEDRON_IsoPlane::POLYHEDRON_IsoPlane(int n_elem, int n_conn, int n_nodes, int Type,
                                         const int *el, const int *cl, const int *tl,
                                         const float *x_in, const float *y_in, const float *z_in,
                                         const float *s_in, const float *i_in,
                                         const float *u_in, const float *v_in, const float *w_in, float isovalue,
                                         bool isConnected, char *ib)
    : IsoPlane(n_elem, n_nodes, Type, el, cl, tl, x_in, y_in, z_in, s_in, i_in, u_in, v_in, w_in, isovalue, isConnected, ib)
{
    num_conn = n_conn;
    elem_out = NULL;
    conn_out = NULL;
    x_coord_out = NULL;
    y_coord_out = NULL;
    z_coord_out = NULL;
    sdata_out = NULL;
    udata_out = NULL;
    vdata_out = NULL;
    wdata_out = NULL;
}

POLYHEDRON_IsoPlane::~POLYHEDRON_IsoPlane()
{
    if (elem_out)
        delete[] elem_out;
    if (conn_out)
        delete[] conn_out;
    if (x_coord_out)
        delete[] x_coord_out;
    if (y_coord_out)
        delete[] y_coord_out;
    if (z_coord_out)
        delete[] z_coord_out;
    if (sdata_out)
        delete[] sdata_out;
    if (udata_out)
        delete[] udata_out;
    if (vdata_out)
        delete[] vdata_out;
    if (wdata_out)
        delete[] wdata_out;
}

UNI_IsoPlane::~UNI_IsoPlane()
{
    delete[] x_in;
    delete[] y_in;
    delete[] z_in;
}

UNI_IsoPlane::UNI_IsoPlane(int n_elem, int n_nodes, int Type,
                           float x_min, float x_max, float y_min,
                           float y_max, float z_min, float z_max,
                           int xsiz, int ysiz, int zsiz,
                           const float *sin, const float *iin,
                           const float *uin, const float *vin, const float *win, float isovalue,
                           bool isConnected, char *ib)
    : IsoPlane(n_elem, n_nodes, Type, -1, NULL, NULL, NULL, NULL, NULL, NULL,
               sin, iin, uin, vin, win, isovalue, isConnected, ib)
    , x_size(xsiz)
    , y_size(ysiz)
    , z_size(zsiz)
{
    int i;
    float xdisc, ydisc, zdisc;
    x_in = new float[x_size];
    y_in = new float[y_size];
    z_in = new float[z_size];
    xdisc = (x_max - x_min) / (x_size - 1);
    ydisc = (y_max - y_min) / (y_size - 1);
    zdisc = (z_max - z_min) / (z_size - 1);
    float *xin = const_cast<float *>(x_in);
    float *yin = const_cast<float *>(y_in);
    float *zin = const_cast<float *>(z_in);
    for (i = 0; i < x_size; i++)
        xin[i] = x_min + xdisc * i;
    for (i = 0; i < y_size; i++)
        yin[i] = y_min + ydisc * i;
    for (i = 0; i < z_size; i++)
        zin[i] = z_min + zdisc * i;
}

RECT_IsoPlane::RECT_IsoPlane(int n_elem, int n_nodes, int Type,
                             int xsiz, int ysiz, int zsiz,
                             const float *xin, const float *yin, const float *zin,
                             const float *sin, const float *iin,
                             const float *uin, const float *vin, const float *win, float isovalue,
                             bool isConnected, char *ib)
    : IsoPlane(n_elem, n_nodes, Type, -1, NULL, NULL, NULL, xin, yin, zin,
               sin, iin, uin, vin, win, isovalue,
               isConnected, ib)
    , x_size(xsiz)
    , y_size(ysiz)
    , z_size(zsiz)
{
}

bool IsoPlane::createIsoPlane()
{
    bool standard_cells_found;
    int element;
    int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int i;
    const int *node_list;
    const int *node;
    int elementtype;
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    cutting_info *C_Info;
// just for testing
#ifdef DEBUG
    int cases[256];
    for (i = 0; i < 256; i++)
        cases[i] = 0;
#endif
    standard_cells_found = false;
    polyhedral_cells_found = false;

    for (element = 0; element < num_elem; element++)
    {
        if (iblank == NULL || iblank[element] != '\0')
        {
            elementtype = tl[element];
            bitmap = 0;
            i = UnstructuredGrid_Num_Nodes[elementtype];
            // Avoid polyhedral cells if polyhedral cell support is not activated
            if (i != -1)
            {
                standard_cells_found = true;
                // number of nodes for current element
                node_list = cl + el[element];
                // pointer to nodes of current element
                node = node_list + i;
                // node = pointer to last node of current element

                while (i--)
                    bitmap |= node_table[*--node].side << i;
                // bitmap is now an index to the Cuttingtable
                C_Info = Cutting_Info[elementtype] + bitmap;
// just for testing
#ifdef DEBUG
                cases[bitmap]++;
#endif
                numIntersections = C_Info->nvert;
                if (numIntersections)
                {
                    polygon_nodes = C_Info->node_pairs;
                    switch (numIntersections)
                    {
                    case 1:
                        num_triangles++;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 2:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 3:
                        /*
                      *      Something of a special case here:  If the average of the vertices
                      *      is greater than the isovalue, we create two separated polygons
                      *      at the vertices.  If it is less, then we make a little valley
                      *      shape.
                      */
                        no1 = node_list[*polygon_nodes++];
                        no2 = node_list[*polygon_nodes++];
                        no3 = node_list[*polygon_nodes++];
                        no4 = node_list[*polygon_nodes++];
                        no5 = node_list[*polygon_nodes++];
                        no6 = node_list[*polygon_nodes++];
                        if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) < 0)
                        {
                            num_triangles += 2;
                            n1 = no1;
                            n2 = no2;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n2 = no4;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no4;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n2 = no6;
                            ADDVERTEX;
                        }
                        else
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEX;
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no2;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEX;
                        }
                        break;
                    case 4:
                        /*
                      *      Something of a special case here:  If the average of the vertices
                      *      is smaller than the isovalue, we create two separated polygons
                      *      at the vertices.  If it is less, then we make a little valley
                      *      shape.
                      */
                        no1 = node_list[*polygon_nodes++];
                        no2 = node_list[*polygon_nodes++];
                        no3 = node_list[*polygon_nodes++];
                        no4 = node_list[*polygon_nodes++];
                        no5 = node_list[*polygon_nodes++];
                        no6 = node_list[*polygon_nodes++];
                        if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) > 0)
                        {
                            num_triangles += 2;
                            n1 = no1;
                            n2 = no2;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n2 = no4;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no4;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n2 = no6;
                            ADDVERTEX;
                        }
                        else
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEX;
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no2;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEX;
                        }
                        break;
                    case 5:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 6:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 7:
                        num_triangles += 2;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 8:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 9:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 10:
                        num_triangles += 3;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 11:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 12:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 13:
                        num_triangles += 4;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 14:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 15:
                        num_triangles += 4;
                        if (*polygon_nodes)
                        {
                            n1 = node_list[1];
                            n2 = node_list[0];
                            ADDVERTEX;
                            n2 = node_list[5];
                            ADDVERTEX;
                            n2 = node_list[2];
                            ADDVERTEX;
                            n1 = node_list[4];
                            n2 = node_list[5];
                            ADDVERTEX;
                            n2 = node_list[0];
                            ADDVERTEX;
                            n2 = node_list[7];
                            ADDVERTEX;
                            n1 = node_list[6];
                            n2 = node_list[2];
                            ADDVERTEX;
                            n2 = node_list[5];
                            ADDVERTEX;
                            n2 = node_list[7];
                            ADDVERTEX;
                            n1 = node_list[3];
                            n2 = node_list[0];
                            ADDVERTEX;
                            n2 = node_list[2];
                            ADDVERTEX;
                            n2 = node_list[7];
                            ADDVERTEX;
                        }
                        else
                        {
                            n1 = node_list[0];
                            n2 = node_list[1];
                            ADDVERTEX;
                            n2 = node_list[3];
                            ADDVERTEX;
                            n2 = node_list[4];
                            ADDVERTEX;
                            n1 = node_list[5];
                            n2 = node_list[1];
                            ADDVERTEX;
                            n2 = node_list[4];
                            ADDVERTEX;
                            n2 = node_list[6];
                            ADDVERTEX;
                            n1 = node_list[2];
                            n2 = node_list[1];
                            ADDVERTEX;
                            n2 = node_list[6];
                            ADDVERTEX;
                            n2 = node_list[3];
                            ADDVERTEX;
                            n1 = node_list[7];
                            n2 = node_list[4];
                            ADDVERTEX;
                            n2 = node_list[3];
                            ADDVERTEX;
                            n2 = node_list[6];
                            ADDVERTEX;
                        }
                        break;
                    }
                }
            }

            else
            {
                polyhedral_cells_found = true;
            }
        }
    }
// just for testing
#ifdef DEBUG
    fprintf(stderr, " Dreiecke: %d\n", num_triangles);
    for (i = 0; i < 256; i++)
        fprintf(stderr, " %d : %d\n", i, cases[i]);
#endif
    // No standard cells found in the dataset
    if (standard_cells_found == false)
    {
        return false;
    }
    return true;
}

void UNI_IsoPlane::createIsoPlane()
{
    int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int node_list[8];
    int x_add[] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    int y_add[] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    int z_add[] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk;
    int no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    cutting_info *C_Info;

    for (ii = 0; ii < x_size - 1; ii++)
    {
        for (jj = 0; jj < y_size - 1; jj++)
        {
            for (kk = 0; kk < z_size - 1; kk++)
            {
                if (iblank == NULL || iblank[*n_1] != '\0')
                {
                    bitmap = node_table[*n_1].side | node_table[*n_2].side << 1
                             | node_table[*n_3].side << 2 | node_table[*n_4].side << 3
                             | node_table[*n_5].side << 4 | node_table[*n_6].side << 5
                             | node_table[*n_7].side << 6 | node_table[*n_8].side << 7;

                    // bitmap is now an index to the Cuttingtable
                    C_Info = Cutting_Info[TYPE_HEXAGON] + bitmap;
                    numIntersections = C_Info->nvert;
                    if (numIntersections)
                    {
                        polygon_nodes = C_Info->node_pairs;
                        switch (numIntersections)
                        {
                        case 1:
                            num_triangles++;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 2:
                            num_triangles += 2;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 3:
                            /*
                         *      Something of a special case here:  If the average of the vertices
                         *      is greater than the isovalue, we create two separated polygons
                         *      at the vertices.  If it is less, then we make a little valley
                         *      shape.
                         */
                            no1 = node_list[*(polygon_nodes)];
                            no2 = node_list[*(polygon_nodes + 1)];
                            no3 = node_list[*(polygon_nodes + 2)];
                            no4 = node_list[*(polygon_nodes + 3)];
                            no5 = node_list[*(polygon_nodes + 4)];
                            no6 = node_list[*(polygon_nodes + 5)];
                            if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) < 0)
                            {
                                num_triangles += 2;
                                n1 = no1;
                                n2 = no2;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n2 = no4;
                                ADDVERTEXX03;
                                n1 = no5;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                                n2 = no3;
                                ADDVERTEXX(4, 2);
                                n2 = no6;
                                ADDVERTEXX(4, 5);
                            }
                            else
                            {
                                num_triangles += 4;
                                n1 = no1;
                                n2 = no2;
                                vertex1 = vertex;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n1 = no5;
                                n2 = no3;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 2);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no6;
                                ADDVERTEXX(4, 6);
                                n1 = no1;
                                n2 = no4;
                                vertex1 = vertex;
                                ADDVERTEXX04;
                                n2 = no2;
                                ADDVERTEXX01;
                                n1 = no5;
                                n2 = no6;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 5);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                            }
                            break;
                        case 4:
                            /*
                         *      Something of a special case here:  If the average of the vertices
                         *      is smaller than the isovalue, we create two separated polygons
                         *      at the vertices.  If it is less, then we make a little valley
                         *      shape.
                         */
                            no1 = node_list[*(polygon_nodes)];
                            no2 = node_list[*(polygon_nodes + 1)];
                            no3 = node_list[*(polygon_nodes + 2)];
                            no4 = node_list[*(polygon_nodes + 3)];
                            no5 = node_list[*(polygon_nodes + 4)];
                            no6 = node_list[*(polygon_nodes + 5)];
                            if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) > 0)
                            {
                                num_triangles += 2;
                                n1 = no1;
                                n2 = no2;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n2 = no4;
                                ADDVERTEXX03;
                                n1 = no5;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                                n2 = no3;
                                ADDVERTEXX(4, 2);
                                n2 = no6;
                                ADDVERTEXX(4, 5);
                            }
                            else
                            {
                                num_triangles += 4;
                                n1 = no1;
                                n2 = no2;
                                vertex1 = vertex;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n1 = no5;
                                n2 = no3;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 2);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no6;
                                ADDVERTEXX(4, 5);
                                n1 = no1;
                                n2 = no4;
                                vertex1 = vertex;
                                ADDVERTEXX03;
                                n2 = no2;
                                ADDVERTEXX01;
                                n1 = no5;
                                n2 = no6;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 5);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                            }
                            break;
                        case 5:
                            num_triangles += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 6:
                            num_triangles += 2;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            break;
                        case 7:
                            num_triangles += 2;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 8:
                            num_triangles += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 9:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX(3, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 10:
                            num_triangles += 3;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 11:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            polygon_nodes += 3;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 12:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX(2, 1);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX(2, 3);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*(polygon_nodes + 5)];
                            n2 = node_list[*(polygon_nodes + 4)];
                            ADDVERTEXX(5, 4);
                            break;
                        case 13:
                            num_triangles += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            vertex2 = vertex;
                            n1 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX(3, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 4)];
                            ADDVERTEXX(3, 4);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*(polygon_nodes + 5)];
                            n2 = node_list[*(polygon_nodes + 6)];
                            ADDVERTEXX(5, 6);
                            break;
                        case 14:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 15:
                            num_triangles += 4;
                            if (*polygon_nodes)
                            {
                                n1 = node_list[1];
                                n2 = node_list[0];
                                ADDVERTEXXX(1, 0);
                                n2 = node_list[5];
                                ADDVERTEXXX(1, 5);
                                n2 = node_list[2];
                                ADDVERTEXXX(1, 2);
                                n1 = node_list[4];
                                n2 = node_list[5];
                                ADDVERTEXXX(4, 5);
                                n2 = node_list[0];
                                ADDVERTEXXX(4, 0);
                                n2 = node_list[7];
                                ADDVERTEXXX(4, 7);
                                n1 = node_list[6];
                                n2 = node_list[2];
                                ADDVERTEXXX(6, 2);
                                n2 = node_list[5];
                                ADDVERTEXXX(6, 5);
                                n2 = node_list[7];
                                ADDVERTEXXX(6, 7);
                                n1 = node_list[3];
                                n2 = node_list[0];
                                ADDVERTEXXX(3, 0);
                                n2 = node_list[2];
                                ADDVERTEXXX(3, 2);
                                n2 = node_list[7];
                                ADDVERTEXXX(3, 7);
                            }
                            else
                            {
                                n1 = node_list[0];
                                n2 = node_list[1];
                                ADDVERTEXXX(0, 1);
                                n2 = node_list[3];
                                ADDVERTEXXX(0, 3);
                                n2 = node_list[4];
                                ADDVERTEXXX(0, 4);
                                n1 = node_list[5];
                                n2 = node_list[1];
                                ADDVERTEXXX(5, 1);
                                n2 = node_list[4];
                                ADDVERTEXXX(5, 4);
                                n2 = node_list[6];
                                ADDVERTEXXX(5, 6);
                                n1 = node_list[2];
                                n2 = node_list[1];
                                ADDVERTEXXX(2, 1);
                                n2 = node_list[6];
                                ADDVERTEXXX(2, 6);
                                n2 = node_list[3];
                                ADDVERTEXXX(2, 3);
                                n1 = node_list[7];
                                n2 = node_list[4];
                                ADDVERTEXXX(7, 4);
                                n2 = node_list[3];
                                ADDVERTEXXX(7, 3);
                                n2 = node_list[6];
                                ADDVERTEXXX(7, 6);
                            }
                            break;
                        }
                    }
                }
                (*n_1)++;
                (*n_2)++;
                (*n_3)++;
                (*n_4)++;
                (*n_5)++;
                (*n_6)++;
                (*n_7)++;
                (*n_8)++;
            }
            (*n_1)++;
            (*n_2)++;
            (*n_3)++;
            (*n_4)++;
            (*n_5)++;
            (*n_6)++;
            (*n_7)++;
            (*n_8)++;
        }
        (*n_1) += z_size;
        (*n_2) += z_size;
        (*n_3) += z_size;
        (*n_4) += z_size;
        (*n_5) += z_size;
        (*n_6) += z_size;
        (*n_7) += z_size;
        (*n_8) += z_size;
    }
}

void RECT_IsoPlane::createIsoPlane()
{
    int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int node_list[8];
    int x_add[] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    int y_add[] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    int z_add[] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk;
    int no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    cutting_info *C_Info;

    int cell = 0;
    for (ii = 0; ii < x_size - 1; ++ii)
    {
        for (jj = 0; jj < y_size - 1; ++jj)
        {
            for (kk = 0; kk < z_size - 1; ++kk, ++cell)
            {
                if (iblank == NULL || iblank[*n_1] != '\0')
                {
                    bitmap = node_table[*n_1].side | node_table[*n_2].side << 1
                             | node_table[*n_3].side << 2 | node_table[*n_4].side << 3
                             | node_table[*n_5].side << 4 | node_table[*n_6].side << 5
                             | node_table[*n_7].side << 6 | node_table[*n_8].side << 7;

                    // bitmap is now an index to the Cuttingtable
                    C_Info = Cutting_Info[TYPE_HEXAGON] + bitmap;

                    numIntersections = C_Info->nvert;

                    if (numIntersections)
                    {
                        polygon_nodes = C_Info->node_pairs;
                        switch (numIntersections)
                        {
                        case 1:
                            num_triangles++;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 2:
                            num_triangles += 2;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 3:
                            /*
                         *      Something of a special case here:  If the average of the vertices
                         *      is greater than the isovalue, we create two separated polygons
                         *      at the vertices.  If it is less, then we make a little valley
                         *      shape.
                         */
                            no1 = node_list[*(polygon_nodes)];
                            no2 = node_list[*(polygon_nodes + 1)];
                            no3 = node_list[*(polygon_nodes + 2)];
                            no4 = node_list[*(polygon_nodes + 3)];
                            no5 = node_list[*(polygon_nodes + 4)];
                            no6 = node_list[*(polygon_nodes + 5)];
                            if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) < 0)
                            {
                                num_triangles += 2;
                                n1 = no1;
                                n2 = no2;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n2 = no4;
                                ADDVERTEXX03;
                                n1 = no5;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                                n2 = no3;
                                ADDVERTEXX(4, 2);
                                n2 = no6;
                                ADDVERTEXX(4, 5);
                            }
                            else
                            {
                                num_triangles += 4;
                                n1 = no1;
                                n2 = no2;
                                vertex1 = vertex;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n1 = no5;
                                n2 = no3;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 2);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no6;
                                ADDVERTEXX(4, 6);
                                n1 = no1;
                                n2 = no4;
                                vertex1 = vertex;
                                ADDVERTEXX04;
                                n2 = no2;
                                ADDVERTEXX01;
                                n1 = no5;
                                n2 = no6;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 5);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                            }
                            break;
                        case 4:
                            /*
                         *      Something of a special case here:  If the average of the vertices
                         *      is smaller than the isovalue, we create two separated polygons
                         *      at the vertices.  If it is less, then we make a little valley
                         *      shape.
                         */
                            no1 = node_list[*(polygon_nodes)];
                            no2 = node_list[*(polygon_nodes + 1)];
                            no3 = node_list[*(polygon_nodes + 2)];
                            no4 = node_list[*(polygon_nodes + 3)];
                            no5 = node_list[*(polygon_nodes + 4)];
                            no6 = node_list[*(polygon_nodes + 5)];
                            if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) > 0)
                            {
                                num_triangles += 2;
                                n1 = no1;
                                n2 = no2;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n2 = no4;
                                ADDVERTEXX03;
                                n1 = no5;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                                n2 = no3;
                                ADDVERTEXX(4, 2);
                                n2 = no6;
                                ADDVERTEXX(4, 5);
                            }
                            else
                            {
                                num_triangles += 4;
                                n1 = no1;
                                n2 = no2;
                                vertex1 = vertex;
                                ADDVERTEXX01;
                                n2 = no3;
                                ADDVERTEXX02;
                                n1 = no5;
                                n2 = no3;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 2);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no6;
                                ADDVERTEXX(4, 5);
                                n1 = no1;
                                n2 = no4;
                                vertex1 = vertex;
                                ADDVERTEXX03;
                                n2 = no2;
                                ADDVERTEXX01;
                                n1 = no5;
                                n2 = no6;
                                vertex2 = vertex;
                                ADDVERTEXX(4, 5);
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no4;
                                ADDVERTEXX(4, 3);
                            }
                            break;
                        case 5:
                            num_triangles += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 6:
                            num_triangles += 2;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            break;
                        case 7:
                            num_triangles += 2;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 8:
                            num_triangles += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 9:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            polygon_nodes += 2;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX(3, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 10:
                            num_triangles += 3;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            break;
                        case 11:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            polygon_nodes += 3;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 12:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX(2, 1);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX(2, 3);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*(polygon_nodes + 5)];
                            n2 = node_list[*(polygon_nodes + 4)];
                            ADDVERTEXX(5, 4);
                            break;
                        case 13:
                            num_triangles += 4;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            n2 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX03;
                            polygon_nodes += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            vertex2 = vertex;
                            n1 = node_list[*(polygon_nodes + 3)];
                            ADDVERTEXX(3, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*(polygon_nodes + 4)];
                            ADDVERTEXX(3, 4);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*(polygon_nodes + 5)];
                            n2 = node_list[*(polygon_nodes + 6)];
                            ADDVERTEXX(5, 6);
                            break;
                        case 14:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            polygon_nodes += 3;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            ADDVERTEXX01;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*(polygon_nodes + 2)];
                            ADDVERTEXX02;
                            break;
                        case 15:
                            num_triangles += 4;
                            if (*polygon_nodes)
                            {
                                n1 = node_list[1];
                                n2 = node_list[0];
                                ADDVERTEXXX(1, 0);
                                n2 = node_list[5];
                                ADDVERTEXXX(1, 5);
                                n2 = node_list[2];
                                ADDVERTEXXX(1, 2);
                                n1 = node_list[4];
                                n2 = node_list[5];
                                ADDVERTEXXX(4, 5);
                                n2 = node_list[0];
                                ADDVERTEXXX(4, 0);
                                n2 = node_list[7];
                                ADDVERTEXXX(4, 7);
                                n1 = node_list[6];
                                n2 = node_list[2];
                                ADDVERTEXXX(6, 2);
                                n2 = node_list[5];
                                ADDVERTEXXX(6, 5);
                                n2 = node_list[7];
                                ADDVERTEXXX(6, 7);
                                n1 = node_list[3];
                                n2 = node_list[0];
                                ADDVERTEXXX(3, 0);
                                n2 = node_list[2];
                                ADDVERTEXXX(3, 2);
                                n2 = node_list[7];
                                ADDVERTEXXX(3, 7);
                            }
                            else
                            {
                                n1 = node_list[0];
                                n2 = node_list[1];
                                ADDVERTEXXX(0, 1);
                                n2 = node_list[3];
                                ADDVERTEXXX(0, 3);
                                n2 = node_list[4];
                                ADDVERTEXXX(0, 4);
                                n1 = node_list[5];
                                n2 = node_list[1];
                                ADDVERTEXXX(5, 1);
                                n2 = node_list[4];
                                ADDVERTEXXX(5, 4);
                                n2 = node_list[6];
                                ADDVERTEXXX(5, 6);
                                n1 = node_list[2];
                                n2 = node_list[1];
                                ADDVERTEXXX(2, 1);
                                n2 = node_list[6];
                                ADDVERTEXXX(2, 6);
                                n2 = node_list[3];
                                ADDVERTEXXX(2, 3);
                                n1 = node_list[7];
                                n2 = node_list[4];
                                ADDVERTEXXX(7, 4);
                                n2 = node_list[3];
                                ADDVERTEXXX(7, 3);
                                n2 = node_list[6];
                                ADDVERTEXXX(7, 6);
                            }
                            break;
                        }
                    }
                }
                (*n_1)++;
                (*n_2)++;
                (*n_3)++;
                (*n_4)++;
                (*n_5)++;
                (*n_6)++;
                (*n_7)++;
                (*n_8)++;
            }
            (*n_1)++;
            (*n_2)++;
            (*n_3)++;
            (*n_4)++;
            (*n_5)++;
            (*n_6)++;
            (*n_7)++;
            (*n_8)++;
        }
        (*n_1) += z_size;
        (*n_2) += z_size;
        (*n_3) += z_size;
        (*n_4) += z_size;
        (*n_5) += z_size;
        (*n_6) += z_size;
        (*n_7) += z_size;
        (*n_8) += z_size;
    }
}

bool STR_IsoPlane::createIsoPlane()
{
    int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int node_list[8];
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk;
    int no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    cutting_info *C_Info;

    for (ii = 0; ii < x_size - 1; ii++)
    {
        for (jj = 0; jj < y_size - 1; jj++)
        {
            for (kk = 0; kk < z_size - 1; kk++)
            {
                if (iblank == NULL || iblank[*n_1] != '\0')
                {
                    bitmap = node_table[*n_1].side | node_table[*n_2].side << 1
                             | node_table[*n_3].side << 2 | node_table[*n_4].side << 3
                             | node_table[*n_5].side << 4 | node_table[*n_6].side << 5
                             | node_table[*n_7].side << 6 | node_table[*n_8].side << 7;

                    // bitmap is now an index to the Cuttingtable
                    C_Info = Cutting_Info[TYPE_HEXAGON] + bitmap;
                    numIntersections = C_Info->nvert;

                    if (numIntersections)
                    {
                        polygon_nodes = C_Info->node_pairs;
                        switch (numIntersections)
                        {
                        case 1:
                            num_triangles++;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 2:
                            num_triangles += 2;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 3:
                            /*
                         *      Something of a special case here:  If the average of the vertices
                         *      is greater than the isovalue, we create two separated polygons
                         *      at the vertices.  If it is less, then we make a little valley
                         *      shape.
                         */
                            no1 = node_list[*polygon_nodes++];
                            no2 = node_list[*polygon_nodes++];
                            no3 = node_list[*polygon_nodes++];
                            no4 = node_list[*polygon_nodes++];
                            no5 = node_list[*polygon_nodes++];
                            no6 = node_list[*polygon_nodes++];
                            if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) < 0)
                            {
                                num_triangles += 2;
                                n1 = no1;
                                n2 = no2;
                                ADDVERTEX;
                                n2 = no3;
                                ADDVERTEX;
                                n2 = no4;
                                ADDVERTEX;
                                n1 = no5;
                                n2 = no4;
                                ADDVERTEX;
                                n2 = no3;
                                ADDVERTEX;
                                n2 = no6;
                                ADDVERTEX;
                            }
                            else
                            {
                                num_triangles += 4;
                                n1 = no1;
                                n2 = no2;
                                vertex1 = vertex;
                                ADDVERTEX;
                                n2 = no3;
                                ADDVERTEX;
                                n1 = no5;
                                n2 = no3;
                                vertex2 = vertex;
                                ADDVERTEX;
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no6;
                                ADDVERTEX;
                                n1 = no1;
                                n2 = no4;
                                vertex1 = vertex;
                                ADDVERTEX;
                                n2 = no2;
                                ADDVERTEX;
                                n1 = no5;
                                n2 = no6;
                                vertex2 = vertex;
                                ADDVERTEX;
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no4;
                                ADDVERTEX;
                            }
                            break;
                        case 4:
                            /*
                         *      Something of a special case here:  If the average of the vertices
                         *      is smaller than the isovalue, we create two separated polygons
                         *      at the vertices.  If it is less, then we make a little valley
                         *      shape.
                         */
                            no1 = node_list[*polygon_nodes++];
                            no2 = node_list[*polygon_nodes++];
                            no3 = node_list[*polygon_nodes++];
                            no4 = node_list[*polygon_nodes++];
                            no5 = node_list[*polygon_nodes++];
                            no6 = node_list[*polygon_nodes++];
                            if ((node_table[no1].dist + node_table[no3].dist + node_table[no4].dist + node_table[no5].dist) > 0)
                            {
                                num_triangles += 2;
                                n1 = no1;
                                n2 = no2;
                                ADDVERTEX;
                                n2 = no3;
                                ADDVERTEX;
                                n2 = no4;
                                ADDVERTEX;
                                n1 = no5;
                                n2 = no4;
                                ADDVERTEX;
                                n2 = no3;
                                ADDVERTEX;
                                n2 = no6;
                                ADDVERTEX;
                            }
                            else
                            {
                                num_triangles += 4;
                                n1 = no1;
                                n2 = no2;
                                vertex1 = vertex;
                                ADDVERTEX;
                                n2 = no3;
                                ADDVERTEX;
                                n1 = no5;
                                n2 = no3;
                                vertex2 = vertex;
                                ADDVERTEX;
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no6;
                                ADDVERTEX;
                                n1 = no1;
                                n2 = no4;
                                vertex1 = vertex;
                                ADDVERTEX;
                                n2 = no2;
                                ADDVERTEX;
                                n1 = no5;
                                n2 = no6;
                                vertex2 = vertex;
                                ADDVERTEX;
                                *vertex = *vertex1;
                                vertex++;
                                *vertex = *vertex2;
                                vertex++;
                                n2 = no4;
                                ADDVERTEX;
                            }
                            break;
                        case 5:
                            num_triangles += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 6:
                            num_triangles += 2;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 7:
                            num_triangles += 2;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 8:
                            num_triangles += 3;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 9:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 10:
                            num_triangles += 3;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 11:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 12:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            n1 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 13:
                            num_triangles += 4;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            vertex2 = vertex;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 14:
                            num_triangles += 4;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex1 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            vertex2 = vertex;
                            n1 = node_list[*polygon_nodes++];
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = node_list[*polygon_nodes++];
                            ADDVERTEX;
                            break;
                        case 15:
                            num_triangles += 4;
                            if (*polygon_nodes)
                            {
                                n1 = node_list[1];
                                n2 = node_list[0];
                                ADDVERTEX;
                                n2 = node_list[5];
                                ADDVERTEX;
                                n2 = node_list[2];
                                ADDVERTEX;
                                n1 = node_list[4];
                                n2 = node_list[5];
                                ADDVERTEX;
                                n2 = node_list[0];
                                ADDVERTEX;
                                n2 = node_list[7];
                                ADDVERTEX;
                                n1 = node_list[6];
                                n2 = node_list[2];
                                ADDVERTEX;
                                n2 = node_list[5];
                                ADDVERTEX;
                                n2 = node_list[7];
                                ADDVERTEX;
                                n1 = node_list[3];
                                n2 = node_list[0];
                                ADDVERTEX;
                                n2 = node_list[2];
                                ADDVERTEX;
                                n2 = node_list[7];
                                ADDVERTEX;
                            }
                            else
                            {
                                n1 = node_list[0];
                                n2 = node_list[1];
                                ADDVERTEX;
                                n2 = node_list[3];
                                ADDVERTEX;
                                n2 = node_list[4];
                                ADDVERTEX;
                                n1 = node_list[5];
                                n2 = node_list[1];
                                ADDVERTEX;
                                n2 = node_list[4];
                                ADDVERTEX;
                                n2 = node_list[6];
                                ADDVERTEX;
                                n1 = node_list[2];
                                n2 = node_list[1];
                                ADDVERTEX;
                                n2 = node_list[6];
                                ADDVERTEX;
                                n2 = node_list[3];
                                ADDVERTEX;
                                n1 = node_list[7];
                                n2 = node_list[4];
                                ADDVERTEX;
                                n2 = node_list[3];
                                ADDVERTEX;
                                n2 = node_list[6];
                                ADDVERTEX;
                            }
                            break;
                        }
                    }
                }
                (*n_1)++;
                (*n_2)++;
                (*n_3)++;
                (*n_4)++;
                (*n_5)++;
                (*n_6)++;
                (*n_7)++;
                (*n_8)++;
            }
            (*n_1)++;
            (*n_2)++;
            (*n_3)++;
            (*n_4)++;
            (*n_5)++;
            (*n_6)++;
            (*n_7)++;
            (*n_8)++;
        }
        (*n_1) += z_size;
        (*n_2) += z_size;
        (*n_3) += z_size;
        (*n_4) += z_size;
        (*n_5) += z_size;
        (*n_6) += z_size;
        (*n_7) += z_size;
        (*n_8) += z_size;
    }
    return true;
}

bool POLYHEDRON_IsoPlane::createIsoPlane()
{
    int element;
    int next_elem_index;
    int start_vertex;
    int sign;
    int new_sign;

    /*********************/
    /* Auxiliary Variables */
    /*********************/

    bool start_vertex_set;
    bool cell_intersection;

    int i;
    int j;
    int k;
    int current_index;
    int next_index;
    int new_elem_address;
    int new_conn_address;

    int *temp_elem_list;
    int *temp_conn_list;
    int *temp_poly_list;
    int *temp_idx_list;

    float *new_x_coord_in;
    float *new_y_coord_in;
    float *new_z_coord_in;
    float *temp_isodata_in;
    float *temp_sdata_in;
    float *temp_udata_in;
    float *temp_vdata_in;
    float *temp_wdata_in;

    vector<int> temp_elem_in;
    vector<int> temp_conn_in;
    vector<int> new_temp_conn_in;
    vector<int> temp_polygon_list;
    vector<int> temp_index_list;
    vector<int> temp_vertex_list;
    vector<int> temp_elem_out;
    vector<int> temp_conn_out;

    vector<float> temp_x_coord_out;
    vector<float> temp_y_coord_out;
    vector<float> temp_z_coord_out;
    vector<float> temp_sdata_out;
    vector<float> temp_udata_out;
    vector<float> temp_vdata_out;
    vector<float> temp_wdata_out;

    /*********************/
    /* Contour Variables */
    /*********************/

    CONTOUR contour;
    ISOSURFACE_EDGE_INTERSECTION_VECTOR intsec_vector;
    TESSELATION triangulation;

    /* Avoid Unnecessary Reallocations */
    contour.ring.reserve(15);
    contour.ring_index.reserve(5);
    contour.polyhedron_faces.reserve(15);
    intsec_vector.reserve(15);

    temp_sdata_in = NULL;
    temp_udata_in = NULL;
    temp_vdata_in = NULL;
    temp_wdata_in = NULL;

    temp_elem_out.clear();
    temp_conn_out.clear();
    temp_x_coord_out.clear();
    temp_y_coord_out.clear();
    temp_z_coord_out.clear();
    temp_sdata_out.clear();
    temp_udata_out.clear();
    temp_vdata_out.clear();
    temp_wdata_out.clear();

    for (element = 0; element < num_elem; element++)
    {
        start_vertex_set = false;
        cell_intersection = false;

        next_elem_index = (element < num_elem - 1) ? el[element + 1] : num_conn;

        /* Avoid additional calculations if the cell is not cut by the isosurface */
        for (i = el[element]; i < next_elem_index; i++)
        {
            if (i_in[cl[i]] > _isovalue)
            {
                if (i == el[element])
                {
                    sign = 1;
                }

                new_sign = 1;
            }

            else if (i_in[cl[i]] == _isovalue)
            {
                cell_intersection = true;
                break;
            }

            else if (i_in[cl[i]] < _isovalue)
            {
                if (i == el[element])
                {
                    sign = -1;
                }

                new_sign = -1;
            }

            if (new_sign != sign)
            {
                cell_intersection = true;
                break;
            }
        }

        if (cell_intersection == true)
        {
            temp_elem_in.clear();
            temp_conn_in.clear();
            temp_vertex_list.clear();
            new_temp_conn_in.clear();
            temp_index_list.clear();
            temp_polygon_list.clear();
            contour.ring.clear();
            contour.ring_index.clear();
            contour.polyhedron_faces.clear();
            intsec_vector.clear();

            switch (tl[element])
            {
            case TYPE_POLYHEDRON:

                /* Construct DO_Polygons Element and Connectivity Lists */
                for (j = el[element]; j < next_elem_index; j++)
                {
                    if (j == el[element] && start_vertex_set == false)
                    {
                        start_vertex = cl[el[element]];
                        temp_elem_in.push_back((int)temp_conn_in.size());
                        temp_conn_in.push_back(start_vertex);
                        start_vertex_set = true;
                    }

                    if (j > el[element] && start_vertex_set == true)
                    {
                        if (cl[j] != start_vertex)
                        {
                            temp_conn_in.push_back(cl[j]);
                        }

                        else
                        {
                            start_vertex_set = false;
                            continue;
                        }
                    }

                    if (j > el[element] && start_vertex_set == false)
                    {
                        start_vertex = cl[j];
                        temp_elem_in.push_back((int)temp_conn_in.size());
                        temp_conn_in.push_back(start_vertex);
                        start_vertex_set = true;
                    }
                }

                /* Construct Vertex List */
                for (i = 0; i < temp_conn_in.size(); i++)
                {
                    if (temp_vertex_list.size() == 0)
                    {
                        temp_vertex_list.push_back(temp_conn_in[i]);
                    }

                    else
                    {
                        if (find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[i]) == temp_vertex_list.end())
                        {
                            temp_vertex_list.push_back(temp_conn_in[i]);
                        }
                    }
                }

                sort(temp_vertex_list.begin(), temp_vertex_list.end());
                break;

            case TYPE_HEXAEDER:

                /* Construct DO_Polygons Element and Connectivity Lists */
                for (j = el[element]; j < next_elem_index; j++)
                {
                    if (j == el[element])
                    {
                        temp_elem_in.push_back((int)temp_conn_in.size());
                    }

                    if ((j - el[element]) == 4)
                    {
                        temp_elem_in.push_back(4);
                    }

                    temp_conn_in.push_back(cl[j]);
                }

                /* Construct Vertex List */
                for (i = 0; i < temp_conn_in.size(); i++)
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }

                sort(temp_vertex_list.begin(), temp_vertex_list.end());

                /* Complete DO_Polygons Element and Connectivity Lists */
                temp_elem_in.push_back(8);

                temp_conn_in.push_back(temp_conn_in[1]);
                temp_conn_in.push_back(temp_conn_in[2]);
                temp_conn_in.push_back(temp_conn_in[6]);
                temp_conn_in.push_back(temp_conn_in[5]);

                temp_elem_in.push_back(12);

                temp_conn_in.push_back(temp_conn_in[0]);
                temp_conn_in.push_back(temp_conn_in[3]);
                temp_conn_in.push_back(temp_conn_in[7]);
                temp_conn_in.push_back(temp_conn_in[4]);

                temp_elem_in.push_back(16);

                temp_conn_in.push_back(temp_conn_in[2]);
                temp_conn_in.push_back(temp_conn_in[3]);
                temp_conn_in.push_back(temp_conn_in[7]);
                temp_conn_in.push_back(temp_conn_in[6]);

                temp_elem_in.push_back(20);

                temp_conn_in.push_back(temp_conn_in[1]);
                temp_conn_in.push_back(temp_conn_in[0]);
                temp_conn_in.push_back(temp_conn_in[4]);
                temp_conn_in.push_back(temp_conn_in[5]);
                break;

            case TYPE_PRISM:

                /* Construct DO_Polygons Element and Connectivity Lists */
                for (j = el[element]; j < next_elem_index; j++)
                {
                    if (j == el[element])
                    {
                        temp_elem_in.push_back((int)temp_conn_in.size());
                    }

                    if ((j - el[element]) == 3)
                    {
                        temp_elem_in.push_back(3);
                    }

                    temp_conn_in.push_back(cl[j]);
                }

                /* Construct Vertex List */
                for (i = 0; i < temp_conn_in.size(); i++)
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }

                /* Complete DO_Polygons Element and Connectivity Lists */
                temp_elem_in.push_back(6);

                temp_conn_in.push_back(temp_conn_in[0]);
                temp_conn_in.push_back(temp_conn_in[1]);
                temp_conn_in.push_back(temp_conn_in[4]);
                temp_conn_in.push_back(temp_conn_in[3]);

                temp_elem_in.push_back(10);

                temp_conn_in.push_back(temp_conn_in[1]);
                temp_conn_in.push_back(temp_conn_in[2]);
                temp_conn_in.push_back(temp_conn_in[5]);
                temp_conn_in.push_back(temp_conn_in[4]);

                temp_elem_in.push_back(14);

                temp_conn_in.push_back(temp_conn_in[0]);
                temp_conn_in.push_back(temp_conn_in[2]);
                temp_conn_in.push_back(temp_conn_in[5]);
                temp_conn_in.push_back(temp_conn_in[3]);
                break;

            case TYPE_PYRAMID:

                /* Construct DO_Polygons Element and Connectivity Lists */
                for (j = el[element]; j < next_elem_index; j++)
                {
                    if (j == el[element])
                    {
                        temp_elem_in.push_back((int)temp_conn_in.size());
                    }

                    if ((j - el[element]) == 4)
                    {
                        temp_elem_in.push_back(4);
                    }

                    temp_conn_in.push_back(cl[j]);
                }

                /* Construct Vertex List */
                for (i = 0; i < temp_conn_in.size(); i++)
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }

                /* Complete DO_Polygons Element and Connectivity Lists */
                temp_conn_in.push_back(temp_conn_in[1]);
                temp_conn_in.push_back(temp_conn_in[2]);

                temp_elem_in.push_back(7);

                temp_conn_in.push_back(temp_conn_in[4]);
                temp_conn_in.push_back(temp_conn_in[0]);
                temp_conn_in.push_back(temp_conn_in[1]);

                temp_elem_in.push_back(10);

                temp_conn_in.push_back(temp_conn_in[4]);
                temp_conn_in.push_back(temp_conn_in[3]);
                temp_conn_in.push_back(temp_conn_in[0]);

                temp_elem_in.push_back(13);

                temp_conn_in.push_back(temp_conn_in[4]);
                temp_conn_in.push_back(temp_conn_in[2]);
                temp_conn_in.push_back(temp_conn_in[3]);
                break;

            case TYPE_TETRAHEDER:

                /* Construct DO_Polygons Element and Connectivity Lists */
                for (j = el[element]; j < next_elem_index; j++)
                {
                    if (j == el[element])
                    {
                        temp_elem_in.push_back((int)temp_conn_in.size());
                    }

                    if ((j - el[element]) == 3)
                    {
                        temp_elem_in.push_back(3);
                    }

                    temp_conn_in.push_back(cl[j]);
                }

                /* Construct Vertex List */
                for (i = 0; i < temp_conn_in.size(); i++)
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }

                /* Complete DO_Polygons Element and Connectivity Lists */
                temp_conn_in.push_back(temp_conn_in[1]);
                temp_conn_in.push_back(temp_conn_in[2]);

                temp_elem_in.push_back(6);

                temp_conn_in.push_back(temp_conn_in[3]);
                temp_conn_in.push_back(temp_conn_in[0]);
                temp_conn_in.push_back(temp_conn_in[1]);

                temp_elem_in.push_back(9);

                temp_conn_in.push_back(temp_conn_in[3]);
                temp_conn_in.push_back(temp_conn_in[2]);
                temp_conn_in.push_back(temp_conn_in[0]);
                break;
            }

            /* Construct New Connectivity List */
            for (i = 0; i < temp_conn_in.size(); i++)
            {
                vector<int>::iterator pos = find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[i]);

                if (pos != temp_vertex_list.end())
                {
                    new_temp_conn_in.push_back((int)(pos - temp_vertex_list.begin()));
                }
            }

            /* Construct DO_Polygons Polygon and Index Lists */
            for (i = 0; i < temp_vertex_list.size(); i++)
            {
                temp_index_list.push_back((int)temp_polygon_list.size());
                for (j = 0; j < temp_elem_in.size(); j++)
                {
                    current_index = temp_elem_in[j];
                    next_index = (j < temp_elem_in.size() - 1) ? temp_elem_in[j + 1] : (int)temp_conn_in.size();

                    for (k = current_index; k < next_index; k++)
                    {
                        if (new_temp_conn_in[k] == i)
                        {
                            temp_polygon_list.push_back(j);
                        }
                    }
                }
            }

            temp_elem_list = new int[temp_elem_in.size()];
            temp_conn_list = new int[temp_conn_in.size()];
            temp_poly_list = new int[temp_polygon_list.size()];
            temp_idx_list = new int[temp_index_list.size()];
            new_x_coord_in = new float[temp_vertex_list.size()];
            new_y_coord_in = new float[temp_vertex_list.size()];
            new_z_coord_in = new float[temp_vertex_list.size()];
            temp_isodata_in = new float[temp_vertex_list.size()];

            for (i = 0; i < temp_elem_in.size(); i++)
            {
                temp_elem_list[i] = temp_elem_in[i];
            }

            for (i = 0; i < new_temp_conn_in.size(); i++)
            {
                temp_conn_list[i] = new_temp_conn_in[i];
            }

            for (i = 0; i < temp_index_list.size(); i++)
            {
                temp_idx_list[i] = temp_index_list[i];
            }

            for (i = 0; i < temp_polygon_list.size(); i++)
            {
                temp_poly_list[i] = temp_polygon_list[i];
            }

            /* Construct New Set of Coordinates */
            for (i = 0; i < temp_vertex_list.size(); i++)
            {
                new_x_coord_in[i] = x_in[temp_vertex_list[i]];
                new_y_coord_in[i] = y_in[temp_vertex_list[i]];
                new_z_coord_in[i] = z_in[temp_vertex_list[i]];
            }

            /* Construct New Input Data Set */
            for (i = 0; i < temp_vertex_list.size(); i++)
            {
                temp_isodata_in[i] = i_in[temp_vertex_list[i]];
            }

            if (_isConnected)
            {
                if (Datatype)
                {
                    temp_sdata_in = new float[temp_vertex_list.size()];

                    for (i = 0; i < temp_vertex_list.size(); i++)
                    {
                        temp_sdata_in[i] = s_in[temp_vertex_list[i]];
                    }
                }

                else
                {
                    temp_udata_in = new float[temp_vertex_list.size()];
                    temp_vdata_in = new float[temp_vertex_list.size()];
                    temp_wdata_in = new float[temp_vertex_list.size()];

                    for (i = 0; i < temp_vertex_list.size(); i++)
                    {
                        temp_udata_in[i] = u_in[temp_vertex_list[i]];
                        temp_vdata_in[i] = v_in[temp_vertex_list[i]];
                        temp_wdata_in[i] = w_in[temp_vertex_list[i]];
                    }
                }
            }

            /***********************/
            /* Generate Isosurface */
            /***********************/

            create_isocontour((int)temp_elem_in.size(), temp_elem_list, (int)new_temp_conn_in.size(), temp_conn_list, (int)temp_vertex_list.size(), new_x_coord_in, new_y_coord_in, new_z_coord_in, temp_poly_list, temp_idx_list, temp_isodata_in, temp_sdata_in, temp_udata_in, temp_vdata_in, temp_wdata_in, _isovalue, intsec_vector, contour, triangulation);

            /* Construct Partial Output */
            if (intsec_vector.size() >= 3)
            {
                if (temp_conn_out.size() == 0)
                {
                    for (i = 0; i < triangulation.size(); i++)
                    {
                        temp_elem_out.push_back(i * 3);
                        temp_conn_out.push_back(triangulation[i].vertex1);
                        temp_conn_out.push_back(triangulation[i].vertex2);
                        temp_conn_out.push_back(triangulation[i].vertex3);
                    }

                    new_elem_address = (int)temp_conn_out.size();
                    new_conn_address = (int)intsec_vector.size();
                }

                else
                {
                    for (i = 0; i < triangulation.size(); i++)
                    {
                        temp_elem_out.push_back(i * 3 + new_elem_address);
                        temp_conn_out.push_back(triangulation[i].vertex1 + new_conn_address);
                        temp_conn_out.push_back(triangulation[i].vertex2 + new_conn_address);
                        temp_conn_out.push_back(triangulation[i].vertex3 + new_conn_address);
                    }

                    new_elem_address = (int)temp_conn_out.size();
                    new_conn_address += (int)intsec_vector.size();
                }

                for (i = 0; i < intsec_vector.size(); i++)
                {
                    temp_x_coord_out.push_back(intsec_vector[i].intersection.x);
                    temp_y_coord_out.push_back(intsec_vector[i].intersection.y);
                    temp_z_coord_out.push_back(intsec_vector[i].intersection.z);
                }

                if (!_isConnected)
                {
                    for (i = 0; i < intsec_vector.size(); i++)
                    {
                        temp_sdata_out.push_back(_isovalue);
                    }
                }

                else if (Datatype)
                {
                    for (i = 0; i < intsec_vector.size(); i++)
                    {
                        temp_sdata_out.push_back(intsec_vector[i].data_vertex_int.v[0]);
                    }
                }

                else
                {
                    for (i = 0; i < intsec_vector.size(); i++)
                    {
                        temp_udata_out.push_back(intsec_vector[i].data_vertex_int.v[0]);
                        temp_vdata_out.push_back(intsec_vector[i].data_vertex_int.v[1]);
                        temp_wdata_out.push_back(intsec_vector[i].data_vertex_int.v[2]);
                    }
                }
            }

            delete[] temp_elem_list;
            delete[] temp_conn_list;
            delete[] temp_poly_list;
            delete[] temp_idx_list;
            delete[] new_x_coord_in;
            delete[] new_y_coord_in;
            delete[] new_z_coord_in;
            delete[] temp_isodata_in;

            if (_isConnected)
            {
                if (Datatype)
                {
                    delete[] temp_sdata_in;
                }

                else
                {
                    delete[] temp_udata_in;
                    delete[] temp_vdata_in;
                    delete[] temp_wdata_in;
                }
            }
        }
    }

    /********************/
    /* Generate Output */
    /********************/

    if (temp_conn_out.size() == 0)
    {
        /**************************/
        /* Generate NULL Output  */
        /**************************/

        num_coord_out = 0;
        num_elem_out = 0;
        num_conn_out = 0;
    }

    else
    {
        x_coord_out = new float[temp_x_coord_out.size()];
        y_coord_out = new float[temp_y_coord_out.size()];
        z_coord_out = new float[temp_z_coord_out.size()];
        elem_out = new int[temp_elem_out.size()];
        conn_out = new int[temp_conn_out.size()];
        sdata_out = new float[temp_sdata_out.size()];

        for (i = 0; i < temp_x_coord_out.size(); i++)
        {
            x_coord_out[i] = temp_x_coord_out[i];
            y_coord_out[i] = temp_y_coord_out[i];
            z_coord_out[i] = temp_z_coord_out[i];
        }

        for (i = 0; i < temp_elem_out.size(); i++)
        {
            elem_out[i] = temp_elem_out[i];
        }

        for (i = 0; i < temp_conn_out.size(); i++)
        {
            conn_out[i] = temp_conn_out[i];
        }

        if (_isConnected)
        {
            if (Datatype)
            {
                sdata_out = new float[temp_sdata_out.size()];

                for (i = 0; i < temp_sdata_out.size(); i++)
                {
                    sdata_out[i] = temp_sdata_out[i];
                }
            }

            else
            {
                udata_out = new float[temp_udata_out.size()];
                vdata_out = new float[temp_vdata_out.size()];
                wdata_out = new float[temp_wdata_out.size()];

                for (i = 0; i < temp_udata_out.size(); i++)
                {
                    udata_out[i] = temp_udata_out[i];
                    vdata_out[i] = temp_vdata_out[i];
                    wdata_out[i] = temp_wdata_out[i];
                }
            }
        }

        else
        {
            sdata_out = new float[temp_sdata_out.size()];

            for (i = 0; i < temp_sdata_out.size(); i++)
            {
                sdata_out[i] = temp_sdata_out[i];
            }
        }

        num_coord_out = (int)temp_x_coord_out.size();
        num_elem_out = (int)temp_elem_out.size();
        num_conn_out = (int)temp_conn_out.size();

        temp_x_coord_out.clear();
        temp_y_coord_out.clear();
        temp_z_coord_out.clear();
        temp_elem_out.clear();
        temp_conn_out.clear();
        temp_sdata_out.clear();
        temp_udata_out.clear();
        temp_vdata_out.clear();
        temp_wdata_out.clear();
    }

    return true;
}

void POLYHEDRON_IsoPlane::create_isocontour(int num_elem_in, int *elem_in,
                                            int num_conn_in, int *conn_in,
                                            int num_coord_in, float *x_coord_in, float *y_coord_in, float *z_coord_in,
                                            int *polygon_list, int *index_list, float *isodata_in, float *sdata_in, float *udata_in, float *vdata_in, float *wdata_in,
                                            float isovalue, ISOSURFACE_EDGE_INTERSECTION_VECTOR &intsec_vector, CONTOUR &contour, TESSELATION &triangulation)
{
    int i;
    int j;
    //int face_count;
    int current_face;
    int edge_vertex1;
    int edge_vertex2;
    int new_edge_vertex1;
    int new_edge_vertex2;

    float data_vertex1;
    float data_vertex2;

    /************************/
    /* Contouring Variables */
    /************************/

    bool ring_open;
    bool remaining_intersections;
    bool improper_topology;
    bool abort_tracing_isocontour;

    int ring_counter;
    int ring_end = 0; // was uninitialized, is this OK?
    int num_of_rings;
    int start_ring_vertex1;
    int start_ring_vertex2;
    //int end_ring_vertex1;
    //int end_ring_vertex2;
    //int rem_int_index;
    int rem_int_flag2;

    vector<int> loop;

    /***************************************/
    /* Calculation of the edge intersections  */
    /***************************************/

    improper_topology = false;
    abort_tracing_isocontour = false;

    intsec_vector = calculate_intersections(num_elem_in, elem_in, num_conn_in, conn_in, x_coord_in, y_coord_in, z_coord_in, isodata_in, isovalue, sdata_in, udata_in, vdata_in, wdata_in, _isConnected, improper_topology);

    /********************************/
    /* Generation of the Contour(s)  */
    /********************************/

    //face_count = 0;

    /* Existence of one or more rings was determined */
    if (intsec_vector.size() >= 3)
    {
        edge_vertex1 = intsec_vector[0].vertex1;
        edge_vertex2 = intsec_vector[0].vertex2;
        data_vertex1 = intsec_vector[0].data_vertex1;
        data_vertex2 = intsec_vector[0].data_vertex2;

        new_edge_vertex1 = edge_vertex1;
        new_edge_vertex2 = edge_vertex2;

        start_ring_vertex1 = edge_vertex1;
        start_ring_vertex2 = edge_vertex2;
        //end_ring_vertex1 = edge_vertex1;
        //end_ring_vertex2 = edge_vertex2;

        ring_counter = 0;
        num_of_rings = 0;
        ring_open = true;

        do
        {
            /******************************************************************************/
            /*  Locate a face of the polyhedron which contains the edge of the intersection */
            /******************************************************************************/

            find_current_face(contour, intsec_vector, edge_vertex1, edge_vertex2, data_vertex1, data_vertex2, isodata_in, elem_in, conn_in, index_list, polygon_list, num_coord_in, num_conn_in, num_elem_in, ring_counter, current_face, x_coord_in, y_coord_in, z_coord_in, improper_topology, abort_tracing_isocontour);

            /**********************************************/
            /* Determine the direction of the tracing route */
            /**********************************************/

            if (!abort_tracing_isocontour)
            {
                generate_isocontour(intsec_vector, data_vertex1, data_vertex2, edge_vertex1, edge_vertex2, new_edge_vertex1, new_edge_vertex2, elem_in, conn_in, isovalue, num_elem_in, num_conn_in, current_face, x_coord_in, y_coord_in, z_coord_in, improper_topology, abort_tracing_isocontour, contour, num_of_rings, ring_end);

                /*****************************/
                /* Update vertex information */
                /*****************************/

                edge_vertex1 = new_edge_vertex1;
                edge_vertex2 = new_edge_vertex2;

                data_vertex1 = isodata_in[edge_vertex1];
                data_vertex2 = isodata_in[edge_vertex2];

                /************************************/
                /* Check if the ring has been closed */
                /************************************/

                if ((new_edge_vertex1 == start_ring_vertex1) && (new_edge_vertex2 == start_ring_vertex2))
                {
                    ring_open = false;
                    contour.ring_index.push_back((int)contour.ring.size() - ring_counter);
                    num_of_rings++;
                    ring_end = (int)contour.ring.size();
                }

                else if ((new_edge_vertex2 == start_ring_vertex1) && (new_edge_vertex1 == start_ring_vertex2))
                {
                    ring_open = false;
                    contour.ring_index.push_back((int)contour.ring.size() - ring_counter);
                    num_of_rings++;
                    ring_end = (int)contour.ring.size();
                }

                /**********************************************/
                /* Check if there are any unused intersections */
                /**********************************************/

                rem_int_flag2 = 0;

                if (intsec_vector.size() == contour.ring.size())
                {
                    remaining_intersections = false;
                }

                else if (!ring_open && intsec_vector.size() - contour.ring.size() < 3)
                {
                    remaining_intersections = false;
                }

                else
                {
                    remaining_intersections = true;
                    if ((intsec_vector.size() != contour.ring.size()) && !ring_open)
                    {
                        for (i = 0; i < intsec_vector.size(); i++)
                        {
                            for (j = 0; j < contour.ring.size(); j++)
                            {
                                if (rem_int_flag2 == 0)
                                {
                                    if (contour.ring[j] == i)
                                    {
                                        remaining_intersections = false;
                                        break;
                                    }

                                    if (j == contour.ring.size() - 1)
                                    {
                                        //rem_int_index = i;
                                        rem_int_flag2 = 1;
                                        edge_vertex1 = intsec_vector[i].vertex1;
                                        edge_vertex2 = intsec_vector[i].vertex2;

                                        data_vertex1 = intsec_vector[i].data_vertex1;
                                        data_vertex2 = intsec_vector[i].data_vertex2;

                                        new_edge_vertex1 = edge_vertex1;
                                        new_edge_vertex2 = edge_vertex2;

                                        start_ring_vertex1 = edge_vertex1;
                                        start_ring_vertex2 = edge_vertex2;
                                        //end_ring_vertex1 = edge_vertex1;
                                        //end_ring_vertex2 = edge_vertex2;

                                        remaining_intersections = true;
                                        ring_open = true;
                                        ring_counter = 0;

                                        contour.polyhedron_faces.clear();
                                    }
                                }
                            }

                            if (rem_int_flag2 == 1)
                            {
                                break;
                            }
                        }
                    }
                }
            }
        } while (ring_open && remaining_intersections && !abort_tracing_isocontour);

        contour.polyhedron_faces.clear();
    }

    /********************/
    /* Generate Output */
    /********************/

    triangulation.clear();

    if (abort_tracing_isocontour && num_of_rings == 0)
    {
        contour.ring_index.clear();
        intsec_vector.clear();
    }

    else if (abort_tracing_isocontour && num_of_rings > 0)
    {
        contour.ring.erase(contour.ring.begin() + ring_end, contour.ring.end());
    }

    generate_tesselation(triangulation, contour, intsec_vector);
}

void POLYHEDRON_IsoPlane::createcoDistributedObjects(coOutputPort *p_GridOut, coOutputPort *p_DataOut)
{
    //    float *u_out,*v_out,*w_out;
    //    int *vl,*pl,i;
    coDoFloat *s_data_out;
    coDoVec3 *v_data_out;
    coDoPolygons *polygons_out;
    //    DO_TriangleStrips*     strips_out;
    //    DO_Unstructured_V3D_Data* normals_out;
    const char *DataOut = p_DataOut->getObjName();
    //    const char *NormalsOut    =  p_NormalsOut->getObjName();
    const char *GridOut = p_GridOut->getObjName();

    if (_isConnected)
    {
        if (Datatype) // (Scalar Data)
        {
            s_data_out = new coDoFloat(DataOut, num_coord_out, sdata_out);
            if (!s_data_out->objectOk())
            {
                Covise::sendError("ERROR: creation of data object 'dataOut' failed");
                return;
            }

            // delete s_data_out;
            p_DataOut->setCurrentObject(s_data_out);
        }

        else
        {
            v_data_out = new coDoVec3(DataOut, num_coord_out, udata_out, vdata_out, wdata_out);
            if (!v_data_out->objectOk())
            {
                Covise::sendError("ERROR: creation of data object 'dataOut' failed");
                return;
            }

            // delete v_data_out;
            p_DataOut->setCurrentObject(v_data_out);
        }
    }

    else
    {
        s_data_out = new coDoFloat(DataOut, num_coord_out, sdata_out);
        if (!s_data_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }

        // delete s_data_out;
        p_DataOut->setCurrentObject(s_data_out);
    }

    //    num_vertices=vertex-vertice_list;
    //    if(gennormals)
    //    {
    //       createNormals(genstrips);
    //       normals_out = new DO_Unstructured_V3D_Data(NormalsOut, num_coords,Normals_U,Normals_V,Normals_W);
    //       if(!normals_out->obj_ok())
    //       {
    //          Covise::send_error("ERROR: creation of data object 'normalsOut' failed");
    //          return;
    //       }
    //       // delete normals_out;
    //       p_NormalsOut->setObj(normals_out);
    //       delete [] Normals_U;
    //       delete [] Normals_V;
    //       delete [] Normals_W;
    //    }
    //    if(genstrips)
    //    {
    //       createStrips(gennormals);
    //       strips_out = new DO_TriangleStrips(GridOut,num_coords,coords_x,coords_y,coords_z,num_triangles+2*num_strips,ts_vertice_list,num_strips,ts_line_list);
    //       if(strips_out->obj_ok())
    //       {
    //          strips_out->set_attribute("vertexOrder","2");
    //          if(colorn[0] != '\0')
    //             strips_out->set_attribute("COLOR",colorn);
    //       }
    //       else
    //       {
    //          Covise::send_error("ERROR: creation of data object 'dataOut' failed");
    //          return;
    //       }
    //       // delete strips_out;
    //       p_GridOut->setObj(strips_out);
    //       delete [] ts_vertice_list;
    //       delete [] ts_line_list;
    //    }
    //    else
    //    {
    //       polygons_out = new DO_Polygons(GridOut,num_coords,num_vertices,num_triangles);
    polygons_out = new coDoPolygons(GridOut, num_coord_out, x_coord_out, y_coord_out, z_coord_out, num_conn_out, conn_out, num_elem_out, elem_out);
    //       if(polygons_out->obj_ok())
    //       {
    //          polygons_out->get_addresses(&u_out,&v_out,&w_out,&vl,&pl);
    //          memcpy(u_out,coords_x,num_coords*sizeof(float));
    //          memcpy(v_out,coords_y,num_coords*sizeof(float));
    //          memcpy(w_out,coords_z,num_coords*sizeof(float));
    //          memcpy(vl,vertice_list,num_vertices*sizeof(int));
    //          for(i=0;i<num_triangles;i++)
    //             pl[i]=i*3;
    //          polygons_out->set_attribute("vertexOrder","2");
    //          if(colorn[0] != '\0')
    //             polygons_out->set_attribute("COLOR",colorn);
    //       }
    //       else
    //       {
    //          Covise::send_error("ERROR: creation of data object 'dataOut' failed");
    //          return;
    //       }
    // delete polygons_out;
    p_GridOut->setCurrentObject(polygons_out);
    //    }
}

bool IsoPlane::add_vertex(int n1, int n2)
{

    int *targets, *indices; // Pointers into the node_info structure
    float w2, w1;

    targets = node_table[n1].targets;
    indices = node_table[n1].vertice_list;

    while (*targets)
    {
        if (*targets == n2) // did we already calculate this vertex?
        {
            *vertex++ = *indices; // great! just put in the right index.
            return true;
        }

        if (*(targets + 1))
        {
            //fprintf(stderr,"Target_overflow\n");
            break;
        }

        else
        {
            targets++;
            indices++;
        }
    }

    // don't overrun buffers
    if (num_coords == max_coords)
        return false;

    // remember the target we will calculate now
    *targets++ = n2;
    *targets = 0;
    *indices = num_coords;
    *vertex++ = *indices;

    // Calculate the interpolation weights (linear interpolation)
    if (node_table[n1].dist == node_table[n2].dist)
        w2 = 1.0;

    else
    {
        w2 = (float)((double)node_table[n1].dist / (double)(node_table[n1].dist - node_table[n2].dist));
        if (w2 > 1.0)
            w2 = 1.0;
        if (w2 < 0)
            w2 = 0.0;
    }

    w1 = 1.0f - w2;
    *coord_x++ = x_in[n1] * w1 + x_in[n2] * w2;
    *coord_y++ = y_in[n1] * w1 + y_in[n2] * w2;
    *coord_z++ = z_in[n1] * w1 + z_in[n2] * w2;

    if (!_isConnected)
    {
        *S_Data_p++ = _isovalue;
    }

    else if (Datatype)
        *S_Data_p++ = s_in[n1] * w1 + s_in[n2] * w2;

    else
    {
        *V_Data_U_p++ = u_in[n1] * w1 + u_in[n2] * w2;
        *V_Data_V_p++ = v_in[n1] * w1 + v_in[n2] * w2;
        *V_Data_W_p++ = w_in[n1] * w1 + w_in[n2] * w2;
    }

    num_coords++;

    return true;
}

void IsoPlane::add_vertex(int n1, int n2, int x, int y, int z, int u, int v, int w)
{
    int *targets, *indices; // Pointers into the node_info structure
    float w2, w1;

    targets = node_table[n1].targets;
    indices = node_table[n1].vertice_list;

    while (*targets)
    {
        if (*targets == n2) // did we already calculate this vertex?
        {
            *vertex++ = *indices; // great! just put in the right index.
            return;
        }

        if (*(targets + 1))
        {
            //fprintf(stderr,"Target_overflow\n");
            break;
        }

        else
        {
            targets++;
            indices++;
        }
    }

    // remember the target we will calculate now
    *targets++ = n2;
    *targets = 0;
    *indices = num_coords;
    *vertex++ = *indices;

    // Calculate the interpolation weights (linear interpolation)
    if (node_table[n1].dist == node_table[n2].dist)
        w2 = 1.0;

    else
    {
        w2 = (float)((double)node_table[n1].dist / (double)(node_table[n1].dist - node_table[n2].dist));
        if (w2 > 1.0)
            w2 = 1.0;
        if (w2 < 0)
            w2 = 0.0;
    }

    w1 = 1.0f - w2;
    *coord_x++ = x_in[x] * w1 + x_in[u] * w2;
    *coord_y++ = y_in[y] * w1 + y_in[v] * w2;
    *coord_z++ = z_in[z] * w1 + z_in[w] * w2;

    if (!_isConnected)
    {
        *S_Data_p++ = _isovalue;
    }

    else if (Datatype)
        *S_Data_p++ = s_in[n1] * w1 + s_in[n2] * w2;

    else
    {
        *V_Data_U_p++ = u_in[n1] * w1 + u_in[n2] * w2;
        *V_Data_V_p++ = v_in[n1] * w1 + v_in[n2] * w2;
        *V_Data_W_p++ = w_in[n1] * w1 + w_in[n2] * w2;
    }

    num_coords++;
}

void IsoPlane::createNeighbourList()
{
    triPerVertex = maxTriPerVertex;

    // commodity for faster access
    int triPerVertexP1 = triPerVertex + 1;

    //////////// repeat this block until successful

    // no problems yet
    bool neighbourListBad = false;
    neighbors = NULL;

    do
    {
        // old bad run? Clean up and calculate new value
        if (neighbourListBad)
        {
            // cerr << "delete [] neighbors;" << endl;
            delete[] neighbors;
            triPerVertex = (int)((triPerVertex + 1) * 1.5);
            triPerVertexP1 = triPerVertex + 1;
            neighbourListBad = false;
        }
        // cerr << "starting neighbour calc with triPerVertex=" << triPerVertex << endl;
        // cerr << "neighbors = new int["<<num_coords*triPerVertexP1<<"];" << endl;
        neighbors = new int[num_coords * triPerVertexP1];
        memset(neighbors, 0, num_coords * triPerVertexP1 * sizeof(int));

        /// Loop über alle triangles:
        //    i = Index in vertice_list, läuft mit 3-fachem inc
        //    n = Dreiecks-Nummer
        int n, i;
        int *np;

        for (i = 0, n = 0; i < num_vertices; i += 3, n++)
        {
            // find positions of all coords in neighbors field
            // if  over the limit, we cannot not count it
            np = neighbors + triPerVertexP1 * vertice_list[i];
            if ((*np) < triPerVertex)
            {
                (*np)++;
                *(np + (*np)) = n;
            }
            else
            {
                neighbourListBad = true;
                break;
            }

            np = neighbors + triPerVertexP1 * vertice_list[i + 1];
            if ((*np) < triPerVertex)
            {
                (*np)++;
                *(np + (*np)) = n;
            }
            else
            {
                neighbourListBad = true;
                break;
            }

            np = neighbors + triPerVertexP1 * vertice_list[i + 2];
            if ((*np) < triPerVertex)
            {
                (*np)++;
                *(np + (*np)) = n;
            }
            else
            {
                neighbourListBad = true;
                break;
            }
        }
    } while (neighbourListBad);

    if (triPerVertex > 2 * maxTriPerVertex)
        cerr << "IsoSurface: Increase MAX_TRI_PER_VERT to at least "
             << triPerVertex << endl;
}

void IsoPlane::createNormals(int genstrips)
{
    int i, n0, n1, n2, n, *np, *np2;
    float *U, *V, *W, x1, y1, z1, x2, y2, z2;
    float *NU;
    float *NV;
    float *NW;
    float *F_Normals_U;
    float *F_Normals_V;
    float *F_Normals_W;

    NU = Normals_U = new float[num_coords];
    NV = Normals_V = new float[num_coords];
    NW = Normals_W = new float[num_coords];

    U = F_Normals_U = new float[num_vertices];
    V = F_Normals_V = new float[num_vertices];
    W = F_Normals_W = new float[num_vertices];

    createNeighbourList();

    // Loop über alle triangles: Calc Normals of triangles
    //    i = Index in vertice_list, läuft mit 3-fachem inc
    //    n = Dreiecks-Nummer
    for (i = 0; i < num_vertices; i += 3)
    {
        n0 = vertice_list[i]; // Indices der 3 Vertices des aktuellen Dreiecks
        n1 = vertice_list[i + 1];
        n2 = vertice_list[i + 2];

        x1 = coords_x[n1] - coords_x[n0];
        y1 = coords_y[n1] - coords_y[n0];
        z1 = coords_z[n1] - coords_z[n0];
        x2 = coords_x[n2] - coords_x[n0];
        y2 = coords_y[n2] - coords_y[n0];
        z2 = coords_z[n2] - coords_z[n0];

        *U = y1 * z2 - y2 * z1;
        *V = x2 * z1 - x1 * z2;
        *W = x1 * y2 - x2 * y1;

        U++;
        V++;
        W++;
    }

    np = neighbors;
    for (i = 0; i < num_coords; i++)
    {
        np2 = np;
        *NU = *NV = *NW = 0;

        for (n = 0; n < *np; n++)
        {
            np2++;
            *NU += F_Normals_U[*np2];
            *NV += F_Normals_V[*np2];
            *NW += F_Normals_W[*np2];
        }
        np += triPerVertex + 1; // forward to next element in list
        NU++;
        NV++;
        NW++;
    }
    delete[] F_Normals_U;
    delete[] F_Normals_V;
    delete[] F_Normals_W;
    if (!genstrips) // do not delete the neighborlist because we need it for the strips
        delete[] neighbors;
}

void IsoPlane::createStrips(int gennormals)
{
    int i, n0, n1, n2, next_n, j, tn, el = 0, num_try;
    int *np, *ts_vl, *ts_ll, *td, *triangle_done;
    td = triangle_done = new int[num_triangles];
    ts_vl = ts_vertice_list = new int[num_vertices];
    ts_ll = ts_line_list = new int[num_triangles];

    for (i = 0; i < num_triangles; i++)
        td[i] = 0;

    if (!gennormals)
    {
        createNeighbourList();
    }

    int triPerVertexP1 = triPerVertex + 1;

    np = neighbors;
    num_strips = 0;
    el = 0;
    td = triangle_done;
    n0 = 0;
    n1 = 1;
    n2 = 2;
    num_try = 3;
    for (i = 0; i < num_vertices; i += 3)
    {
        if (!(*td)) // Skip Triangle if we already processed it
        {
            // First Triangle of strip
            //printf("%d\n",el);
            *td = 1;
            num_strips++;
            el = 0;
            num_try = 0;
            *ts_ll++ = (int)(ts_vl - ts_vertice_list); // line list points to beginning of strip
            *ts_vl++ = n0 = vertice_list[i]; // first and second vertex of strip
            *ts_vl++ = n1 = vertice_list[i + 1];
            next_n = n2 = vertice_list[i + 2];
            while ((el < 2) && (num_try < 3))
            {
                while (next_n != -1)
                {
                    el++;
                    *ts_vl++ = next_n; // next vertex of Strip
                    n2 = next_n;
                    next_n = -1;
                    // find the next vertex now
                    np = neighbors + triPerVertexP1 * n2; // look for neighbors at point 2
                    for (j = *np; j > 0; j--)
                    {
                        tn = np[j]; // this could be the next triangle
                        if (!triangle_done[np[j]]) // if the neighbortriangle is not already processed
                        {
                            tn *= 3; // tn is now an index to the verice_list
                            if (n2 == vertice_list[tn])
                            {
                                if (n1 == vertice_list[tn + 1])
                                {
                                    next_n = vertice_list[tn + 2];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                                else if (n1 == vertice_list[tn + 2])
                                {
                                    next_n = vertice_list[tn + 1];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                            }
                            else if (n2 == vertice_list[tn + 1])
                            {
                                if (n1 == vertice_list[tn])
                                {
                                    next_n = vertice_list[tn + 2];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                                else if (n1 == vertice_list[tn + 2])
                                {
                                    next_n = vertice_list[tn];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                            }
                            else if (n2 == vertice_list[tn + 2])
                            {
                                if (n1 == vertice_list[tn])
                                {
                                    next_n = vertice_list[tn + 1];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                                else if (n1 == vertice_list[tn + 1])
                                {
                                    next_n = vertice_list[tn];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                            }
                        }
                    }
                }
                num_try++;
                if ((el == 1) && (num_try < 2)) // Try the other two Sides if no neighbor found
                {
                    el = 0;
                    next_n = n0;
                    n0 = n1;
                    n1 = n2;
                    ts_vl--;
                    *(ts_vl - 1) = n1;
                    *(ts_vl - 2) = n0;
                }
            }
        }
        td++;
    }
    delete[] neighbors; // we don´t need it anymore
    delete[] triangle_done;
}

void IsoPlane::createcoDistributedObjects(coOutputPort *p_GridOut, coOutputPort *p_NormalsOut, coOutputPort *p_DataOut, int gennormals, int genstrips, const char *colorn)
{
    float *u_out, *v_out, *w_out;
    int *vl, *pl, i;
    coDoFloat *s_data_out;
    coDoVec3 *v_data_out;
    coDoPolygons *polygons_out;
    coDoTriangleStrips *strips_out;
    coDoVec3 *normals_out;
    const char *DataOut = p_DataOut->getObjName();
    const char *NormalsOut = p_NormalsOut->getObjName();
    const char *GridOut = p_GridOut->getObjName();

    if (Datatype) // (Scalar Data)
    {
        s_data_out = new coDoFloat(DataOut,
                                   num_coords, S_Data);
        if (!s_data_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }

        // delete s_data_out;
        p_DataOut->setCurrentObject(s_data_out);
    }

    else
    {
        v_data_out = new coDoVec3(DataOut,
                                  num_coords,
                                  V_Data_U, V_Data_V, V_Data_W);
        if (!v_data_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
        // delete v_data_out;
        p_DataOut->setCurrentObject(v_data_out);
    }

    num_vertices = (int)(vertex - vertice_list);

    if (gennormals)
    {
        createNormals(genstrips);
        normals_out = new coDoVec3(NormalsOut, num_coords, Normals_U, Normals_V, Normals_W);
        if (!normals_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'normalsOut' failed");
            return;
        }
        // delete normals_out;
        p_NormalsOut->setCurrentObject(normals_out);
        delete[] Normals_U;
        delete[] Normals_V;
        delete[] Normals_W;
    }

    if (genstrips)
    {
        createStrips(gennormals);
        strips_out = new coDoTriangleStrips(GridOut, num_coords, coords_x, coords_y, coords_z, num_triangles + 2 * num_strips, ts_vertice_list, num_strips, ts_line_list);

        if (strips_out->objectOk())
        {
            strips_out->addAttribute("vertexOrder", "2");
            if (colorn[0] != '\0')
                strips_out->addAttribute("COLOR", colorn);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
        // delete strips_out;
        p_GridOut->setCurrentObject(strips_out);
        delete[] ts_vertice_list;
        delete[] ts_line_list;
    }

    else
    {
        polygons_out = new coDoPolygons(GridOut, num_coords, num_vertices, num_triangles);
        if (polygons_out->objectOk())
        {
            polygons_out->getAddresses(&u_out, &v_out, &w_out, &vl, &pl);
            memcpy(u_out, coords_x, num_coords * sizeof(float));
            memcpy(v_out, coords_y, num_coords * sizeof(float));
            memcpy(w_out, coords_z, num_coords * sizeof(float));
            memcpy(vl, vertice_list, num_vertices * sizeof(int));
            for (i = 0; i < num_triangles; i++)
                pl[i] = i * 3;
            polygons_out->addAttribute("vertexOrder", "2");
            if (colorn[0] != '\0')
                polygons_out->addAttribute("COLOR", colorn);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
        // delete polygons_out;
        p_GridOut->setCurrentObject(polygons_out);
    }
}
