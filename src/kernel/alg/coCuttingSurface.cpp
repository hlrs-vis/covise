/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "coCuttingSurface.h"
#include "CuttingTables.h"
#include <do/coDistributedObject.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoPolygons.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

#include <covise/covise.h>
#include <float.h>

#ifdef _WIN32
#include <math.h>
#include <time.h>
#else
#include <sys/time.h>
#endif

using namespace covise;

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

#define ERR1(cond, text, arg1, action) \
    {                                  \
        if (cond)                      \
        {                              \
            sprintf(buf, text, arg1);  \
            Covise::sendError(buf);    \
            {                          \
                action                 \
            }                          \
        }                              \
    }

#define ERR2(cond, text, arg1, arg2, action) \
    {                                        \
        if (cond)                            \
        {                                    \
            sprintf(buf, text, arg1, arg2);  \
            Covise::sendError(buf);          \
            {                                \
                action                       \
            }                                \
        }                                    \
    }

AttributeContainer::AttributeContainer(const coDistributedObject *p_obj)
{
    p_obj_ = p_obj;
    if (p_obj)
        no_attrs_ = p_obj->getAllAttributes(&attr_, &setting_);
    else
        no_attrs_ = 0;
}

const coDistributedObject *AttributeContainer::getPObj()
{
    return p_obj_;
}

void AttributeContainer::addAttributes(coDistributedObject *p_obj)
{
    if (no_attrs_)
        p_obj->addAttributes(no_attrs_, attr_, setting_);
}

void AttributeContainer::addAttributes(coDistributedObject *p_obj, const char *probeAttr)
{
    if (no_attrs_ && p_obj_->getAttribute("Probe2D") == NULL)
    {
        p_obj->addAttributes(no_attrs_, attr_, setting_);
    }
    else
    {
        for (int i = 0; i < no_attrs_; i++)
        {
            if (strcmp(attr_[i], "Probe2D") != 0)
            {
                p_obj->addAttribute(attr_[i], setting_[i]);
            }
        }
    }
    p_obj->addAttribute("Probe2D", probeAttr);
}

//========================= Plane ====================================

Plane::Plane()
{
    initialize();
}

Plane::Plane(int n_elem, int n_nodes, int Type, int *p_el,
             int *p_cl, int *p_tl, float *p_x_in, float *p_y_in, float *p_z_in,
             float *p_s_in, unsigned char *p_bs_in, float *p_i_in,
             float *p_u_in, float *p_v_in, float *p_w_in,
             const coDoStructuredGrid *p_sgrid_in,
             const coDoUnstructuredGrid *p_grid_in,
             float vertexRatio, int maxPoly,
             float planei_, float planej_, float planek_, float startx_,
             float starty_, float startz_, float myDistance_, float radius_,
             int gennormals_, int option_, int genstrips_, char *ib)
    : planei(planei_)
    , planej(planej_)
    , planek(planek_)
    , startx(startx_)
    , starty(starty_)
    , startz(startz_)
    , myDistance(myDistance_)
    , radius(radius_)
    , gennormals(gennormals_)
    , option(option_)
    , genstrips(genstrips_)
{
    initialize();

    unstr_ = true;
    maxPolyPerVertex = maxPoly;
    NodeInfo *node;
    int i;
    iblank = ib;
    el = p_el;
    cl = p_cl;
    tl = p_tl;
    x_in = p_x_in;
    y_in = p_y_in;
    z_in = p_z_in;
    s_in = p_s_in;
    bs_in = p_bs_in;
    i_in = p_i_in;
    u_in = p_u_in;
    v_in = p_v_in;
    w_in = p_w_in;
    sgrid_in = p_sgrid_in;
    ugrid_in = NULL;
    rgrid_in = NULL;
    grid_in = p_grid_in;
    Datatype = Type;
    num_nodes = n_nodes;
    num_elem = n_elem;
    //    node_table   = (NodeInfo *)malloc(n_nodes*sizeof(NodeInfo));
    node_table = new NodeInfo[num_nodes];
    node = node_table;
    float tmpi, tmpj, tmpk;
    cur_line_elem = 0;
    if (option == 0)
    {
        for (i = 0; i < n_nodes; i++)
        {
            node->targets[0] = 0;
            // Calculate the myDistance of each node
            // to the Cuttingplane
            node->dist = (planei * x_in[i] + planej * y_in[i] + planek * z_in[i] - myDistance);
            node->side = (node->dist >= 0 ? 1 : 0);
            node++;
        }
    }
    else if (option == 1) //sphere
    {
        for (i = 0; i < n_nodes; i++)
        {
            node->targets[0] = 0; // Calculate the myDistance of each node
            // to the Cuttingsphere
            tmpi = planei - x_in[i];
            tmpj = planej - y_in[i];
            tmpk = planek - z_in[i];
            node->dist = sqrt(tmpi * tmpi + tmpj * tmpj + tmpk * tmpk) - radius;
            node->side = (node->dist >= 0 ? 1 : 0);
            node++;
        }
    }
    else if (option == 2) //cylinder-X
    {
        for (i = 0; i < n_nodes; i++)
        {
            node->targets[0] = 0; // Calculate the myDistance of each node
            // to the Cuttingsphere
            tmpj = planej - y_in[i]; // start <-> plane
            tmpk = planek - z_in[i];
            node->dist = sqrt(tmpk * tmpk + tmpj * tmpj) - radius;
            node->side = (node->dist >= 0 ? 1 : 0);
            node++;
        }
    }
    else if (option == 3) //cylinder-Y
    {
        for (i = 0; i < n_nodes; i++)
        {
            node->targets[0] = 0; // Calculate the myDistance of each node
            // to the Cuttingsphere
            tmpi = planei - x_in[i]; // start <-> plane
            tmpk = planek - z_in[i];
            node->dist = sqrt(tmpi * tmpi + tmpk * tmpk) - radius;
            node->side = (node->dist >= 0 ? 1 : 0);
            node++;
        }
    }
    else if (option == 4) //cylinder-Z
    {
        for (i = 0; i < n_nodes; i++)
        {
            node->targets[0] = 0; // Calcuulate the myDistance of each node
            // to the Cuttingsphere
            tmpi = planei - x_in[i]; // start <-> plane
            tmpj = planej - y_in[i];
            node->dist = sqrt(tmpi * tmpi + tmpj * tmpj) - radius;
            node->side = (node->dist >= 0 ? 1 : 0);
            node++;
        }
    }
    num_triangles = num_vertices = num_coords = 0;
#ifdef DEBUGMEM
    fprintf(stderr, "CPUsg: Line: %d new vertice_list[%d] returned %d\n", __LINE__, n_elem * 12, vertice_list);
#endif

    // new-style alloc only for USG so far
    if (vertexRatio > 0)
    {
        max_coords = (int)(pow((float)n_nodes, (float)0.666666666) * vertexRatio);
        vertice_list = new int[max_coords * 6];
    }
    else
    {
        max_coords = num_nodes / 1 /* 6 */; //@@@
        vertice_list = new int[num_elem * 18];
    }

    vertex = vertice_list;

    coords_x = new float[max_coords];
    coords_y = new float[max_coords];
    coords_z = new float[max_coords];
    coord_x = coords_x;
    coord_y = coords_y;
    coord_z = coords_z;

    if ((Datatype == 1) || (Datatype == 2)) // (1: scalar data, 2: scalar and vector data)
    {
        S_Data_p = S_Data = new float[max_coords];
    }
    if ((Datatype == 0) || (Datatype == 2)) // (0: vector data, 2: scalar and vector data)
    {
        V_Data_U_p = V_Data_U = new float[max_coords];
        V_Data_V_p = V_Data_V = new float[max_coords];
        V_Data_W_p = V_Data_W = new float[max_coords];
    }
    if (i_in)
    {
        I_Data_p = I_Data = new float[max_coords];
    }
    else
        I_Data = S_Data;
}

void Plane::initialize()
{
    grid_in = NULL;
    sgrid_in = NULL;
    ugrid_in = NULL;
    rgrid_in = NULL;

    polygons_out = NULL;
    strips_out = NULL;
    normals_out = NULL;
    s_data_out = NULL;
    v_data_out = NULL;

    el = NULL;
    cl = NULL;
    tl = NULL;
    x_in = NULL;
    y_in = NULL;
    z_in = NULL;
    s_in = NULL;
    bs_in = NULL;
    i_in = NULL;
    u_in = NULL;
    v_in = NULL;
    w_in = NULL;
    vertice_list = NULL;
    ts_vertice_list = NULL;
    ts_line_list = NULL;
    vertex = NULL;

    coords_x = NULL;
    coords_y = NULL;
    coords_z = NULL;
    coord_x = NULL;
    coord_y = NULL;
    coord_z = NULL;
    V_Data_U = NULL;
    V_Data_V = NULL;
    V_Data_W = NULL;
    S_Data = NULL;
    I_Data = NULL;
    V_Data_U_p = NULL;
    V_Data_V_p = NULL;
    V_Data_W_p = NULL;
    S_Data_p = NULL;
    I_Data_p = NULL;
    node_table = NULL;
    iblank = NULL;
}

Plane::~Plane()
{
    if (ugrid_in != NULL)
    {
        delete[] x_in;
        delete[] y_in;
        delete[] z_in;
    }
    delete[] node_table;
    delete[] vertice_list;
    delete[] coords_x;
    delete[] coords_y;
    delete[] coords_z;
    if (S_Data)
    {
        delete[] S_Data;
        S_Data = NULL;
    }
    if (i_in)
    {
        delete[] I_Data;
        I_Data = NULL;
    }
    if (V_Data_U)
    {
        delete[] V_Data_U;
        delete[] V_Data_V;
        delete[] V_Data_W;
        V_Data_U = NULL;
        V_Data_V = NULL;
        V_Data_W = NULL;
    }
}

bool Plane::createPlane()
{
    // 1 = above; 0 = below
    for (int element = 0; element < num_elem; element++)
    {
        if (iblank == NULL || iblank[element] != '\0')
        {
            int elementtype = tl[element];
            int bitmap = 0; // index in the MarchingCubes table
            int i = UnstructuredGrid_Num_Nodes[elementtype];
            // number of nodes for current element
            int *node_list = cl + el[element];
            // pointer to nodes of current element
            int *node = node_list + i;
            // node = pointer to last node of current element
            while (i--)
                bitmap |= node_table[*--node].side << i;
            // bitmap is now an index to the Cuttingtable
            if (Cutting_Info[elementtype])
            {
                cutting_info *C_Info = Cutting_Info[elementtype] + bitmap;
                int numIntersections = C_Info->nvert;
                if (numIntersections)
                {
                    int *polygon_nodes = C_Info->node_pairs;
                    num_triangles += numIntersections - 2;
                    int *firstvertex = vertex;
                    for (i = 0; i < numIntersections; i++)
                    {
                        int n1 = node_list[*polygon_nodes++];
                        int n2 = node_list[*polygon_nodes++];
                        if (i > 2)
                        {
                            *vertex++ = *firstvertex;
                            *vertex = *(vertex - 2);
                            vertex++;
                        }
                        if (n1 < n2)
                        {
                            if (!add_vertex(n1, n2))
                                return false;
                        }
                        else
                        {
                            if (!add_vertex(n2, n1))
                                return false;
                        }
                    }
                }
            }
        }
    }
    if (getenv("CUTTINGSURFACE_STATISTICS"))
    {
        Covise::sendInfo("Used %d of %d vertices: Usage=%f%%",
                         num_coords, max_coords, ((float)num_coords) / max_coords);
    }
    return true;
}

// return false if  no  space left
bool Plane::add_vertex(int n1, int n2)
{

    int *targets, *indices; // Pointers into the node_info structure
    float w2, w1;

    targets = node_table[n1].targets;
    indices = node_table[n1].vertice_list;

    int n = 0;
    while ((*targets) && (n < 11))
    {
        if (*targets == n2) // did we already calculate this vertex?
        {
            *vertex++ = *indices; // great! just put in the right index.
            return true;
        }
        targets++;
        indices++;
        n++;
    }

    if (num_coords == max_coords)
        return false;

    // remember the target we will calculate now

    *targets++ = n2;
    if (n < 11)
        *targets = 0;
    *indices = num_coords;
    *vertex++ = *indices;

    // Calculate the interpolation weights (linear interpolation)

    w2 = node_table[n1].dist / (node_table[n1].dist - node_table[n2].dist);
    w1 = 1.0f - w2;
    *coord_x++ = x_in[n1] * w1 + x_in[n2] * w2;
    *coord_y++ = y_in[n1] * w1 + y_in[n2] * w2;
    *coord_z++ = z_in[n1] * w1 + z_in[n2] * w2;
    if (i_in)
        *I_Data_p++ = i_in[n1] * w1 + i_in[n2] * w2;
    if ((Datatype == 1) || (Datatype == 2))
    {
        if (bs_in)
            *S_Data_p++ = bs_in[n1] / 255.f * w1 + bs_in[n2] / 255.f * w2;
        else
            *S_Data_p++ = s_in[n1] * w1 + s_in[n2] * w2;
    }
    if ((Datatype == 0) || (Datatype == 2))
    {
        *V_Data_U_p++ = u_in[n1] * w1 + u_in[n2] * w2;
        *V_Data_V_p++ = v_in[n1] * w1 + v_in[n2] * w2;
        *V_Data_W_p++ = w_in[n1] * w1 + w_in[n2] * w2;
    }
    num_coords++;

    return true;
}

void Plane::add_vertex(int n1, int n2, int x, int y, int z, int u, int v, int w)
{

    int *targets, *indices; // Pointers into the node_info structure
    float w2, w1;

    targets = node_table[n1].targets;
    indices = node_table[n1].vertice_list;

    int n = 0;
    while ((*targets) && (n < 11))
    {
        if (*targets == n2) // did we already calculate this vertex?
        {
            *vertex++ = *indices; // great! just put in the right index.
            return;
        }
        targets++;
        indices++;
        n++;
    }

    // remember the target we will calculate now

    *targets++ = n2;
    if (n < 11)
        *targets = 0;
    *indices = num_coords;
    *vertex++ = *indices;

    // Calculate the interpolation weights (linear interpolation)

    w2 = node_table[n1].dist / (node_table[n1].dist - node_table[n2].dist);
    w1 = 1.0f - w2;
    *coord_x++ = x_in[x] * w1 + x_in[u] * w2;
    *coord_y++ = y_in[y] * w1 + y_in[v] * w2;
    *coord_z++ = z_in[z] * w1 + z_in[w] * w2;
    if (i_in)
        *I_Data_p++ = i_in[n1] * w1 + i_in[n2] * w2;
    if ((Datatype == 1) || (Datatype == 2))
    {
        if (bs_in)
            *S_Data_p++ = bs_in[n1] / 255.f * w1 + bs_in[n2] / 255.f * w2;
        else
            *S_Data_p++ = s_in[n1] * w1 + s_in[n2] * w2;
    }
    if ((Datatype == 0) || (Datatype == 2))
    {
        *V_Data_U_p++ = u_in[n1] * w1 + u_in[n2] * w2;
        *V_Data_V_p++ = v_in[n1] * w1 + v_in[n2] * w2;
        *V_Data_W_p++ = w_in[n1] * w1 + w_in[n2] * w2;
    }
    num_coords++;
}

void Plane::border_proj(float *px, float *py, float *pz,
                        float xminb,
                        float xmaxb,
                        float yminb,
                        float ymaxb,
                        float zminb,
                        float zmaxb,
                        float pli, float plj, float plk, float distance)
{
    float x[8], y[8], z[8]; // border box points
    x[0] = xminb;
    y[0] = yminb;
    z[0] = zminb;
    x[1] = xmaxb;
    y[1] = yminb;
    z[1] = zminb;
    x[2] = xmaxb;
    y[2] = yminb;
    z[2] = zmaxb;
    x[3] = xminb;
    y[3] = yminb;
    z[3] = zmaxb;
    x[4] = xminb;
    y[4] = ymaxb;
    z[4] = zminb;
    x[5] = xmaxb;
    y[5] = ymaxb;
    z[5] = zminb;
    x[6] = xmaxb;
    y[6] = ymaxb;
    z[6] = zmaxb;
    x[7] = xminb;
    y[7] = ymaxb;
    z[7] = zmaxb;
    float norm2 = pli * pli + plj * plj + plk * plk;
    float norm = sqrt(norm2);
    if (norm2 != 0.0)
        norm2 = 1.0f / norm2;
    int i;
    for (i = 0; i < 8; i++)
    {
        // X projections
        px[i] = x[i] - (pli * x[i] + plj * y[i] + plk * z[i] - distance * norm) * pli * norm2;
        // Y projections
        py[i] = y[i] - (pli * x[i] + plj * y[i] + plk * z[i] - distance * norm) * plj * norm2;
        // Z projections
        pz[i] = z[i] - (pli * x[i] + plj * y[i] + plk * z[i] - distance * norm) * plk * norm2;
    }
}

float Plane::gsin(float angle) { return ((float)(sin((angle * M_PI) / 180.0))); }
float Plane::gcos(float angle) { return ((float)(cos((angle * M_PI) / 180.0))); }

void
Plane::buildSphere(float *xSphere, float *ySphere, float *zSphere,
                   float xPoint, float yPoint, float zPoint, float rad)
{

    int b;

    // coordinates
    for (b = 0; b < 5; b++)
    {
        xSphere[b] = (float)(xPoint + gcos((float)b * 72));
        ySphere[b] = (float)(yPoint + gsin((float)b * 72));
        zSphere[b] = zPoint;

        xSphere[b + 5] = (float)(xPoint + gcos((float)b * 72 + 36) * gsin(45.0f));
        ySphere[b + 5] = (float)(yPoint + gsin((float)b * 72 + 36) * gsin(45.0f));
        zSphere[b + 5] = (float)(zPoint + gsin(45.0f));

        xSphere[b + 10] = (float)(xPoint + gcos((float)b * 72 + 36) * gsin(45.0f));
        ySphere[b + 10] = (float)(yPoint + gsin((float)b * 72 + 36) * gsin(45.0f));
        zSphere[b + 10] = (float)(zPoint - gsin(45.0f));
    }

    xSphere[15] = xPoint;
    ySphere[15] = yPoint;
    zSphere[15] = zPoint + 1.0f;

    xSphere[16] = xPoint;
    ySphere[16] = yPoint;
    zSphere[16] = zPoint - 1.0f;

    int i;
    for (i = 0; i < 17; i++)
    {
        xSphere[i] = xPoint - (xPoint - xSphere[i]) * rad;
        ySphere[i] = yPoint - (yPoint - ySphere[i]) * rad;
        zSphere[i] = zPoint - (zPoint - zSphere[i]) * rad;
    }
}

void Plane::build_SphereStrips(int *tsl, int *vl)
{
    // triangle strips
    tsl[0] = 0;
    tsl[1] = 12;
    tsl[2] = 12 + 12;
    tsl[3] = 12 + 12 + 5;
    tsl[4] = 12 + 12 + 5 + 4;
    tsl[5] = 12 + 12 + 5 + 4 + 5;

    // vertex list
    vl[0] = 5;
    vl[1] = 1;
    vl[2] = 6;
    vl[3] = 2;
    vl[4] = 7;
    vl[5] = 3;
    vl[6] = 8;
    vl[7] = 4;
    vl[8] = 9;
    vl[9] = 0;
    vl[10] = 5;
    vl[11] = 1;

    vl[12] = 0;
    vl[13] = 10;
    vl[14] = 1;
    vl[15] = 11;
    vl[16] = 2;
    vl[17] = 12;
    vl[18] = 3;
    vl[19] = 13;
    vl[20] = 4;
    vl[21] = 14;
    vl[22] = 0;
    vl[23] = 10;

    vl[24] = 9;
    vl[25] = 5;
    vl[26] = 15;
    vl[27] = 6;
    vl[28] = 7;

    vl[29] = 9;
    vl[30] = 15;
    vl[31] = 8;
    vl[32] = 7;

    vl[33] = 12;
    vl[34] = 11;
    vl[35] = 16;
    vl[36] = 10;
    vl[37] = 14;

    vl[38] = 12;
    vl[39] = 16;
    vl[40] = 13;
    vl[41] = 14;
}

// sl:
// Given 8 points on the cutting plane, calculate 4 points on that plane
// such that the inertia is preserved...
void Plane::preserve_inertia(const float *x8, const float *y8, const float *z8,
                             float *x4, float *y4, float *z4,
                             float pli, float plj, float plk)
{
    float cg[3];
    int i;

    // first calculate center of gravity
    for (i = 0, cg[0] = 0.0, cg[1] = 0.0, cg[2] = 0.0; i < 8; ++i)
    {
        cg[0] += x8[i];
        cg[1] += y8[i];
        cg[2] += z8[i];
    }
    cg[0] /= 8.0;
    cg[1] /= 8.0;
    cg[2] /= 8.0;

    // Change to a coordinate system in which
    // the CG is 0 and the plane lies on the XY plane
    float rot[3][3];
    float length = sqrt(pli * pli + plj * plj + plk * plk);
    if (length == 0.0)
    {
        pli = 1.0;
        plj = 0.0;
        plk = 0.0;
    }
    pli /= length;
    plj /= length;
    plk /= length;

    // the new k vector
    rot[0][2] = pli;
    rot[1][2] = plj;
    rot[2][2] = plk;

    // the new j vector
    if (fabs(pli) <= fabs(plj) && fabs(pli) <= fabs(plk))
    {
        rot[0][1] = 0.0;
        rot[1][1] = -plk;
        rot[2][1] = plj;
        float norm = sqrt(plk * plk + plj * plj);
        rot[1][1] /= norm;
        rot[2][1] /= norm;
    }
    else if (fabs(plj) <= fabs(pli) && fabs(plj) <= fabs(plk))
    {
        rot[0][1] = -plk;
        rot[1][1] = 0.0;
        rot[2][1] = pli;
        float norm = sqrt(pli * pli + plk * plk);
        rot[0][1] /= norm;
        rot[2][1] /= norm;
    }
    else
    {
        rot[0][1] = -plj;
        rot[1][1] = pli;
        rot[2][1] = 0.0;
        float norm = sqrt(pli * pli + plj * plj);
        rot[0][1] /= norm;
        rot[1][1] /= norm;
    }

    // the new i vector
    rot[0][0] = rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1];
    rot[1][0] = rot[2][1] * rot[0][2] - rot[2][2] * rot[0][1];
    rot[2][0] = rot[0][1] * rot[1][2] - rot[0][2] * rot[1][1];

    // First we transform input points on the shadow to the new coord. system
    float x_new[8];
    float y_new[8];
    // multiply by transposed matrix
    for (i = 0; i < 8; i++)
    {
        x_new[i] = rot[0][0] * (x8[i] - cg[0]) + rot[1][0] * (y8[i] - cg[1]) + rot[2][0] * (z8[i] - cg[2]);
        y_new[i] = rot[0][1] * (x8[i] - cg[0]) + rot[1][1] * (y8[i] - cg[1]) + rot[2][1] * (z8[i] - cg[2]);
    }

    // Consider the tensor J_{i,j}=Sum_{particles} mass_{particle} * x_i * x_j
    // this is not the usual inertia tensor, but it is good enough for
    // our purposes
    float J[2][2] = {
        { 0.0, 0.0 },
        { 0.0, 0.0 }
    };

    for (i = 0; i < 8; i++)
    {
        J[0][0] += x_new[i] * x_new[i];
        J[1][0] += x_new[i] * y_new[i];
        J[1][1] += y_new[i] * y_new[i];
    }
    J[0][1] = J[1][0];

    // Now calculate the angle for a new coordinate change given
    // by a rotation about the normal to the plane, so that
    // J is diagonal...
    float angle;
    float x2, y2; // half of the rectangle sides
    float trace, det; // invariants
    trace = J[0][0] + J[1][1];
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    float disc = trace * trace - 32.0f * det;
    if (disc < 0.0)
        disc = 0.0; // impossible with infinite precision
    x2 = (float)(sqrt(trace + sqrt(disc)) / 4.0f);
    y2 = (float)(sqrt(trace - sqrt(disc)) / 4.0f);

    if (fabs(J[0][0] - J[1][1]) <= 1e-6 * fabs(J[0][1]))
    {
        angle = (float)(M_PI / 4);
    }
    else
    {
        angle = (float)(0.5f * atan(2.0f * J[0][1] / (J[0][0] - J[1][1])));
    }
    // We may be wrong by PI/2...
    float cosinus, sinus;
    cosinus = cos(angle);
    sinus = sin(angle);
    if (J[0][0] * cosinus * cosinus + J[1][1] * sinus * sinus + 2 * J[1][0] * sinus * cosinus
        < J[0][0] * sinus * sinus + J[1][1] * cosinus * cosinus - 2 * J[1][0] * sinus * cosinus)
    {
        float tmp = cosinus;
        angle += (float)(M_PI * 0.5f);
        cosinus = -sinus;
        sinus = tmp;
    }

    // the four points in the rotated coord. sys in which J is diagonal...
    float x4d[4], y4d[4];
    x4d[0] = x2;
    x4d[1] = -x2;
    x4d[2] = -x2;
    x4d[3] = x2;
    y4d[0] = y2;
    y4d[1] = y2;
    y4d[2] = -y2;
    y4d[3] = -y2;
    // the four points in the rotated coord. sys in which J is not diagonal (in general)...
    float x4n[4], y4n[4];
    for (i = 0; i < 4; ++i)
    {
        x4n[i] = cosinus * x4d[i] - sinus * y4d[i];
        y4n[i] = sinus * x4d[i] + cosinus * y4d[i];
    }
    // now we transform to the global coordinate system
    for (i = 0; i < 4; ++i)
    {
        x4[i] = rot[0][0] * x4n[i] + rot[0][1] * y4n[i] + cg[0];
        y4[i] = rot[1][0] * x4n[i] + rot[1][1] * y4n[i] + cg[1];
        z4[i] = rot[2][0] * x4n[i] + rot[2][1] * y4n[i] + cg[2];
    }
}

// u_out, v_out, w_out are loaded with the normals
// coords_? are the point coordinates
void Plane::fill_normals(float *u_out, float *v_out, float *w_out,
                         const float *coords_x, const float *coords_y, const float *coords_z,
                         int nb_coords, int param_option,
                         float pla[3], float rad, float start[3])
{
    (void)start;
    int i;
    float radius = rad;
    float planei = pla[0];
    float planej = pla[1];
    float planek = pla[2];
    /*
     float startx = start[0];
     float starty = start[1];
     float startz = start[2];
   */

    if (param_option == 0)
    {
        for (i = 0; i < nb_coords; i++)
        {
            *u_out++ = -planei;
            *v_out++ = -planej;
            *w_out++ = -planek;
        }
    }
    else if (param_option == 1)
    {
        for (i = 0; i < nb_coords; i++)
        {
            *u_out++ = (planei - coords_x[i]) / radius;
            *v_out++ = (planej - coords_y[i]) / radius;
            *w_out++ = (planek - coords_z[i]) / radius;
        }
    }
    else if (param_option == 2)
    {
        for (i = 0; i < nb_coords; i++)
        {
            *u_out++ = 0.0;
            *v_out++ = (planej - coords_y[i]) / radius; // start <-> plane
            *w_out++ = (planek - coords_z[i]) / radius;
        }
    }
    else if (param_option == 3)
    {
        for (i = 0; i < nb_coords; i++)
        {
            *u_out++ = (planei - coords_x[i]) / radius;
            *v_out++ = 0.0;
            *w_out++ = (planek - coords_z[i]) / radius;
        }
    }
    else if (param_option == 4)
    {
        for (i = 0; i < nb_coords; i++)
        {
            *u_out++ = (planei - coords_x[i]) / radius;
            *v_out++ = (planej - coords_y[i]) / radius;
            *w_out++ = 0.0;
        }
    }
}

// x4, y4, z4 keep the coordinates of three points
// and we check their orientation with respect to a
// vector given by planei planej planek
int Plane::check_orientation(float *x4, float *y4, float *z4,
                             float planei, float planej, float planek)
{
    float v0[3];
    float v1[3];
    v0[0] = x4[1] - x4[0];
    v1[0] = x4[2] - x4[1];
    v0[1] = y4[1] - y4[0];
    v1[1] = y4[2] - y4[1];
    v0[2] = z4[1] - z4[0];
    v1[2] = z4[2] - z4[1];

    float v2[3]; // v0xv1
    v2[0] = v0[1] * v1[2] - v0[2] * v1[1];
    v2[1] = v0[2] * v1[0] - v0[0] * v1[2];
    v2[2] = v0[0] * v1[1] - v0[1] * v1[0];

    return (planei * v2[0] + planej * v2[1] + planek * v2[2] > 0);
}

coDoTriangleStrips *Plane::dummy_tr_strips(const char *tr_name,
                                           float xminb,
                                           float xmaxb,
                                           float yminb,
                                           float ymaxb,
                                           float zminb,
                                           float zmaxb, int param_option,
                                           float pli, float plj, float plk, float distance,
                                           float strtx, float strty, float strtz)
{
    coDoTriangleStrips *dummy;
    int *cor_list;
    int *pol_list;
    float angle, w, rad = distance;
    float *x, *y, *z;
    int i, n;

    n = CREFN; // the degree of refinement for cylinder surfaces
    switch (param_option)
    {
    case 0:
    { //Plane
        float xp[8], yp[8], zp[8]; // projection of border box points
        // on cutting surface
        float x4[8], y4[8], z4[8]; // sl: the shadow will be a rectangle
        //     in order to set up a feasible
        //     algorithm without overlapping
        //     polygons
        border_proj(xp, yp, zp, xminb, xmaxb, yminb, ymaxb, zminb, zmaxb,
                    pli, plj, plk, distance);
        preserve_inertia(xp, yp, zp, x4, y4, z4, pli, plj, plk);

/*
                      cor_list = new int[4];
                      if(check_orientation(x4,y4,z4,pli,plj,plk))
                      {
                         cor_list[0] = 0;
                         cor_list[1] = 1;
                         cor_list[2] = 3;
                         cor_list[3] = 2;
                      }
                      else
                      {
         cor_list[0] = 3;
         cor_list[1] = 2;
         cor_list[2] = 0;
         cor_list[3] = 1;
         }
         */
// Frame instead of rectangle - calc new points
#define FR(x, y) ((x)*0.95f + (y)*0.05f)

        x4[4] = FR(x4[0], x4[2]);
        x4[5] = FR(x4[1], x4[3]);
        x4[6] = FR(x4[2], x4[0]);
        x4[7] = FR(x4[3], x4[1]);

        y4[4] = FR(y4[0], y4[2]);
        y4[5] = FR(y4[1], y4[3]);
        y4[6] = FR(y4[2], y4[0]);
        y4[7] = FR(y4[3], y4[1]);

        z4[4] = FR(z4[0], z4[2]);
        z4[5] = FR(z4[1], z4[3]);
        z4[6] = FR(z4[2], z4[0]);
        z4[7] = FR(z4[3], z4[1]);
#undef FR

        pol_list = new int[1];

        int *cor_list;
        int right[] = { 1, 5, 0, 4, 3, 7, 2, 6, 1, 5 };
        int left[] = { 5, 1, 6, 2, 7, 3, 4, 0, 5, 1 };
        if (check_orientation(x4, y4, z4, pli, plj, plk))
            cor_list = left;
        else
            cor_list = right;

        pol_list[0] = 0;
        dummy = new coDoTriangleStrips(tr_name, 8, x4, y4, z4, 10, cor_list, 1, pol_list);
        //delete [] cor_list;
        delete[] pol_list;
        break;
    }
    case 1: // Sphere
        float *xSphere, *ySphere, *zSphere;
        int *vl, *tsl;
        dummy = new coDoTriangleStrips(tr_name, 17, 42, 6);
        dummy->getAddresses(&xSphere, &ySphere, &zSphere, &vl, &tsl);
        buildSphere(xSphere, ySphere, zSphere, pli, plj, plk, rad);
        build_SphereStrips(tsl, vl);
        break;
    case 2: // X - Cylinder
        // build points
        x = new float[2 * n];
        y = new float[2 * n];
        z = new float[2 * n];
        cor_list = new int[2 * n + 4];
        pol_list = new int[2];
        rad = sqrt((plj - strty) * (plj - strty) + (plk - strtz) * (plk - strtz));
        w = (float)(2.0f * M_PI / ((float)(n)));
        for (i = 0; i < n; i++)
        {
            angle = w * ((float)i);

            x[i] = xminb;
            y[i] = plj + sin(angle) * rad; // strt <-> pl
            z[i] = plk + cos(angle) * rad;

            x[i + n] = xmaxb;
            y[i + n] = y[i];
            z[i + n] = z[i];
        }
        for (i = 0; i < n; i++)
        {
            cor_list[2 * i] = i;
            cor_list[2 * i + 1] = i + n;
        }
        cor_list[2 * n] = n - 1;
        cor_list[2 * n + 1] = 2 * n - 1;
        cor_list[2 * n + 2] = 0;
        cor_list[2 * n + 3] = n;
        pol_list[0] = 0;
        pol_list[1] = 2 * n;
        dummy = new coDoTriangleStrips(tr_name, 2 * n, x, y, z, 2 * n + 4, cor_list, 2, pol_list);
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] cor_list;
        delete[] pol_list;
        break;
    case 3: // Y - Cylinder
        // build points
        x = new float[2 * n];
        y = new float[2 * n];
        z = new float[2 * n];
        cor_list = new int[2 * n + 4];
        pol_list = new int[2];
        rad = sqrt((pli - strtx) * (pli - strtx) + (plk - strtz) * (plk - strtz));
        w = (float)(2.0 * M_PI) / ((float)(n));
        for (i = 0; i < n; i++)
        {
            angle = w * ((float)i);

            x[i] = pli + sin(angle) * rad; // strt <-> pl
            y[i] = ymaxb;
            z[i] = plk + cos(angle) * rad;

            x[i + n] = x[i];
            y[i + n] = yminb;
            z[i + n] = z[i];
        }
        for (i = 0; i < n; i++)
        {
            cor_list[2 * i] = i;
            cor_list[2 * i + 1] = i + n;
        }
        cor_list[2 * n] = n - 1;
        cor_list[2 * n + 1] = 2 * n - 1;
        cor_list[2 * n + 2] = 0;
        cor_list[2 * n + 3] = n;
        pol_list[0] = 0;
        pol_list[1] = 2 * n;
        dummy = new coDoTriangleStrips(tr_name, 2 * n, x, y, z, 2 * n + 4, cor_list, 2, pol_list);
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] cor_list;
        delete[] pol_list;
        break;
    case 4: // Z - Cylinder
        x = new float[2 * n];
        y = new float[2 * n];
        z = new float[2 * n];
        // cor_list = new int[5*n];
        cor_list = new int[2 * n + 4];
        pol_list = new int[2];
        rad = sqrt((plj - strty) * (plj - strty) + (pli - strtx) * (pli - strtx));
        w = (float)(2.0f * M_PI) / ((float)(n));
        for (i = 0; i < n; i++)
        {
            angle = w * ((float)i);

            x[i] = pli + sin(angle) * rad; // strt <-> pl
            y[i] = plj + cos(angle) * rad;
            z[i] = zminb;

            x[i + n] = x[i];
            y[i + n] = y[i];
            z[i + n] = zmaxb;
        }
        // for(i=0; i<n-1; i++ )
        for (i = 0; i < n; i++)
        {
            cor_list[2 * i] = i;
            cor_list[2 * i + 1] = i + n;
        }
        cor_list[2 * n] = n - 1;
        cor_list[2 * n + 1] = 2 * n - 1;
        cor_list[2 * n + 2] = 0;
        cor_list[2 * n + 3] = n;

        pol_list[0] = 0;
        pol_list[1] = 2 * n;
        dummy = new coDoTriangleStrips(tr_name, 2 * n, x, y, z, 2 * n + 4, cor_list, 2, pol_list);
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] cor_list;
        delete[] pol_list;
        break;
    default:
        dummy = new coDoTriangleStrips(tr_name, 0, 0, 0);
        Covise::sendError("ERROR: invalid option choice");
        //  Covise::send_stop_pipeline();
    }
    return dummy;
}

/******************************************************************
             nb_pol = trs2pol(42,6,vl,tsl,v_list,pl_list);
             dummy = new coDoPolygons(tr_name,17,xS,yS,zS,3*nb_pol,v_list,nb_pol,pl_list);
*******************************************************************/
int Plane::trs2pol(int nb_con, int nb_tr, int *trv, int *tr_list, int *plv, int *pol_list)
{
    int i, j, nb_pol, nb_v, crt_con;

    nb_pol = 0;
    crt_con = 0;
    for (i = 0; i < nb_tr - 1; i++)
    {
        crt_con = tr_list[i];
        nb_v = tr_list[i + 1] - tr_list[i];
        for (j = 0; j < nb_v - 2; j++)
        {
            plv[3 * nb_pol] = trv[crt_con + j];
            if (j % 2 == 0)
            {
                plv[3 * nb_pol + 1] = trv[crt_con + j + 1];
                plv[3 * nb_pol + 2] = trv[crt_con + j + 2];
            }
            else
            {
                plv[3 * nb_pol + 1] = trv[crt_con + j + 2];
                plv[3 * nb_pol + 2] = trv[crt_con + j + 1];
            }
            pol_list[nb_pol] = 3 * nb_pol;
            nb_pol++;
        }
    }
    crt_con = tr_list[nb_tr - 1];
    nb_v = nb_con - tr_list[nb_tr - 1];
    for (j = 0; j < nb_v - 2; j++)
    {
        plv[3 * nb_pol] = trv[crt_con + j];
        if (j % 2 == 0)
        {
            plv[3 * nb_pol + 1] = trv[crt_con + j + 1];
            plv[3 * nb_pol + 2] = trv[crt_con + j + 2];
        }
        else
        {
            plv[3 * nb_pol + 1] = trv[crt_con + j + 2];
            plv[3 * nb_pol + 2] = trv[crt_con + j + 1];
        }
        pol_list[nb_pol] = 3 * nb_pol;
        nb_pol++;
    }

    return nb_pol;
}

coDoPolygons *
Plane::dummy_polygons(const char *tr_name,
                      float xminb,
                      float xmaxb,
                      float yminb,
                      float ymaxb,
                      float zminb,
                      float zmaxb, int param_option,
                      float pli, float plj, float plk, float distance,
                      float strtx, float strty, float strtz)
{
    coDoPolygons *dummy;
    int *cor_list;
    int *pol_list;
    float angle, w, rad = distance; // radius for sphere FIXME
    float *x, *y, *z;
    int i, n;

    n = CREFN; // the degree of refinement for cylinder surfaces
    switch (param_option)
    {
    case 0:
    { //Plane
        float xp[8], yp[8], zp[8]; // projection of border box points
        // on cutting surface
        float x4[8], y4[8], z4[8]; // sl: the shadow will be a rectangle
        //     in order to set up a feasible
        //     algorithm without overlapping
        //     polygons
        border_proj(xp, yp, zp, xminb, xmaxb, yminb, ymaxb, zminb, zmaxb,
                    pli, plj, plk, distance);
        preserve_inertia(xp, yp, zp, x4, y4, z4, pli, plj, plk);

/*
         cor_list = new int[4];
         pol_list = new int[1];
         if(check_orientation(x4,y4,z4,pli,plj,plk)){
            cor_list[0] = 0;cor_list[1] = 1;cor_list[2] = 2;cor_list[3] = 3;
         } else {
            cor_list[0] = 0;cor_list[1] = 3;cor_list[2] = 2;cor_list[3] = 1;
         }
         pol_list[0]=0;
         */

#define FR(x, y) ((x)*0.95f + (y)*0.05f)

        x4[4] = FR(x4[0], x4[2]);
        x4[5] = FR(x4[1], x4[3]);
        x4[6] = FR(x4[2], x4[0]);
        x4[7] = FR(x4[3], x4[1]);

        y4[4] = FR(y4[0], y4[2]);
        y4[5] = FR(y4[1], y4[3]);
        y4[6] = FR(y4[2], y4[0]);
        y4[7] = FR(y4[3], y4[1]);

        z4[4] = FR(z4[0], z4[2]);
        z4[5] = FR(z4[1], z4[3]);
        z4[6] = FR(z4[2], z4[0]);
        z4[7] = FR(z4[3], z4[1]);
#undef FR

        int pol_list[] = { 0, 4, 8, 12 };

        int *cor_list;
        int right[] = { 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7 };
        int left[] = { 0, 3, 7, 4, 3, 2, 6, 7, 2, 1, 5, 6, 1, 0, 4, 5 };
        if (check_orientation(x4, y4, z4, pli, plj, plk))
            cor_list = left;
        else
            cor_list = right;

        dummy = new coDoPolygons(tr_name, 8, x4, y4, z4, 16, cor_list, 4, pol_list);
        //delete [] cor_list;
        //delete [] pol_list;
        break;
    }
    case 1: // Sphere
        float xS[17], yS[17], zS[17];
        int v_list[42 * 3], pl_list[42], vl[42], tsl[6], nb_pol;

        buildSphere(xS, yS, zS, pli, plj, plk, rad);
        build_SphereStrips(tsl, vl);
        nb_pol = trs2pol(42, 6, vl, tsl, v_list, pl_list);
        dummy = new coDoPolygons(tr_name, 17, xS, yS, zS, 3 * nb_pol, v_list, nb_pol, pl_list);
        break;
    case 2: // X - Cylinder
        // build points
        x = new float[2 * n];
        y = new float[2 * n];
        z = new float[2 * n];
        cor_list = new int[4 * n];
        pol_list = new int[n];
        rad = sqrt((plj - strty) * (plj - strty) + (plk - strtz) * (plk - strtz));
        w = (float)(2.0f * M_PI) / ((float)(n));
        for (i = 0; i < n; i++)
        {
            angle = w * ((float)i);

            x[i] = xmaxb;
            y[i] = plj + sin(angle) * rad; // change str <-> pl
            z[i] = plk + cos(angle) * rad;

            x[i + n] = xminb;
            y[i + n] = y[i];
            z[i + n] = z[i];
        }
        for (i = 0; i < n - 1; i++)
        {
            cor_list[4 * i] = i;
            cor_list[4 * i + 1] = i + n;
            cor_list[4 * i + 2] = i + n + 1;
            cor_list[4 * i + 3] = i + 1;
        }
        cor_list[4 * n - 4] = n - 1;
        cor_list[4 * n - 3] = 2 * n - 1;
        cor_list[4 * n - 2] = n;
        cor_list[4 * n - 1] = 0;
        for (i = 0; i < n; i++)
            pol_list[i] = 4 * i;
        dummy = new coDoPolygons(tr_name, 2 * n, x, y, z, 4 * n, cor_list, n, pol_list);
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] cor_list;
        delete[] pol_list;
        break;
    case 3: // Y - Cylinder
        // build points
        x = new float[2 * n];
        y = new float[2 * n];
        z = new float[2 * n];
        cor_list = new int[4 * n];
        pol_list = new int[n];
        rad = sqrt((pli - strtx) * (pli - strtx) + (plk - strtz) * (plk - strtz));
        w = (float)(2.0f * M_PI) / ((float)(n));
        for (i = 0; i < n; i++)
        {
            angle = w * ((float)i);

            x[i] = pli + sin(angle) * rad; // change str <-> pl
            y[i] = yminb;
            z[i] = plk + cos(angle) * rad;

            x[i + n] = x[i];
            y[i + n] = ymaxb;
            z[i + n] = z[i];
        }
        for (i = 0; i < n - 1; i++)
        {
            cor_list[4 * i] = i;
            cor_list[4 * i + 1] = i + n;
            cor_list[4 * i + 2] = i + n + 1;
            cor_list[4 * i + 3] = i + 1;
        }
        cor_list[4 * n - 4] = n - 1;
        cor_list[4 * n - 3] = 2 * n - 1;
        cor_list[4 * n - 2] = n;
        cor_list[4 * n - 1] = 0;
        for (i = 0; i < n; i++)
            pol_list[i] = 4 * i;
        dummy = new coDoPolygons(tr_name, 2 * n, x, y, z, 4 * n, cor_list, n, pol_list);
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] cor_list;
        delete[] pol_list;
        break;
    case 4: // Z - Cylinder
        // build points
        x = new float[2 * n];
        y = new float[2 * n];
        z = new float[2 * n];
        cor_list = new int[4 * n];
        pol_list = new int[n];
        rad = sqrt((pli - strtx) * (pli - strtx) + (plj - strty) * (plj - strty));
        w = (float)(2.0f * M_PI) / ((float)(n));
        for (i = 0; i < n; i++)
        {
            angle = w * ((float)i);

            x[i] = pli + cos(angle) * rad; // change str <-> pl
            y[i] = plj + sin(angle) * rad;
            z[i] = zminb;

            x[i + n] = x[i];
            y[i + n] = y[i];
            z[i + n] = zmaxb;
        }
        for (i = 0; i < n - 1; i++)
        {
            cor_list[4 * i] = i;
            cor_list[4 * i + 1] = i + n;
            cor_list[4 * i + 2] = i + n + 1;
            cor_list[4 * i + 3] = i + 1;
        }
        cor_list[4 * n - 4] = n - 1;
        cor_list[4 * n - 3] = 2 * n - 1;
        cor_list[4 * n - 2] = n;
        cor_list[4 * n - 1] = 0;
        for (i = 0; i < n; i++)
            pol_list[i] = 4 * i;
        dummy = new coDoPolygons(tr_name, 2 * n, x, y, z, 4 * n, cor_list, n, pol_list);
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] cor_list;
        delete[] pol_list;
        break;
    default:
        dummy = new coDoPolygons(tr_name, 0, 0, 0);
        Covise::sendError("ERROR: invalid option choice");
        // Covise::send_stop_pipeline();
    }
    return dummy;
}

// sl: Only called when we have to produce a dummy and
// we have received a set... In the case of non sets, it is better not to
// modify anything, lest we may get entangled in a knotty point.
coDoVec3 *Plane::dummy_normals(const char *nname,
                               float *coords_x, float *coords_y, float *coords_z, int param_option,
                               float pla[3], float rad, float start[3])
{
    int nb_coords;
    coDoVec3 *p_return;
    float *u_out;
    float *v_out;
    float *w_out;

    // if(!gennormals) return 0;

    nb_coords = DummyNormals(param_option);

    p_return = new coDoVec3(nname, nb_coords);
    if (!p_return->objectOk())
    {
        Covise::sendWarning("Normals could not be 'normally' created");
        return 0;
    }
    p_return->getAddresses(&u_out, &v_out, &w_out);
    fill_normals(u_out, v_out, w_out, coords_x, coords_y, coords_z, nb_coords, param_option, pla, rad, start);
    return p_return;
}

int Plane::DummyData(int dtype, float **data1, float **data2, float **data3)
{
    int fnc, i;

    fnc = 0;
    if (*data1)
    {
        delete[] * data1;
        *data1 = NULL;
    }
    if (!dtype) // Vector Data
    {
        if (*data2)
        {
            delete[] * data2;
            *data2 = NULL;
        }
        if (*data3)
        {
            delete[] * data3;
            *data3 = NULL;
        }
    }

    switch (option)
    {
    case 0: // Plane
        fnc = 4; // sl: now 4 instead of 8
        break;
    case 1: // Sphere
        fnc = 17;
        break;
    case 2: // X - Cylinder
        fnc = 2 * CREFN;
        break;
    case 3: // Y - Cylinder
        fnc = 2 * CREFN;
        break;
    case 4: // Z - Cylinder
        fnc = 2 * CREFN;
        break;
    default:
        cerr << "Error in DummyData :Unknown option" << option << endl;
    }
    if (fnc != 0)
    {
        *data1 = new float[fnc];
        for (i = 0; i < fnc; i++)
            (*data1)[i] = 0;
        if (!dtype) // Vector Data
        {
            *data2 = new float[fnc];
            *data3 = new float[fnc];
            for (i = 0; i < fnc; i++)
                (*data2)[i] = 0;
            for (i = 0; i < fnc; i++)
                (*data3)[i] = 0;
        }
    }

    return fnc;
}

int Plane::DummyNormals(int param_option)
{
    int fnc;

    fnc = 0;

    switch (param_option)
    {
    case 0:
        fnc = 8; // Plane
        break; // 8 for frame
    case 1:
        fnc = 17; // Sphere
        break;
    case 2:
        fnc = 2 * CREFN; // X - Cylinder
        break;
    case 3:
        fnc = 2 * CREFN; // Y - Cylinder
        break;
    case 4:
        fnc = 2 * CREFN; // Z - Cylinder
        break;
    default:
        cerr << "Error in DummyNormals :Unknown option" << param_option << endl;
    }

    return fnc;
}

// Data_name: name of output data object
// Normal_name: name of output data object
// Triangle_name: name of output poly/strip object
// Data_set, Normal_set, Triangle_set:
//    lists for a set in which the previous output objects have to be inserted
// currNumber: location in the previous lists
// species for Data_name

void Plane::createcoDistributedObjects(const char *Data_name_scal,
                                       const char *Data_name_vect,
                                       const char *Normal_name,
                                       const char *Triangle_name,
                                       AttributeContainer &gridAttrs,
                                       AttributeContainer &dataAttrs)
{
    float *u_out, *v_out, *w_out;
    int *vl, *pl, i;
    int nb_coords = 0;
    s_data_out = NULL;
    v_data_out = NULL;
    polygons_out = NULL;
    strips_out = NULL;
    normals_out = NULL;

    //if(num_coords==0)
    //   return;

    if ((Data_name_scal) || (Data_name_vect))
    {

        if ((Datatype == 1) || (Datatype == 2)) // (Scalar Data)
        {
            nb_coords = num_coords;
            if (nb_coords == 0)
                s_data_out = new coDoFloat(Data_name_scal, 0);
            else
                s_data_out = new coDoFloat(Data_name_scal, nb_coords, S_Data);

            if (!s_data_out->objectOk())
            {
                int n = 0;
                const char **attr = NULL, **setting = NULL;
                if (grid_in)
                    n = grid_in->getAllAttributes(&attr, &setting);
                else if (sgrid_in)
                    n = sgrid_in->getAllAttributes(&attr, &setting);
                else if (ugrid_in)
                    n = ugrid_in->getAllAttributes(&attr, &setting);
                else if (rgrid_in)
                    n = rgrid_in->getAllAttributes(&attr, &setting);
                if (n > 0)
                    s_data_out->addAttributes(n, attr, setting);

                Covise::sendError("ERROR: creation of data object 'dataOut' failed");

                if (Datatype == 1)
                    return;
            }

            dataAttrs.addAttributes(s_data_out);
        }
        if ((Datatype == 0) || (Datatype == 2))
        { // (Vector Data)
            nb_coords = num_coords;
            if (nb_coords == 0)
                v_data_out = new coDoVec3(Data_name_vect, 0);
            else
                v_data_out = new coDoVec3(Data_name_vect, nb_coords, V_Data_U, V_Data_V, V_Data_W);

            if (!v_data_out->objectOk())
            {
                int n = 0;
                const char **attr, **setting;
                if (grid_in)
                    n = grid_in->getAllAttributes(&attr, &setting);
                else if (sgrid_in)
                    n = sgrid_in->getAllAttributes(&attr, &setting);
                else if (ugrid_in)
                    n = ugrid_in->getAllAttributes(&attr, &setting);
                else if (rgrid_in)
                    n = rgrid_in->getAllAttributes(&attr, &setting);
                if (n > 0)
                    v_data_out->addAttributes(n, attr, setting);
                Covise::sendError("ERROR: creation of data object 'dataOut' failed");

                return;
            }

            dataAttrs.addAttributes(v_data_out);
        }
    }

    num_vertices = (int)(vertex - vertice_list);

    ////// generate strips
    int no_cut = 0; // set to 1 if there is no cut
    if (genstrips)
    {

        if (num_coords)
        {
            createStrips();
        }

        if (num_coords)
        {
            strips_out = new coDoTriangleStrips(Triangle_name, num_coords, coords_x, coords_y, coords_z,
                                                num_triangles + 2 * num_strips, ts_vertice_list, num_strips,
                                                ts_line_list);
            delete[] ts_vertice_list;
            delete[] ts_line_list;
        }
        else
        {
            strips_out = new coDoTriangleStrips(Triangle_name, 0, 0, 0);
        }

        if (strips_out->objectOk())
        {
            if ((Datatype == 1) || (Datatype == 2))
                gridAttrs.addAttributes(strips_out, Data_name_scal);
            if ((Datatype == 0) || (Datatype == 2))
                gridAttrs.addAttributes(strips_out, Data_name_vect);

            strips_out->addAttribute("vertexOrder", "2");

            if (strips_out->getAttribute("COLOR") == 0)
                strips_out->addAttribute("COLOR", "white");

        } // objectOk
        else
        {
            Covise::sendError("ERROR: creation of tri_strip object 'meshOut' failed");
            return;
        }
    } // trianglestrips

    ///// Create Polygons

    else
    {
        if (num_coords)
            polygons_out = new coDoPolygons(Triangle_name, num_coords, num_vertices, num_triangles);
        else
        {
            polygons_out = new coDoPolygons(Triangle_name, 0, 0, 0);
        }

        if (polygons_out->objectOk())
        {
            if ((Datatype == 1) || (Datatype == 2))
                gridAttrs.addAttributes(polygons_out, Data_name_scal);
            if ((Datatype == 0) || (Datatype == 2))
                gridAttrs.addAttributes(polygons_out, Data_name_vect);

            if (num_coords)
            {
                polygons_out->getAddresses(&u_out, &v_out, &w_out, &vl, &pl);
                if (no_cut == 0) // if there is a cut,
                {
                    // otherwise coords_? have no meaning
                    // for the dummy...
                    memcpy(u_out, coords_x, num_coords * sizeof(float));
                    memcpy(v_out, coords_y, num_coords * sizeof(float));
                    memcpy(w_out, coords_z, num_coords * sizeof(float));
                    memcpy(vl, vertice_list, num_vertices * sizeof(int));
                    for (i = 0; i < num_triangles; i++)
                        pl[i] = i * 3;
                }
            }

            // 0?
            polygons_out->addAttribute("vertexOrder", "2");

            // we don't need this any more
            //char *DataIn 	=  Covise::get_object_name("dataIn");
            //polygons_out->addAttribute("DataObjectName",DataIn);

            if (polygons_out->getAttribute("COLOR") == 0)
                polygons_out->addAttribute("COLOR", "blue");

        } // objectOk
        else
        {
            Covise::sendError("ERROR: creation of polygonal object failed");
            return;
        }
    } // polygons

    if (gennormals && Normal_name)
    {
        normals_out = new coDoVec3(Normal_name, nb_coords);
        if (normals_out->objectOk())
        {
            normals_out->getAddresses(&u_out, &v_out, &w_out);

            { // there is a real cut
                // and coords_? have physical meaning
                float pla[3], star[3];
                float rad = radius;
                pla[0] = planei;
                pla[1] = planej;
                pla[2] = planek;
                star[0] = startx;
                star[1] = starty;
                star[2] = startz;
                fill_normals(u_out, v_out, w_out,
                             coords_x, coords_y, coords_z, nb_coords,
                             option, pla, rad, star);
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'normalsOut' failed");

            return;
        }

    } // gennormals
}

void Plane::createStrips()
{
    int i, n0, n1, n2, n, next_n, j, tn, el = 0, num_try;
    int *neighbors, *np, *ts_vl, *ts_ll, *td, *triangle_done;

    np = neighbors = new int[num_coords * maxPolyPerVertex];
    td = triangle_done = new int[num_triangles];
    ts_vl = ts_vertice_list = new int[num_vertices];
    ts_ll = ts_line_list = new int[num_triangles];
    for (i = 0; i < num_coords * maxPolyPerVertex; i++)
    {
        *np++ = 0;
    }
    n = 0;
    for (i = 0; i < num_vertices; i += 3)
    {
        np = neighbors + maxPolyPerVertex * vertice_list[i];
        if (*np < (maxPolyPerVertex - 1))
        {
            (*np)++;
            *(np + (*np)) = n;
        }

        np = neighbors + maxPolyPerVertex * vertice_list[i + 1];
        if (*np < (maxPolyPerVertex - 1))
        {
            (*np)++;
            *(np + (*np)) = n;
        }

        np = neighbors + maxPolyPerVertex * vertice_list[i + 2];
        if (*np < (maxPolyPerVertex - 1))
        {
            (*np)++;
            *(np + (*np)) = n;
        }

        *td++ = 0; // flag = TRUE if Triangle already done
        n++;
    }
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
            // printf("%d\n",el);
            *td = 1;
            num_strips++;
            el = 0;
            num_try = 0;
            *ts_ll++ = (int)(ts_vl - ts_vertice_list); // line list points to beginning of strip
            *ts_vl++ = n0 = vertice_list[i + 1]; // first and second vertex of strip
            *ts_vl++ = n1 = vertice_list[i + 2];
            next_n = n2 = vertice_list[i];
            while ((el < 2) && (num_try < 3))
            {
                while (next_n != -1)
                {
                    el++;
                    *ts_vl++ = next_n; // next vertex of Strip
                    n2 = next_n;
                    next_n = -1;
                    // find the next vertex now
                    np = neighbors + maxPolyPerVertex * n2; // look for neighbors at point 2
                    for (j = *np; j > 1; j--)
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
    //  printf("%d\n",el);
    // *ts_vl++ = next_n; // we nearly forgot to add the last we found
    // if it was a lonly triangle
    delete[] neighbors;
    delete[] triangle_done;
    //  printf("strips: %d\n",num_strips);
    //  printf("triangles: %d\n",num_triangles);
}

//========================= RECT_Plane =================================

RECT_Plane::RECT_Plane(
    int n_elem,
    int n_nodes,
    int Type,
    int *p_el,
    int *p_cl,
    int *p_tl,
    float *p_x_in,
    float *p_y_in,
    float *p_z_in,
    float *p_s_in,
    unsigned char *p_bs_in,
    float *p_i_in,
    float *p_u_in,
    float *p_v_in,
    float *p_w_in,
    const coDistributedObject *p_rgrid_in,
    int p_x_size,
    int p_y_size,
    int p_z_size,
    int maxPoly,
    float planei_, float planej_, float planek_, float startx_,
    float starty_, float startz_, float myDistance_,
    float radius_, int gennormals_, int option_,
    int genstrips_, char *ib)
{
    iblank = ib;
    unstr_ = false;
    maxPolyPerVertex = maxPoly;
    NodeInfo *node;
    int i, j, k;
    el = p_el;
    cl = p_cl;
    tl = p_tl;
    x_in = p_x_in;
    y_in = p_y_in;
    z_in = p_z_in;
    s_in = p_s_in;
    bs_in = p_bs_in;
    i_in = p_i_in;
    u_in = p_u_in;
    v_in = p_v_in;
    w_in = p_w_in;
    x_size = p_x_size;
    y_size = p_y_size;
    z_size = p_z_size;
    sgrid_in = NULL;
    ugrid_in = NULL;
    rgrid_in = (coDoRectilinearGrid *)p_rgrid_in;
    grid_in = NULL;
    cur_line_elem = 0;

    Datatype = Type;
    num_nodes = n_nodes;
    num_elem = n_elem;

    if (option_ > 1)
    {
        planei = startx_;
        planej = starty_;
        planek = startz_;
        startx = planei_;
        starty = planej_;
        startz = planek_;
    }
    else
    {
        planei = planei_;
        planej = planej_;
        planek = planek_;
        startx = startx_;
        starty = starty_;
        startz = startz_;
    }
    myDistance = myDistance_;
    radius = radius_;
    gennormals = gennormals_;
    option = option_;
    genstrips = genstrips_;

    //    node_table   = (NodeInfo *)malloc(n_nodes*sizeof(NodeInfo));
    node_table = new NodeInfo[num_nodes];
    node = node_table;
    float tmpi, tmpj, tmpk;

    // Transform uniform grid into rectilinear grid
    if (const coDoUniformGrid *ugrid_in = dynamic_cast<const coDoUniformGrid *>(p_rgrid_in))
    {
        float xdisc, ydisc, zdisc;
        ugrid_in->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);
        x_in = new float[x_size];
        y_in = new float[y_size];
        z_in = new float[z_size];
        xdisc = (x_max - x_min) / (x_size - 1);
        ydisc = (y_max - y_min) / (y_size - 1);
        zdisc = (z_max - z_min) / (z_size - 1);
        if (xdisc > 0)
        {
            for (i = 0; i < x_size; i++)
                x_in[i] = x_min + xdisc * i;
        }
        else
        {
            for (i = 0; i < x_size; i++)
                x_in[i] = x_max - xdisc * i;
        }
        if (ydisc > 0)
        {
            for (i = 0; i < y_size; i++)
                y_in[i] = y_min + ydisc * i;
        }
        else
        {
            for (i = 0; i < y_size; i++)
                y_in[i] = y_max - ydisc * i;
        }
        if (zdisc > 0)
        {
            for (i = 0; i < z_size; i++)
                z_in[i] = z_min + zdisc * i;
        }
        else
        {
            for (i = 0; i < z_size; i++)
                z_in[i] = z_max - zdisc * i;
        }
    }

    if (option == 0) // unitize normal of plane
    {
        float length = sqrt(planei * planei + planej * planej + planek * planek);
        planei /= length;
        planej /= length;
        planek /= length;
    }

    //
    //
    // calculate necessary cubes
    //
    //

    if (option == 1)
        sradius = myDistance * myDistance;
    else if (option != 0)
        sradius = radius * radius;

    necessary_cubes(option);

    //
    //
    // set necessary corners
    //
    //

    if (zind_min <= 0)
        zind_start = 0;
    else
        zind_start = zind_min - 1;
    if (xind_min <= 0)
        xind_start = 0;
    else
        xind_start = xind_min - 1;
    if (option == 0)
    {

        node += xind_min * y_size * z_size;
        for (i = xind_min; i <= xind_max + 1 && i < x_size; i++)
        {
            node += yind_min * z_size;
            tmpi = planei * x_in[i] - myDistance;
            for (j = yind_min; j <= yind_max + 1 && j < y_size; j++)
            {
                tmpj = tmpi + planej * y_in[j];
                node += zind_start;
                for (k = zind_start; k <= zind_max + 1 && k < z_size; k++)
                {
                    if (Point_is_relevant(i, j, k))
                    {
                        node->targets[0] = 0;
                        // Calculate the myDistance of each node
                        // to the Cuttingplane
                        node->dist = (tmpj + planek * z_in[k]);
                        node->side = (node->dist >= 0 ? 1 : 0);
                    }
                    node++;
                }
                node += z_size - k;
            }
            node += (y_size - j) * z_size;
        }
    }

    else if (option == 1)
    {
        node += xind_min * y_size * z_size;
        for (i = xind_min; i <= xind_max + 1 && i < x_size; i++)
        {
            node += yind_min * z_size;
            tmpi = planei - x_in[i];
            tmpi *= tmpi;
            for (j = yind_min; j <= yind_max + 1 && j < y_size; j++)
            {
                tmpj = planej - y_in[j];
                tmpj = tmpi + tmpj * tmpj;
                node += zind_start;
                for (k = zind_start; k <= zind_max + 1 && k < z_size; k++)
                {
                    if (Point_is_relevant(i, j, k))
                    {
                        node->targets[0] = 0;
                        // Calculate the myDistance of each node
                        // to the Cuttingplane
                        tmpk = planek - z_in[k];
                        node->dist = sqrt(tmpj + tmpk * tmpk) - myDistance;
                        node->side = (node->dist >= 0 ? 1 : 0);
                    }
                    node++;
                }
                node += z_size - k;
            }
            node += (y_size - j) * z_size;
        }
    }

    else if (option == 2)
    {
        node += xind_min * y_size * z_size;
        for (i = xind_min; i <= xind_max + 1 && i < x_size; i++)
        {
            node += yind_min * z_size;
            for (j = yind_min; j <= yind_max + 1 && j < y_size; j++)
            {
                tmpj = starty - y_in[j];
                tmpj *= tmpj;
                node += zind_start;
                for (k = zind_start; k <= zind_max + 1 && k < z_size; k++)
                {
                    if (Point_is_relevant(i, j, k))
                    {
                        node->targets[0] = 0;
                        // Calculate the myDistance of each node
                        // to the Cuttingplane
                        tmpk = startz - z_in[k];
                        node->dist = sqrt(tmpj + tmpk * tmpk) - radius;
                        node->side = (node->dist >= 0 ? 1 : 0);
                    }
                    node++;
                }
                node += z_size - k;
            }
            node += (y_size - j) * z_size;
        }
    }

    else if (option == 3)
    {
        node += xind_min * y_size * z_size;
        for (i = xind_min; i <= xind_max + 1 && i < x_size; i++)
        {
            node += yind_min * z_size;
            tmpi = startx - x_in[i];
            tmpi *= tmpi;
            for (j = yind_min; j <= yind_max + 1 && j < y_size; j++)
            {
                node += zind_start;
                for (k = zind_start; k <= zind_max + 1 && k < z_size; k++)
                {
                    if (Point_is_relevant(i, j, k))
                    {
                        node->targets[0] = 0;
                        // Calculate the myDistance of each node
                        // to the Cuttingplane
                        tmpk = startz - z_in[k];
                        node->dist = sqrt(tmpi + tmpk * tmpk) - radius;
                        node->side = (node->dist >= 0 ? 1 : 0);
                    }
                    node++;
                }
                node += z_size - k;
            }
            node += (y_size - j) * z_size;
        }
    }

    else if (option == 4)
    {
        node += xind_min * y_size * z_size;
        for (i = xind_min; i <= xind_max + 1 && i < x_size; i++)
        {
            node += yind_min * z_size;
            tmpi = startx - x_in[i];
            tmpi *= tmpi;
            for (j = yind_min; j <= yind_max + 1 && j < y_size; j++)
            {
                tmpj = starty - y_in[j];
                tmpj = tmpi + tmpj * tmpj;
                node += zind_start;
                for (k = zind_start; k <= zind_max + 1 && k < z_size; k++)
                {
                    if (Point_is_relevant(i, j, k))
                    {
                        node->targets[0] = 0;
                        // Calculate the myDistance of each node
                        // to the Cuttingplane
                        node->dist = sqrt(tmpj) - radius;
                        node->side = (node->dist >= 0 ? 1 : 0);
                    }
                    node++;
                }
                node += z_size - k;
            }
            node += (y_size - j) * z_size;
        }
    }

    planei = planei_;
    planej = planej_;
    planek = planek_;
    startx = startx_;
    starty = starty_;
    startz = startz_;

    num_triangles = num_vertices = num_coords = 0;
    vertice_list = new int[n_elem * (size_t)12];
    vertex = vertice_list;
    coords_x = new float[n_nodes * (size_t)3];
    coords_y = new float[n_nodes * (size_t)3];
    coords_z = new float[n_nodes * (size_t)3];
    coord_x = coords_x;
    coord_y = coords_y;
    coord_z = coords_z;
    S_Data = NULL;
    if ((Datatype == 1) || (Datatype == 2)) // scalar data or scalar & vector data
    {
        S_Data_p = S_Data = new float[n_nodes * 3];
    }
    if ((Datatype == 0) || (Datatype == 2)) // vector data or scalar & vector data
    {
        V_Data_U_p = V_Data_U = new float[n_nodes * 3];
        V_Data_V_p = V_Data_V = new float[n_nodes * 3];
        V_Data_W_p = V_Data_W = new float[n_nodes * 3];
    }
    if (i_in)
    {
        I_Data_p = I_Data = new float[n_nodes * 3];
    }
    else
        I_Data = S_Data;
}

int RECT_Plane::Point_is_relevant(int i, int j, int k)
{
    if (option == 0) // plane
    {
        if (CHECK_Z)
        {
            if (j == yind_max + 1 || i == xind_max + 1 || k == zind_start || is_between((float)cutting_cubes[i - xind_min][j - yind_min].lower, (float)k,
                                                                                        (float)cutting_cubes[i - xind_min][j - yind_min].upper + 1) || (i - xind_min - 1 >= 0 && is_between((float)cutting_cubes[i - xind_min - 1][j - yind_min].lower, (float)k,
                                                                                                                                                                                            (float)cutting_cubes[i - xind_min - 1][j - yind_min].upper + 1)) || (j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[i - xind_min][j - yind_min - 1].lower - 1, (float)k,
                                                                                                                                                                                                                                                                                                     (float)cutting_cubes[i - xind_min][j - yind_min - 1].upper + 1)) || (i - xind_min - 1 >= 0 && j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[i - xind_min - 1][j - yind_min - 1].lower, (float)k,
                                                                                                                                                                                                                                                                                                                                                                                                                                       (float)cutting_cubes[i - xind_min - 1][j - yind_min - 1].upper + 1)))
                return 1;
        }
        else
        {
            if (j == yind_max + 1 || k == zind_max + 1 || k == zind_start || is_between((float)cutting_cubes[k - zind_min][j - yind_min].lower, (float)i,
                                                                                        (float)cutting_cubes[k - zind_min][j - yind_min].upper + 1) || (k - zind_min - 1 >= 0 && is_between((float)cutting_cubes[k - zind_min - 1][j - yind_min].lower, (float)i,
                                                                                                                                                                                            (float)cutting_cubes[k - zind_min - 1][j - yind_min].upper + 1)) || (j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[k - zind_min][j - yind_min - 1].lower - 1, (float)i,
                                                                                                                                                                                                                                                                                                     (float)cutting_cubes[k - zind_min][j - yind_min - 1].upper + 1)) || (k - zind_min - 1 >= 0 && j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[k - zind_min - 1][j - yind_min - 1].lower, (float)i,
                                                                                                                                                                                                                                                                                                                                                                                                                                       (float)cutting_cubes[k - zind_min - 1][j - yind_min - 1].upper + 1)))
                return 1;
        }
    }
    else if (option == 4) // cylinder- z-dim
    {
        if (j == yind_max + 1 || i == xind_max + 1 || k == zind_start ||

            is_between((float)sym_cutting_cubes[k - zind_min][j - yind_min].lower, (float)i,
                       (float)sym_cutting_cubes[k - zind_min][j - yind_min].upper + 1) || (k - zind_min - 1 >= 0 && is_between((float)sym_cutting_cubes[k - zind_min - 1][j - yind_min].lower, (float)i,
                                                                                                                               (float)sym_cutting_cubes[k - zind_min - 1][j - yind_min].upper + 1)) || (j - yind_min - 1 >= 0 && is_between((float)sym_cutting_cubes[k - zind_min][j - yind_min - 1].lower - 1, (float)i,
                                                                                                                                                                                                                                            (float)sym_cutting_cubes[k - zind_min][j - yind_min - 1].upper + 1)) || (k - zind_min - 1 >= 0 && j - yind_min - 1 >= 0 && is_between((float)sym_cutting_cubes[k - zind_min - 1][j - yind_min - 1].lower - 1, (float)i,
                                                                                                                                                                                                                                                                                                                                                                                  (float)sym_cutting_cubes[k - zind_min - 1][j - yind_min - 1].upper + 1)) ||

            is_between((float)cutting_cubes[k - zind_min][j - yind_min].lower, (float)i,
                       (float)cutting_cubes[k - zind_min][j - yind_min].upper + 1) || (k - zind_min - 1 >= 0 && is_between((float)cutting_cubes[k - zind_min - 1][j - yind_min].lower, (float)i,
                                                                                                                           (float)cutting_cubes[k - zind_min - 1][j - yind_min].upper + 1)) || (j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[k - zind_min][j - yind_min - 1].lower - 1, (float)i,
                                                                                                                                                                                                                                    (float)cutting_cubes[k - zind_min][j - yind_min - 1].upper + 1)) || (k - zind_min - 1 >= 0 && j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[k - zind_min - 1][j - yind_min - 1].lower - 1, (float)i,
                                                                                                                                                                                                                                                                                                                                                                      (float)cutting_cubes[k - zind_min - 1][j - yind_min - 1].upper + 1)))
            return 1;
    }

    else
    {
        if (j == yind_max + 1 || i == xind_max + 1 || k == zind_start ||

            is_between((float)sym_cutting_cubes[i - xind_min][j - yind_min].lower, (float)k,
                       (float)sym_cutting_cubes[i - xind_min][j - yind_min].upper + 1) || (i - xind_min - 1 >= 0 && is_between((float)sym_cutting_cubes[i - xind_min - 1][j - yind_min].lower, (float)k,
                                                                                                                               (float)sym_cutting_cubes[i - xind_min - 1][j - yind_min].upper + 1)) || (j - yind_min - 1 >= 0 && is_between((float)sym_cutting_cubes[i - xind_min][j - yind_min - 1].lower - 1, (float)k,
                                                                                                                                                                                                                                            (float)sym_cutting_cubes[i - xind_min][j - yind_min - 1].upper + 1)) || (i - xind_min - 1 >= 0 && j - yind_min - 1 >= 0 && is_between((float)sym_cutting_cubes[i - xind_min - 1][j - yind_min - 1].lower - 1, (float)k,
                                                                                                                                                                                                                                                                                                                                                                                  (float)sym_cutting_cubes[i - xind_min - 1][j - yind_min - 1].upper + 1)) ||

            is_between((float)cutting_cubes[i - xind_min][j - yind_min].lower, (float)k,
                       (float)cutting_cubes[i - xind_min][j - yind_min].upper + 1) || (i - xind_min - 1 >= 0 && is_between((float)cutting_cubes[i - xind_min - 1][j - yind_min].lower, (float)k,
                                                                                                                           (float)cutting_cubes[i - xind_min - 1][j - yind_min].upper + 1)) || (j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[i - xind_min][j - yind_min - 1].lower - 1, (float)k,
                                                                                                                                                                                                                                    (float)cutting_cubes[i - xind_min][j - yind_min - 1].upper + 1)) || (i - xind_min - 1 >= 0 && j - yind_min - 1 >= 0 && is_between((float)cutting_cubes[i - xind_min - 1][j - yind_min - 1].lower - 1, (float)k,
                                                                                                                                                                                                                                                                                                                                                                      (float)cutting_cubes[i - xind_min - 1][j - yind_min - 1].upper + 1)))
            return 1;
    }
    return 0;
}

int RECT_Plane::find(float a, char find_what, char mode, bool checkborder)
{
    int pos;
    int begin = 0, end = 0;
    float min = 0.0, max = 0.0;
    int ifmin = 0, ifmax = 0;
    float *array = NULL;
    if (find_what == FINDX)
    {
        array = xunit;
        end = x_size - 1;
        min = xmin;
        ifmin = xind_min;
        max = xmax;
        ifmax = xind_max;
    }
    else if (find_what == FINDY)
    {
        array = yunit;
        end = y_size - 1;
        min = ymin;
        ifmin = yind_min;
        max = ymax;
        ifmax = yind_max;
    }
    else if (find_what == FINDZ)
    {
        array = zunit;
        end = z_size - 1;
        min = zmin;
        ifmin = zind_min;
        max = zmax;
        ifmax = zind_max;
    }
    else
    {
        fprintf(stderr, "RECT_Plane::find(): end, min, max, ifmin, ifmax, array  might be used uninitialized\n");
    }

    if (checkborder == true)
    {
        if (a <= min)
            return ifmin;
        if (a >= max)
            return ifmax;
    }

    while (1)
    {
        pos = (int)(0.5 * (begin + end));
        if (array[pos] == a)
        {
            if (mode == UP || pos - 1 < 0)
                return pos;
            else
                return pos - 1;
        }
        if (end - begin <= 1)
        {
            if (array[end] == a)
            {
                if (mode == UP || end - 1 < 0)
                    return end;
                else
                    return end - 1;
            }
            if (array[begin] == a)
            {
                if (mode == UP || begin - 1 < 0)
                    return begin;
                else
                    return begin - 1;
            }
            return begin;
        }
        if (array[pos] > a)
            end = pos;
        else
            begin = pos;
    }
}

inline bool RECT_Plane::is_between(float a, float b, float c)
{
    return (a <= b && b <= c);
}

float RECT_Plane::min_of_four(float a, float b, float c, float d, float cmin, float cmax)
{
    float min = cmax + 1;

    if (a < min && cmin < a && a < cmax)
        min = a;
    if (b < min && cmin < b && b < cmax)
        min = b;
    if (c < min && cmin < c && c < cmax)
        min = c;
    if (d < min && cmin < d && d < cmax)
        min = d;
    return min;
}

float RECT_Plane::max_of_four(float a, float b, float c, float d, float cmin, float cmax)
{
    float max = cmin - 1;
    if (a > max && cmin < a && a < cmax)
        max = a;
    if (b > max && cmin < b && b < cmax)
        max = b;
    if (c > max && cmin < c && c < cmax)
        max = c;
    if (d > max && cmin < d && d < cmax)
        max = d;
    return max;
}

float RECT_Plane::min_of_four(float a, float b, float c, float d)
{
    float min = FLT_MAX;

    if (a < min)
        min = a;
    if (b < min)
        min = b;
    if (c < min)
        min = c;
    if (d < min)
        min = d;
    return min;
}

float RECT_Plane::max_of_four(float a, float b, float c, float d)
{
    float max = -FLT_MAX;
    if (a > max)
        max = a;
    if (b > max)
        max = b;
    if (c > max)
        max = c;
    if (d > max)
        max = d;
    return max;
}

inline float RECT_Plane::x_line_with_plane(float y, float z)
{
    return (myDistance - planej * y - planek * z) / planei;
}

inline float RECT_Plane::y_line_with_plane(float x, float z)
{
    return (myDistance - planei * x - planek * z) / planej;
}

inline float RECT_Plane::z_line_with_plane(float x, float y)
{
    return (myDistance - planei * x - planej * y) / planek;
}

inline float RECT_Plane::xy_line_with_cylinder(float y)
{
    if (sradius - (y - starty) * (y - starty) >= 0)
        return sqrt(sradius - (y - starty) * (y - starty)) + startx;
    else
        return startx - 1e-04f;
}

inline float RECT_Plane::zx_line_with_cylinder(float x)
{
    if (sradius - (x - startx) * (x - startx) >= 0)
        return sqrt(sradius - (x - startx) * (x - startx)) + startz;
    else
        return startz - 1e-04f;
}

inline float RECT_Plane::zy_line_with_cylinder(float y)
{
    if (sradius - (y - starty) * (y - starty) >= 0)
        return sqrt(sradius - (y - starty) * (y - starty)) + startz;
    else
        return startz - 1e-04f;
}

inline float RECT_Plane::z_line_with_sphere(float x, float y)
{
    if (sradius - (y - planej) * (y - planej) - (x - planei) * (x - planei) >= 0)
        return sqrt(sradius - (y - planej) * (y - planej) - (x - planei) * (x - planei)) + planek;
    else
        return zmin - 1e-04f;
}

void RECT_Plane::necessary_cubes(int option)
{
    int i, j, k;

    float xorigmin, xorigmax, yorigmin, yorigmax, zorigmin, zorigmax;

    //
    // Plane
    //

    if (option == 0)
    {
        //
        // calculate cut between plane and outer edges of the grid
        // to get borders for the following calculation
        //

        // f: front, l: low, u:up
        float xfl = 0.0, xfu = 0.0, xbl = 0.0, xbu = 0.0, yfl = 0.0, yfu = 0.0, ybl = 0.0, ybu = 0.0, zfl = 0.0, zfu = 0.0, zbl = 0.0, zbu = 0.0;

        xunit = x_in;
        yunit = y_in;
        zunit = z_in;

        xorigmin = xunit[0];
        xorigmax = xunit[x_size - 1];
        yorigmin = yunit[0];
        yorigmax = yunit[y_size - 1];
        zorigmin = zunit[0];
        zorigmax = zunit[z_size - 1];

        //
        // initialize .min, .max
        //

        xmin = xorigmin - 1;
        xmax = xorigmax + 1;
        ymin = yorigmin - 1;
        ymax = yorigmax + 1;
        zmin = zorigmin - 1;
        zmax = zorigmax + 1;

        xind_min = 0;
        xind_max = x_size - 1;
        yind_min = 0;
        yind_max = y_size - 1;
        zind_min = 0;
        zind_max = z_size - 1;

        if (planei == 0) // x-axis is in the plane
        {
            xmin = xorigmin;
            xmax = xorigmax;
        }
        else
        {
            xfl = x_line_with_plane(yorigmin, zorigmin);
            if (is_between(xorigmin, xfl, xorigmax))
            {
                ymin = yorigmin;
                zmin = zorigmin;
            }

            xfu = x_line_with_plane(yorigmax, zorigmin);
            if (is_between(xorigmin, xfu, xorigmax))
            {
                ymax = yorigmax;
                zmin = zorigmin;
            }

            xbl = x_line_with_plane(yorigmin, zorigmax);
            if (is_between(xorigmin, xbl, xorigmax))
            {
                ymin = yorigmin;
                zmax = zorigmax;
            }

            xbu = x_line_with_plane(yorigmax, zorigmax);
            if (is_between(xorigmin, xbu, xorigmax))
            {
                ymax = yorigmax;
                zmax = zorigmax;
            }
        }

        if (planej == 0) // y-axis is in the plane
        {
            ymin = yorigmin;
            ymax = yorigmax;
        }
        else
        {
            yfl = y_line_with_plane(xorigmin, zorigmin);
            if (is_between(yorigmin, yfl, yorigmax))
            {
                xmin = xorigmin;
                zmin = zorigmin;
            }

            yfu = y_line_with_plane(xorigmax, zorigmin);
            if (is_between(yorigmin, yfu, yorigmax))
            {
                xmax = xorigmax;
                zmin = zorigmin;
            }

            ybl = y_line_with_plane(xorigmin, zorigmax);
            if (is_between(yorigmin, ybl, yorigmax))
            {
                xmin = xorigmin;
                zmax = zorigmax;
            }

            ybu = y_line_with_plane(xorigmax, zorigmax);
            if (is_between(yorigmin, ybu, yorigmax))
            {
                xmax = xorigmax;
                zmax = zorigmax;
            }
        }

        if (!CHECK_Z) // z-axis is in the plane
        {
            zmin = zorigmin;
            zmax = zorigmax;
        }
        else
        {
            zfl = z_line_with_plane(xorigmin, yorigmin);
            if (is_between(zorigmin, zfl, zorigmax))
            {
                xmin = xorigmin;
                ymin = yorigmin;
            }

            zfu = z_line_with_plane(xorigmax, yorigmin);
            if (is_between(zorigmin, zfu, zorigmax))
            {
                xmax = xorigmax;
                ymin = yorigmin;
            }

            zbl = z_line_with_plane(xorigmin, yorigmax);
            if (is_between(zorigmin, zbl, zorigmax))
            {
                xmin = xorigmin;
                ymax = yorigmax;
            }

            zbu = z_line_with_plane(xorigmax, yorigmax);
            if (is_between(zorigmin, zbu, zorigmax))
            {
                xmax = xorigmax;
                ymax = yorigmax;
            }
        }

        //
        //
        // we indentify the coordinate of the cube with the coordinate of the corner with the smallest
        // coordinates in all dimensions
        //
        //
        //
        // we define the x- border of the cubes
        //

        if (xmin < xorigmin)
            xmin = min_of_four(xfl, xfu, xbl, xbu, xorigmin, xorigmax);
        if (xmax > xorigmax)
            xmax = max_of_four(xfl, xfu, xbl, xbu, xorigmin, xorigmax);

        xind_min = find(xmin, FINDX, DOWN, false);
        xind_max = find(xmax, FINDX, UP, false);
        if (xind_max > x_size - 2)
            xind_max = x_size - 2;
        if (xind_min > xind_max)
            xind_min = xind_max = 0;
        xdim = xind_max - xind_min + 1;

        //
        // we define the y-border
        //

        if (ymin < yorigmin)
            ymin = min_of_four(yfl, yfu, ybl, ybu, yorigmin, yorigmax);
        if (ymax > yorigmax)
            ymax = max_of_four(yfl, yfu, ybl, ybu, yorigmin, yorigmax);

        //      " " <<ymax << endl;

        yind_min = find(ymin, FINDY, DOWN, false);
        yind_max = find(ymax, FINDY, UP, false);
        if (yind_min > yind_max)
            yind_min = yind_max = 0;
        if (yind_max > y_size - 2)
            yind_max = y_size - 2;
        ydim = yind_max - yind_min + 1;

        //
        // we define the z- border
        //

        if (zmin < zorigmin)
            zmin = min_of_four(zfl, zfu, zbl, zbu, zorigmin, zorigmax);
        if (zmax > zorigmax)
            zmax = max_of_four(zfl, zfu, zbl, zbu, zorigmin, zorigmax);

        //      " " <<zmax << endl;

        zind_min = find(zmin, FINDZ, DOWN, false);
        zind_max = find(zmax, FINDZ, UP, false);
        if (zind_min > zind_max)
            zind_min = zind_max = 0;
        if (zind_max > z_size - 2)
            zind_max = z_size - 2;
        zdim = zind_max - zind_min + 1;

        //
        // sk:	We indentify the coordinate of the cube with the coordinate of the corner with the smallest
        // 	coordinates in all dimensions
        //
        // cutting_cubes: Detect which cubes of the rectilinear or uniform grid are
        //			really cutted by the chosen object. The base of the
        //			cutting cubes are the indices of the x-y-plane. Every coordinate point
        //			of this plane corresponds to one cube in the grid.
        //			For the indices i,j of the x,y- plane
        //			  cutting_cubes[i][j].lower -  cutting_cubes[i][j].upper
        // 			is the index range of the cutted cubes in z- dimension
        //			If the z-coordinate of the normal is too small we use the z-y-plane instead.
        //

        float one, two, three, four, max_of_them, min_of_them;
        int x, y, z;

        if (CHECK_Z)
        {
            cutting_cubes = new border *[xdim];
            _SpaghettiAntidot.insert(pair<border **, int>(cutting_cubes, xdim));
            memset(cutting_cubes, '\0', xdim * sizeof(border *));
            ERR0(cutting_cubes == NULL, "cannot allocate memory", return;);

            for (i = 0; i < xdim; i++)
            {
                cutting_cubes[i] = new border[ydim];
                ERR0(cutting_cubes[i] == NULL, "cannot allocate memory", return;);
                for (j = 0; j < ydim; j++)
                {
                    cutting_cubes[i][j].upper = -1;
                    cutting_cubes[i][j].lower = z_size + 1;
                }
            }

            //
            // for small data sets, do no further computation
            //

            if (xdim <= TOO_SMALL || ydim <= TOO_SMALL || zdim <= TOO_SMALL || !CHECK_Z)
            {
                for (x = 0; x < xdim; x++)
                {
                    for (y = 0; y < ydim; y++)
                    {
                        cutting_cubes[x][y].lower = zind_min;
                        cutting_cubes[x][y].upper = zind_max;
                    }
                }
                return;
            }

            //
            // otherwise do more
            //

            float **store = new float *[xdim + 1];
            ERR0(store == NULL, "cannot allocate memory", return;);

            for (x = 0; x < xdim + 1; x++)
            {
                store[x] = new float[ydim + 1];
                ERR0(store[x] == NULL, "cannot allocate memory", return;);
                for (y = 0; y < ydim + 1; y++)
                    store[x][y] = z_line_with_plane(xunit[x + xind_min], yunit[y + yind_min]);
            }

            for (x = 0; x < xdim; x++)
            {
                for (y = 0; y < ydim; y++)
                {
                    one = store[x][y];
                    two = store[x][y + 1];
                    three = store[x + 1][y];
                    four = store[x + 1][y + 1];

                    //  cout << x << "(" << xunit[x+xind_min] << ")," << y << "(" << yunit[y+yind_min] << "): " << one << " " << two <<  " " << three << " " << four << endl;
                    if (is_between(zmin, one, zmax) || is_between(zmin, two, zmax) || is_between(zmin, three, zmax) || is_between(zmin, four, zmax))
                    {
                        min_of_them = min_of_four(one, two, three, four);
                        max_of_them = max_of_four(one, two, three, four);

                        //    cout << min_of_them << " " << max_of_them << endl;

                        cutting_cubes[x][y].upper = find(max_of_them, FINDZ, UP, true);
                        cutting_cubes[x][y].lower = find(min_of_them, FINDZ, DOWN, true);
                    }
                }
            }

            /*    for( i=0; i<xdim; i++ )
               {
               for( j=0; j<ydim; j++ )
                cout << "("<< xind_min+i << ","<< yind_min+j << "," << cutting_cubes[i][j].lower <<":"<< cutting_cubes[i][j].upper << endl;
               } */

            for (i = 0; i < xdim + 1; i++)
                delete[] store[i];
            delete[] store;
        }

        else // normal_z too small
        {
            cutting_cubes = new border *[zdim];
            _SpaghettiAntidot.insert(pair<border **, int>(cutting_cubes, zdim));
            memset(cutting_cubes, '\0', zdim * sizeof(border *));
            ERR0(cutting_cubes == NULL, "cannot allocate memory", return;);

            for (k = 0; k < zdim; k++)
            {
                cutting_cubes[k] = new border[ydim];
                ERR0(cutting_cubes[k] == NULL, "cannot allocate memory", return;);
                for (j = 0; j < ydim; j++)
                {
                    cutting_cubes[k][j].upper = -1;
                    cutting_cubes[k][j].lower = x_size + 1;
                }
            }

            //
            // for small data sets, do no further computation
            //

            if (xdim <= TOO_SMALL || ydim <= TOO_SMALL || zdim <= TOO_SMALL)
            {
                for (z = 0; z < zdim; z++)
                {
                    for (y = 0; y < ydim; y++)
                    {
                        cutting_cubes[z][y].lower = xind_min;
                        cutting_cubes[z][y].upper = xind_max;
                    }
                }
                return;
            }

            //
            // otherwise do more
            //

            float **store = new float *[zdim + 1];
            ERR0(store == NULL, "cannot allocate memory", return;);

            for (z = 0; z < zdim + 1; z++)
            {
                store[z] = new float[ydim + 1];
                ERR0(store[z] == NULL, "cannot allocate memory", return;);
                for (y = 0; y < ydim + 1; y++)
                    store[z][y] = x_line_with_plane(yunit[y + yind_min], zunit[z + zind_min]);
            }

            for (z = 0; z < zdim; z++)
            {
                for (y = 0; y < ydim; y++)
                {
                    one = store[z][y];
                    two = store[z][y + 1];
                    three = store[z + 1][y];
                    four = store[z + 1][y + 1];

                    //     cout << y << "(" << yunit[y+yind_min] << ")," << z << "(" << zunit[z+zind_min] << "): " << one << " " << two <<  " " << three << " " << four << endl;
                    if (is_between(xmin, one, xmax) || is_between(xmin, two, xmax) || is_between(xmin, three, xmax) || is_between(xmin, four, xmax))
                    {
                        min_of_them = min_of_four(one, two, three, four);
                        max_of_them = max_of_four(one, two, three, four);

                        //    cout << min_of_them << " " << max_of_them << endl;

                        cutting_cubes[z][y].upper = find(max_of_them, FINDX, UP, true);
                        cutting_cubes[z][y].lower = find(min_of_them, FINDX, DOWN, true);
                    }
                }
            }

            /*    for( j=0; j<ydim; j++ )
               {
               for( k=0; k<zdim; k++ )
                cout << "("<<  yind_min+j << "," << zind_min+k << ") "<< cutting_cubes[k][j].lower <<":"<< cutting_cubes[k][j].upper << endl;
               } */

            for (k = 0; k < zdim + 1; k++)
                delete[] store[k];
            delete[] store;
        }
        //
        //
        // Sphere
        //
        //
    }
    else if (option == 1)
    {
        float one, two, three, four, max_of_them, min_of_them;
        int x, y;

        xunit = x_in;
        yunit = y_in;
        zunit = z_in;

        xorigmin = xunit[0];
        xorigmax = xunit[x_size - 1];
        yorigmin = yunit[0];
        yorigmax = yunit[y_size - 1];
        zorigmin = zunit[0];
        zorigmax = zunit[z_size - 1];

        zmin = zorigmin;
        zmax = zorigmax;

        if (radius <= xorigmax)
        {
            xind_max = find(myDistance + planei, FINDX, UP, false) + 1;
            if (xind_max > x_size - 2)
                xind_max = x_size - 2;
        }
        else
            xind_max = x_size - 2;
        if (-radius >= xorigmin)
        {
            xind_min = find(-myDistance - planei, FINDX, DOWN, false) - 1;
            if (xind_min < 0)
                xind_min = 0;
        }
        else
            xind_min = 0;
        if (radius <= yorigmax)
        {
            yind_max = find(myDistance + planej, FINDY, UP, false) + 1;
            if (yind_max > y_size - 2)
                yind_max = y_size - 2;
        }
        else
            yind_max = y_size - 2;
        if (-radius >= yorigmin)
        {
            yind_min = find(-myDistance - planej, FINDY, DOWN, false) - 1;
            if (yind_min < 0)
                yind_min = 0;
        }
        else
            yind_min = 0;
        if (radius <= zorigmax)
        {
            zind_max = find(myDistance + planek, FINDZ, DOWN, false) + 1;
            if (zind_max > z_size - 2)
                zind_max = z_size - 2;
        }
        else
            zind_max = z_size - 2;
        if (-radius >= zorigmin)
        {
            zind_min = find(-myDistance - planek, FINDZ, DOWN, false) - 1;
            if (zind_min < 0)
                zind_min = 0;
        }
        else
            zind_min = 0;

        if (xind_min > xind_max) // there is no cutting point
            xind_min = xind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;

        xdim = xind_max - xind_min + 1;
        ydim = yind_max - yind_min + 1;
        zdim = zind_max - zind_min + 1;

        //cout << "y: " << yind_min << "  " << yind_max << endl;
        //cout << "z: " << zind_min << "  " << zind_max << endl;
        cutting_cubes = new border *[xdim];
        _SpaghettiAntidot.insert(pair<border **, int>(cutting_cubes, xdim));
        memset(cutting_cubes, '\0', xdim * sizeof(border *));
        ERR0(cutting_cubes == NULL, "cannot allocate memory", return;);
        sym_cutting_cubes = new border *[xdim];
        _SpaghettiAntidot.insert(pair<border **, int>(sym_cutting_cubes, xdim));
        memset(sym_cutting_cubes, '\0', xdim * sizeof(border *));
        ERR0(sym_cutting_cubes == NULL, "cannot allocate memory", return;);

        //
        // for small data sets, don't worry
        //

        if (xdim <= TOO_SMALL || ydim <= TOO_SMALL || zdim <= TOO_SMALL)
        {
            for (x = 0; x < xdim; x++)
            {
                cutting_cubes[x] = new border[ydim];
                ERR0(cutting_cubes[x] == NULL, "cannot allocate memory", return;);
                sym_cutting_cubes[x] = new border[ydim];
                ERR0(sym_cutting_cubes[x] == NULL, "cannot allocate memory", return;);
                for (y = 0; y < ydim; y++)
                {
                    cutting_cubes[x][y].lower = zind_min;
                    cutting_cubes[x][y].upper = zind_max;
                    sym_cutting_cubes[x][y].upper = -1;
                    sym_cutting_cubes[x][y].lower = z_size + 1;
                }
            }
            return;
        }

        //
        // initialize array
        //

        for (i = 0; i < xdim; i++)
        {
            cutting_cubes[i] = new border[ydim];
            ERR0(cutting_cubes[i] == NULL, "cannot allocate memory", return;);
            sym_cutting_cubes[i] = new border[ydim];
            ERR0(cutting_cubes[i] == NULL, "cannot allocate memory", return;);
            for (j = 0; j < ydim; j++)
            {
                cutting_cubes[i][j].upper = -1;
                cutting_cubes[i][j].lower = z_size + 1;
                sym_cutting_cubes[i][j].upper = -1;
                sym_cutting_cubes[i][j].lower = z_size + 1;
            }
        }

        float **store = new float *[xdim + 1];
        ERR0(store == NULL, "cannot allocate memory", return;);

        for (x = 0; x < xdim + 1; x++)
        {
            store[x] = new float[ydim + 1];
            ERR0(store[x] == NULL, "cannot allocate memory", return;);
            for (y = 0; y < ydim + 1; y++)
                store[x][y] = z_line_with_sphere(xunit[x + xind_min], yunit[y + yind_min]);
        }

        for (x = 0; x < xdim; x++)
        {
            for (y = 0; y < ydim; y++)
            {
                one = store[x][y];
                two = store[x][y + 1];
                three = store[x + 1][y];
                four = store[x + 1][y + 1];

                //   cout << x << "," << y << ": " << one << " " << two <<  " " << three << " " << four << endl;
                min_of_them = min_of_four(one, two, three, four);
                max_of_them = max_of_four(one, two, three, four);
                //    cout << min_of_them << " " << max_of_them << endl;

                if (is_between(zmin, one, zmax) || is_between(zmin, two, zmax) || is_between(zmin, three, zmax) || is_between(zmin, four, zmax))
                {
                    cutting_cubes[x][y].upper = find(max_of_them, FINDZ, UP, true);
                    cutting_cubes[x][y].lower = find(min_of_them, FINDZ, DOWN, true);
                }

                if (is_between(zmin, 2 * planek - min_of_them, zmax) || is_between(zmin, 2 * planek - max_of_them, zmax))
                {
                    if (min_of_them > planek)
                        sym_cutting_cubes[x][y].upper = find(2 * planek - min_of_them, FINDZ, UP, true);
                    else
                        sym_cutting_cubes[x][y].upper = cutting_cubes[x][y].lower;
                    if (max_of_them > planek)
                        sym_cutting_cubes[x][y].lower = find(2 * planek - max_of_them, FINDZ, DOWN, true);
                    else
                        sym_cutting_cubes[x][y].lower = cutting_cubes[x][y].upper;
                }
            }
        }

        /*for( i=0; i<xdim; i++ )
       {
       for( j=0; j<ydim; j++ )
        {
        cout << i << "," << j << " :" << cutting_cubes[i][j].lower <<":"<< cutting_cubes[i][j].upper << endl;
        cout << "sym: " << i << " " << j << " :" << sym_cutting_cubes[i][j].lower <<":"<< sym_cutting_cubes[i][j].upper << endl;
        }
       }*/

        for (i = 0; i < xdim + 1; i++)
            delete[] store[i];
        delete[] store;
    }
    //
    //
    // cylinder- x
    //
    //

    else if (option == 2)
    {
        //  cout << "Radius2: " << sradius << endl;
        xunit = x_in;
        yunit = y_in;
        zunit = z_in;

        xorigmin = xunit[0];
        xorigmax = xunit[x_size - 1];
        yorigmin = yunit[0];
        yorigmax = yunit[y_size - 1];
        zorigmin = zunit[0];
        zorigmax = zunit[z_size - 1];

        xmin = startx;
        xmax = xorigmax;
        ymin = starty - radius;
        ymax = starty + radius;
        zmin = startz - radius;
        zmax = startz + radius;
        if (radius <= yorigmax)
            yind_max = find(ymax, FINDY, UP, false);
        else
            yind_max = y_size - 1;
        if (-radius >= yorigmin)
            yind_min = find(ymin, FINDY, DOWN, false);
        else
            yind_min = 0;
        if (radius <= zorigmax)
            zind_max = find(zmax, FINDZ, DOWN, false);
        else
            zind_max = z_size - 1;
        if (-radius >= zorigmin)
            zind_min = find(zmin, FINDZ, DOWN, false);
        else
            zind_min = 0;

        if (xind_min > xind_max) // there is no cutting point
            xind_min = xind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;
        xdim = x_size;
        ydim = yind_max - yind_min + 1;
        zdim = zind_max - zind_min + 1;

        xind_min = 0;
        xind_max = x_size - 1;

        //  cout << ymin << " " << ymax << " " << zmin << " " << zmax << " " << endl;
        //  cout << yind_min << " " << yind_max << " " << zind_min << " " << zind_max << " " << endl;
        //  cout << "Dim: " << xdim << " " << ydim << " " << zdim << endl;

        cutting_cubes = new border *[xdim];
        _SpaghettiAntidot.insert(pair<border **, int>(cutting_cubes, xdim));
        memset(cutting_cubes, '\0', xdim * sizeof(border *));
        ERR0(cutting_cubes == NULL, "cannot allocate memory", return;);
        sym_cutting_cubes = new border *[xdim];
        _SpaghettiAntidot.insert(pair<border **, int>(sym_cutting_cubes, xdim));
        memset(sym_cutting_cubes, '\0', xdim * sizeof(border *));
        ERR0(sym_cutting_cubes == NULL, "cannot allocate memory", return;);

        //
        // for small data sets, don't worry
        //

        int x, y;
        if (xdim <= TOO_SMALL || ydim <= TOO_SMALL || zdim <= TOO_SMALL)
        {
            for (x = 0; x < xdim; x++)
            {
                cutting_cubes[x] = new border[ydim];
                ERR0(cutting_cubes[x] == NULL, "cannot allocate memory", return;);
                sym_cutting_cubes[x] = new border[ydim];
                ERR0(sym_cutting_cubes[x] == NULL, "cannot allocate memory", return;);
                for (y = 0; y < ydim; y++)
                {
                    cutting_cubes[x][y].lower = zind_min;
                    cutting_cubes[x][y].upper = zind_max;
                    sym_cutting_cubes[x][y].upper = -1;
                    sym_cutting_cubes[x][y].lower = z_size + 1;
                }
            }
            return;
        }

        //
        // initialize array
        //

        for (i = 0; i < xdim; i++)
        {
            cutting_cubes[i] = new border[ydim];
            ERR0(cutting_cubes[i] == NULL, "cannot allocate memory", return;);
            sym_cutting_cubes[i] = new border[ydim];
            ERR0(sym_cutting_cubes[i] == NULL, "cannot allocate memory", return;);

            for (j = 0; j < ydim; j++)
            {
                cutting_cubes[i][j].upper = -1;
                cutting_cubes[i][j].lower = z_size + 1;
                sym_cutting_cubes[i][j].upper = -1;
                sym_cutting_cubes[i][j].lower = z_size + 1;
            }
        }

        // XXX: check if max_of_them and min_of_them gets initialized
        float one = 0.0, two, max_of_them = 0.0, min_of_them = 0.0;

        x = 0; // calculate one circle

        for (y = 0; y < ydim; y++)
        {
            if (y > 0)
                two = one;
            else
                two = zy_line_with_cylinder(yunit[y + yind_min]);

            one = zy_line_with_cylinder(yunit[y + yind_min + 1]);

            //  cout << y+yind_min << ": " << one << " " << two << endl;
            if (is_between(zmin, one, zmax) || is_between(zmin, two, zmax))
            {
                if (one > two)
                {
                    max_of_them = one;
                    min_of_them = two;
                }
                else
                {
                    max_of_them = two;
                    min_of_them = one;
                }

                cutting_cubes[x][y].upper = find(max_of_them, FINDZ, UP, true);
                cutting_cubes[x][y].lower = find(min_of_them, FINDZ, DOWN, true);
            }
            //
            // use the results for the symmetrical part
            //
            if (is_between(zmin, 2 * startz - min_of_them, zmax) || is_between(zmin, 2 * startz - max_of_them, zmax))
            {
                sym_cutting_cubes[x][y].upper = find(2 * startz - min_of_them, FINDZ, UP, true);
                sym_cutting_cubes[x][y].lower = find(2 * startz - max_of_them, FINDZ, DOWN, true);
            }
        }

        //
        // copy our results
        //

        for (x = 1; x < xdim; x++)
        {
            for (y = 0; y < ydim; y++)
            {
                cutting_cubes[x][y].lower = cutting_cubes[0][y].lower;
                cutting_cubes[x][y].upper = cutting_cubes[0][y].upper;
                sym_cutting_cubes[x][y].lower = sym_cutting_cubes[0][y].lower;
                sym_cutting_cubes[x][y].upper = sym_cutting_cubes[0][y].upper;
            }
        }

        /*  for( j=0; j<ydim; j++ )
        {
        cout << j << " :" << cutting_cubes[0][j].lower <<":"<< cutting_cubes[0][j].upper << endl;
        cout << "sym  "<<j << " :" << sym_cutting_cubes[0][j].lower <<":"<< sym_cutting_cubes[0][j].upper << endl;
        } */
    }

    //
    //  cylinder y  ( just a copy of the last procedure; x plays the role of y in the last procedure )
    //

    else if (option == 3)
    {
        xunit = x_in;
        yunit = y_in;
        zunit = z_in;

        xorigmin = xunit[0];
        xorigmax = xunit[x_size - 1];
        yorigmin = yunit[0];
        yorigmax = yunit[y_size - 1];
        zorigmin = zunit[0];
        zorigmax = zunit[z_size - 1];

        xmin = startx - radius;
        xmax = startx + radius;
        ymin = yorigmin;
        ymax = yorigmax;
        zmin = startz - radius;
        zmax = startz + radius;
        if (radius <= xorigmax)
            xind_max = find(xmax, FINDX, UP, false);
        else
            xind_max = x_size - 1;
        if (-radius >= xorigmin)
            xind_min = find(xmin, FINDX, DOWN, false);
        else
            xind_min = 0;
        if (radius <= zorigmax)
            zind_max = find(zmax, FINDZ, DOWN, false);
        else
            zind_max = z_size - 1;
        if (-radius >= zorigmin)
            zind_min = find(zmin, FINDZ, DOWN, false);
        else
            zind_min = 0;

        if (xind_min > xind_max) // there is no cutting point
            xind_min = xind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;

        xdim = xind_max - xind_min + 1;
        ydim = y_size;
        zdim = zind_max - zind_min + 1;

        yind_min = 0;
        yind_max = y_size - 1;

        //  cout << xmin << " " << xmax << " " << zmin << " " << zmax << " " << endl;
        //  cout << xind_min << " " << xind_max << " " << zind_min << " " << zind_max << " " << endl;
        //  cout << "Dim: " << xdim << " " << ydim << " " << zdim << endl;

        cutting_cubes = new border *[xdim];
        _SpaghettiAntidot.insert(pair<border **, int>(cutting_cubes, xdim));
        memset(cutting_cubes, '\0', xdim * sizeof(border *));
        ERR0(cutting_cubes == NULL, "cannot allocate memory", return;);
        sym_cutting_cubes = new border *[xdim];
        _SpaghettiAntidot.insert(pair<border **, int>(sym_cutting_cubes, xdim));
        memset(sym_cutting_cubes, '\0', xdim * sizeof(border *));
        ERR0(sym_cutting_cubes == NULL, "cannot allocate memory", return;);

        //
        // for small data sets, don't worry
        //

        int x, y;
        if (xdim <= TOO_SMALL || ydim <= TOO_SMALL || zdim <= TOO_SMALL)
        {
            for (x = 0; x < xdim; x++)
            {
                cutting_cubes[x] = new border[ydim];
                ERR0(cutting_cubes[x] == NULL, "cannot allocate memory", return;);
                sym_cutting_cubes[x] = new border[ydim];
                ERR0(sym_cutting_cubes[x] == NULL, "cannot allocate memory", return;);
                for (y = 0; y < ydim; y++)
                {
                    cutting_cubes[x][y].lower = zind_min;
                    cutting_cubes[x][y].upper = zind_max;
                    sym_cutting_cubes[x][y].upper = -1;
                    sym_cutting_cubes[x][y].lower = z_size + 1;
                }
            }
            return;
        }

        //
        // initialize array
        //

        for (i = 0; i < xdim; i++)
        {
            cutting_cubes[i] = new border[ydim];
            ERR0(cutting_cubes[i] == NULL, "cannot allocate memory", return;);
            sym_cutting_cubes[i] = new border[ydim];
            ERR0(sym_cutting_cubes[i] == NULL, "cannot allocate memory", return;);

            for (j = 0; j < ydim; j++)
            {
                cutting_cubes[i][j].upper = -1;
                cutting_cubes[i][j].lower = z_size + 1;
                sym_cutting_cubes[i][j].upper = -1;
                sym_cutting_cubes[i][j].lower = z_size + 1;
            }
        }

        // XXX: check if max_of_them and min_of_them gets initialized
        float one = 0.0, two, max_of_them = 0.0, min_of_them = 0.0;

        y = 0; // calculate one circle

        for (x = 0; x < xdim; x++)
        {
            if (x > 0)
                two = one;
            else
                two = zx_line_with_cylinder(xunit[x + xind_min]);

            one = zx_line_with_cylinder(xunit[x + xind_min + 1]);

            //  cout << x+xind_min << ": " << one << " " << two << endl;
            if (is_between(zmin, one, zmax) || is_between(zmin, two, zmax))
            {
                if (one > two)
                {
                    max_of_them = one;
                    min_of_them = two;
                }
                else
                {
                    max_of_them = two;
                    min_of_them = one;
                }

                cutting_cubes[x][y].upper = find(max_of_them, FINDZ, UP, true);
                cutting_cubes[x][y].lower = find(min_of_them, FINDZ, DOWN, true);
            }
            //
            // use the results for the symmetrical part
            //
            if (is_between(zmin, 2 * startz - min_of_them, zmax) || is_between(zmin, 2 * startz - max_of_them, zmax))
            {
                sym_cutting_cubes[x][y].upper = find(2 * startz - min_of_them, FINDZ, UP, true);
                sym_cutting_cubes[x][y].lower = find(2 * startz - max_of_them, FINDZ, DOWN, true);
            }
        }

        //
        // copy our results
        //

        for (y = 1; y < ydim; y++)
        {
            for (x = 0; x < xdim; x++)
            {
                cutting_cubes[x][y].lower = cutting_cubes[x][0].lower;
                cutting_cubes[x][y].upper = cutting_cubes[x][0].upper;
                sym_cutting_cubes[x][y].lower = sym_cutting_cubes[x][0].lower;
                sym_cutting_cubes[x][y].upper = sym_cutting_cubes[x][0].upper;
            }
        }

        /* for( j=0; j<xdim; j++ )
       {
       cout << j << " :" << cutting_cubes[j][0].lower <<":"<< cutting_cubes[j][0].upper << endl;
       cout << "sym  "<<j << " :" << sym_cutting_cubes[j][0].lower <<":"<< sym_cutting_cubes[j][0].upper << endl;
       } */
    }

    //
    //  cylinder z  ( just a copy of the last procedure; z plays the role of y in the last procedure )
    //

    else if (option == 4)
    {
        xunit = x_in;
        yunit = y_in;
        zunit = z_in;

        xorigmin = xunit[0];
        xorigmax = xunit[x_size - 1];
        yorigmin = yunit[0];
        yorigmax = yunit[y_size - 1];
        zorigmin = zunit[0];
        zorigmax = zunit[z_size - 1];

        xmin = startx - radius;
        xmax = startx + radius;
        ymin = starty - radius;
        ymax = starty + radius;
        zmin = zorigmin;
        zmax = zorigmax;

        if (radius <= xorigmax)
            xind_max = find(xmax, FINDX, UP, false);
        else
            xind_max = x_size - 1;
        if (-radius >= xorigmin)
            xind_min = find(xmin, FINDX, DOWN, false);
        else
            xind_min = 0;
        if (radius <= yorigmax)
            yind_max = find(ymax, FINDY, DOWN, false);
        else
            yind_max = z_size - 1;
        if (-radius >= zorigmin)
            yind_min = find(ymin, FINDY, DOWN, false);
        else
            yind_min = 0;

        if (xind_min > xind_max) // there is no cutting point
            xind_min = xind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;
        if (yind_min > yind_max)
            yind_min = yind_max = 0;

        xdim = xind_max - xind_min + 1;
        ydim = yind_max - yind_min + 1;
        zdim = z_size;

        zind_min = 0;
        zind_max = z_size - 1;

        //  cout << zmin << " " << zmax << " " << ymin << " " << ymax << " " << endl;
        //  cout << zind_min << " " << zind_max << " " << yind_min << " " << yind_max << " " << endl;
        //  cout << "Dim: " << xdim << " " << ydim << " " << zdim << endl;

        cutting_cubes = new border *[zdim];
        _SpaghettiAntidot.insert(pair<border **, int>(cutting_cubes, zdim));
        memset(cutting_cubes, '\0', zdim * sizeof(border *));
        ERR0(cutting_cubes == NULL, "cannot allocate memory", return;);
        sym_cutting_cubes = new border *[zdim];
        _SpaghettiAntidot.insert(pair<border **, int>(sym_cutting_cubes, zdim));
        memset(sym_cutting_cubes, '\0', zdim * sizeof(border *));
        ERR0(sym_cutting_cubes == NULL, "cannot allocate memory", return;);

        //
        // for small data sets, don't worry
        //

        int z, y;
        if (xdim <= TOO_SMALL || ydim <= TOO_SMALL || zdim <= TOO_SMALL)
        {
            for (z = 0; z < zdim; z++)
            {
                cutting_cubes[z] = new border[ydim];
                ERR0(cutting_cubes[z] == NULL, "cannot allocate memory", return;);
                sym_cutting_cubes[z] = new border[ydim];
                ERR0(sym_cutting_cubes[z] == NULL, "cannot allocate memory", return;);
                for (y = 0; y < ydim; y++)
                {
                    cutting_cubes[z][y].lower = zind_min;
                    cutting_cubes[z][y].upper = zind_max;
                    sym_cutting_cubes[z][y].upper = -1;
                    sym_cutting_cubes[z][y].lower = z_size + 1;
                }
            }
            return;
        }

        //
        // initialize array
        //

        for (i = 0; i < zdim; i++)
        {
            cutting_cubes[i] = new border[ydim];
            ERR0(cutting_cubes[i] == NULL, "cannot allocate memory", return;);
            sym_cutting_cubes[i] = new border[ydim];
            ERR0(sym_cutting_cubes[i] == NULL, "cannot allocate memory", return;);

            for (j = 0; j < ydim; j++)
            {
                cutting_cubes[i][j].upper = -1;
                cutting_cubes[i][j].lower = z_size + 1;
                sym_cutting_cubes[i][j].upper = -1;
                sym_cutting_cubes[i][j].lower = z_size + 1;
            }
        }

        // XXX: check if max_of_them and min_of_them gets initialized
        float one = 0.0, two, max_of_them = 0.0, min_of_them = 0.0;

        z = 0; // calculate one circle

        for (y = 0; y < ydim; y++)
        {
            if (y > 0)
                two = one;
            else
                two = xy_line_with_cylinder(yunit[y + yind_min]);

            one = xy_line_with_cylinder(yunit[y + yind_min + 1]);

            //   cout << y+yind_min << ": " << one << " " << two << endl;
            if (is_between(xmin, one, xmax) || is_between(xmin, two, xmax))
            {
                if (one > two)
                {
                    max_of_them = one;
                    min_of_them = two;
                }
                else
                {
                    max_of_them = two;
                    min_of_them = one;
                }

                cutting_cubes[z][y].upper = find(max_of_them, FINDX, UP, true);
                cutting_cubes[z][y].lower = find(min_of_them, FINDX, DOWN, true);
            }
            //
            // use the results for the symmetrical part
            //
            if (is_between(xmin, 2 * startx - min_of_them, xmax) || is_between(xmin, 2 * startx - max_of_them, xmax))
            {
                sym_cutting_cubes[z][y].upper = find(2 * startx - min_of_them, FINDX, UP, true);
                sym_cutting_cubes[z][y].lower = find(2 * startx - max_of_them, FINDX, DOWN, true);
            }
        }

        //
        // copy our results
        //

        for (z = 1; z < zdim; z++)
        {
            for (y = 0; y < ydim; y++)
            {
                cutting_cubes[z][y].lower = cutting_cubes[0][y].lower;
                cutting_cubes[z][y].upper = cutting_cubes[0][y].upper;
                sym_cutting_cubes[z][y].lower = sym_cutting_cubes[0][y].lower;
                sym_cutting_cubes[z][y].upper = sym_cutting_cubes[0][y].upper;
            }
        }

        /* for( j=0; j<ydim; j++ )
        {
        cout << j << " :" << cutting_cubes[0][j].lower <<":"<< cutting_cubes[0][j].upper << endl;
        cout << "sym  "<<j << " :" << sym_cutting_cubes[0][j].lower <<":"<< sym_cutting_cubes[0][j].upper << endl;
        } */
    }
}

bool RECT_Plane::createPlane()
{
    int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int i;
    int node_list[8];
    int x_add[] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    int y_add[] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    int z_add[] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    int *firstvertex;
    cutting_info *C_Info;

    int num_cuts = 0;

    if (option < 4 && (option > 0 || (option == 0 && CHECK_Z)))
    {
        add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, xind_min * y_size * z_size);
        //    for(ii=0; ii<x_size-1;ii++)
        for (ii = xind_min; ii < x_size - 1 && ii <= xind_max; ii++)
        {
            add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, yind_min * z_size);
            //   for(jj=0; jj<y_size-1;jj++)
            for (jj = yind_min; jj < y_size - 1 && jj <= yind_max; jj++)
            {
                //
                // calculate the symmetrical part
                //

                if ((option == 1 || option == 2 || option == 3) && sym_cutting_cubes[ii - xind_min][jj - yind_min].upper >= sym_cutting_cubes[ii - xind_min][jj - yind_min].lower)
                {
                    int store[8];
                    store[0] = *n_1;
                    store[1] = *n_2;
                    store[2] = *n_3;
                    store[3] = *n_4;
                    store[4] = *n_5;
                    store[5] = *n_6;
                    store[6] = *n_7;
                    store[7] = *n_8;
                    add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, sym_cutting_cubes[ii - xind_min][jj - yind_min].lower);

                    for (kk = sym_cutting_cubes[ii - xind_min][jj - yind_min].lower; kk < z_size - 1 &&
                                                                                     //      for(kk=0;kk<z_size-1;kk++)
                                                                                     kk <= sym_cutting_cubes[ii - xind_min][jj - yind_min].upper;
                         kk++)
                    {
                        /* cout<< ii << "," << jj <<"," << kk <<": "<< *n_1 << " " << *n_2 << " " << *n_3 << " " <<*n_4 << " "
                      <<*n_5 << " " << *n_6 << " " <<
                      *n_7<< " " << *n_8 << " " <<  endl; */
                        // num_cuts++;

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
                            num_triangles += numIntersections - 2;
                            firstvertex = vertex;
                            for (i = 0; i < numIntersections; i++)
                            {
                                n1 = node_list[*polygon_nodes];
                                n2 = node_list[*(polygon_nodes + 1)];
                                if (i > 2)
                                {
                                    *vertex++ = *firstvertex;
                                    *vertex = *(vertex - 2);
                                    vertex++;
                                }
                                if (n1 < n2)
                                    add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)]);
                                else
                                    add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
                                polygon_nodes += 2;
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
                    *n_1 = store[0];
                    *n_2 = store[1];
                    *n_3 = store[2];
                    *n_4 = store[3];
                    *n_5 = store[4];
                    *n_6 = store[5];
                    *n_7 = store[6];
                    *n_8 = store[7];
                }

                //
                // calculate the normal part
                //

                if (cutting_cubes[ii - xind_min][jj - yind_min].upper >= cutting_cubes[ii - xind_min][jj - yind_min].lower)
                {
                    add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, cutting_cubes[ii - xind_min][jj - yind_min].lower);

                    for (kk = cutting_cubes[ii - xind_min][jj - yind_min].lower; kk < z_size - 1 &&
                                                                                 //      for(kk=0;kk<z_size-1;kk++)
                                                                                 kk <= cutting_cubes[ii - xind_min][jj - yind_min].upper;
                         kk++)
                    {
                        /* cout<< ii << "," << jj <<"," << kk <<": "<< *n_1 << " " << *n_2 << " " << *n_3 << " " <<*n_4 << " "
                      <<*n_5 << " " << *n_6 << " " <<
                      *n_7<< " " << *n_8 << " " <<  endl; */
                        // num_cuts++;
                        //                  if( 1)                          //ii==5 && jj==4 )
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
                                num_triangles += numIntersections - 2;
                                firstvertex = vertex;
                                for (i = 0; i < numIntersections; i++)
                                {
                                    n1 = node_list[*polygon_nodes];
                                    n2 = node_list[*(polygon_nodes + 1)];
                                    if (i > 2)
                                    {
                                        *vertex++ = *firstvertex;
                                        *vertex = *(vertex - 2);
                                        vertex++;
                                    }
                                    if (n1 < n2)
                                        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)]);
                                    else
                                        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
                                    polygon_nodes += 2;
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
                    if (kk < z_size - 1)
                        add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, z_size - 1 - cutting_cubes[ii - xind_min][jj - yind_min].upper);
                    else
                    {
                        (*n_1)++;
                        (*n_2)++;
                        (*n_3)++;
                        (*n_4)++;
                        (*n_5)++;
                        (*n_6)++;
                        (*n_7)++;
                        (*n_8)++;
                    }
                }
                else
                    add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, z_size);
            }
            if (jj < y_size - 1)
                add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, (y_size - 1 - yind_max) * z_size);
            else
            {
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

        // char buf[100];
        // sprintf(buf, "%d cuts made", num_cuts );
        // Covise::sendInfo(buf);
    }

    //
    // cutting_cubes array x coordinates
    //

    else // cylinder-z or plane with small normal_z
    {
        add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, xind_min * y_size * z_size);
        //    for(ii=0; ii<x_size-1;ii++)
        for (ii = xind_min; ii < x_size - 1 && ii <= xind_max; ii++)
        {
            add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, yind_min * z_size);
            //   for(jj=0; jj<y_size-1;jj++)
            for (jj = yind_min; jj <= yind_max && jj < y_size - 1; jj++)
            {
                add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, zind_min);
                for (kk = zind_min; kk <= zind_max && kk < z_size - 1; kk++)
                {
                    if ((option == 4 && is_between((float)sym_cutting_cubes[kk - zind_min][jj - yind_min].lower, (float)ii, (float)sym_cutting_cubes[kk - zind_min][jj - yind_min].upper))
                        || is_between((float)cutting_cubes[kk - zind_min][jj - yind_min].lower, (float)ii, (float)cutting_cubes[kk - zind_min][jj - yind_min].upper))
                    {
                        /* cout<< ii << "," << jj <<"," << kk <<": "<< *n_1 << " " << *n_2 << " " << *n_3 << " " <<*n_4 << " "
                    <<*n_5 << " " << *n_6 << " " <<
                    *n_7<< " " << *n_8 << " " <<  endl; */
                        num_cuts++;
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
                                num_triangles += numIntersections - 2;
                                firstvertex = vertex;
                                for (i = 0; i < numIntersections; i++)
                                {
                                    n1 = node_list[*polygon_nodes];
                                    n2 = node_list[*(polygon_nodes + 1)];
                                    if (i > 2)
                                    {
                                        *vertex++ = *firstvertex;
                                        *vertex = *(vertex - 2);
                                        vertex++;
                                    }
                                    if (n1 < n2)
                                        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)]);
                                    else
                                        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
                                    polygon_nodes += 2;
                                }
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
                if (kk < z_size - 1)
                    add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, z_size - 1 - zind_max);
                else
                {
                    (*n_1)++;
                    (*n_2)++;
                    (*n_3)++;
                    (*n_4)++;
                    (*n_5)++;
                    (*n_6)++;
                    (*n_7)++;
                    (*n_8)++;
                }
            }
            if (jj < y_size - 1)
                add_to_corner(n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, (y_size - 1 - yind_max) * z_size);
            else
            {
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
        return true;
    }

    map<border **, int>::iterator p_cutting_cubes = _SpaghettiAntidot.find(cutting_cubes);
    if (p_cutting_cubes != _SpaghettiAntidot.end())
    {
        int i;
        for (i = 0; i < p_cutting_cubes->second; ++i)
        {
            delete[] cutting_cubes[i];
        }
        delete[] cutting_cubes;
    }

    map<border **, int>::iterator p_sym_cutting_cubes = _SpaghettiAntidot.find(sym_cutting_cubes);
    if (p_sym_cutting_cubes != _SpaghettiAntidot.end())
    {
        int i;
        for (i = 0; i < p_sym_cutting_cubes->second; ++i)
        {
            delete[] sym_cutting_cubes[i];
        }
        delete[] sym_cutting_cubes;
    }

    return true;
}

inline void RECT_Plane::add_to_corner(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h, int add)
{
    (*a) += add;
    (*b) += add;
    (*c) += add;
    (*d) += add;
    (*e) += add;
    (*f) += add;
    (*g) += add;
    (*h) += add;
}

//=============================== POLYHEDRON_Plane ===============================

POLYHEDRON_Plane::POLYHEDRON_Plane(int n_elem, int n_nodes, int Type,
                                   int *p_el, int *p_cl, int *p_tl,
                                   float *p_x_in, float *p_y_in, float *p_z_in,
                                   float *p_s_in, unsigned char *p_bs_in, float *p_i_in,
                                   float *p_u_in, float *p_v_in, float *p_w_in,
                                   const coDoStructuredGrid *p_sgrid_in,
                                   const coDoUnstructuredGrid *p_grid_in,
                                   int maxPoly,
                                   float planei_, float planej_, float planek_,
                                   float startx_, float starty_, float startz_, float myDistance_,
                                   float radius_, int gennormals_, int option_,
                                   int genstrips_, char *ib)
    : Plane(n_elem, /*n_nodes*/ 0, Type, p_el, p_cl, p_tl,
            p_x_in, p_y_in, p_z_in, p_s_in, p_bs_in, p_i_in,
            p_u_in, p_v_in, p_w_in, p_sgrid_in, p_grid_in,
            -1.0, maxPoly,
            planei_, planej_, planek_,
            startx_, starty_, startz_,
            myDistance_, radius_, gennormals_, option_, genstrips_, ib)
{
    (void)n_nodes;

    S_Data = NULL;
    I_Data = NULL;
    V_Data_U = NULL;
    V_Data_V = NULL;
    V_Data_W = NULL;
    iblank = NULL;

    num_elem_out = 0;
    num_conn_out = 0;
    num_coord_out = 0;
    elem_out = NULL;
    conn_out = NULL;
    x_coord_out = NULL;
    y_coord_out = NULL;
    z_coord_out = NULL;
    data_out_s = NULL;
    data_out_u = NULL;
    data_out_v = NULL;
    data_out_w = NULL;
};

POLYHEDRON_Plane::~POLYHEDRON_Plane()
{
    if (x_coord_out)
        delete[] x_coord_out;
    if (y_coord_out)
        delete[] y_coord_out;
    if (z_coord_out)
        delete[] z_coord_out;

    if (elem_out)
        delete[] elem_out;
    if (conn_out)
        delete[] conn_out;

    if (data_out_s)
        delete[] data_out_s;
    if (data_out_u)
        delete[] data_out_u;
    if (data_out_v)
        delete[] data_out_v;
    if (data_out_w)
        delete[] data_out_w;
}

bool POLYHEDRON_Plane::createPlane()
{
    /**************/
    /* Input Data  */
    /**************/

    int num_elem_in;
    int num_conn_in;
    int num_coord_in;

    int *elem_in;
    int *conn_in;
    int *type_list;

    float distance;
    float normal_vector_x;
    float normal_vector_y;
    float normal_vector_z;

    float *x_coord_in;
    float *y_coord_in;
    float *z_coord_in;

    /****************************/
    /* Sampling Plane Variables  */
    /****************************/

    bool cell_intersection;

    float p;
    //float d;
    float point_plane_distance;

    EDGE_VECTOR unit_normal_vector;
    EDGE_VECTOR test_point;

    /*********************/
    /* Auxiliary Variables */
    /*********************/

    bool start_vertex_set;

    int i;
    int j;
    int elem_count;
    int next_elem_index;
    int start_vertex;
    int new_elem_address;
    int new_conn_address;
    int first_sign;
    int current_sign;

    int *temp_elem_list;
    int *temp_conn_list;

    float *new_x_coord_in;
    float *new_y_coord_in;
    float *new_z_coord_in;

    float *temp_s_data_in(NULL); // used fo scalar variables

    float *temp_u_data_in(NULL);
    float *temp_v_data_in(NULL);
    float *temp_w_data_in(NULL);

    std::vector<float> s_data; // used fo scalar variables

    std::vector<float> u_data_vector;
    std::vector<float> v_data_vector;
    std::vector<float> w_data_vector;

    vector<int> temp_elem_in;
    vector<int> temp_conn_in;
    vector<int> new_temp_elem_in;
    vector<int> new_temp_conn_in;
    vector<int> temp_vertex_list;
    vector<int> temp_intersec_list;
    vector<int> temp_elem_out;
    vector<int> temp_conn_out;

    vector<float> temp_s_data_out;
    vector<float> temp_u_data_out; // was used fo scalar variables
    vector<float> temp_v_data_out;
    vector<float> temp_w_data_out;
    vector<float> temp_x_coord_out;
    vector<float> temp_y_coord_out;
    vector<float> temp_z_coord_out;

    /*********************/
    /* Contour Variables */
    /*********************/

    CONTOUR capping_contour;
    PLANE_EDGE_INTERSECTION_VECTOR intsec_vector;

    /* Avoid Unnecessary Reallocations */
    capping_contour.ring.reserve(15);
    capping_contour.ring_index.reserve(5);
    capping_contour.polyhedron_faces.reserve(15);
    intsec_vector.reserve(15);

    distance = myDistance;
    normal_vector_x = planei;
    normal_vector_y = planej;
    normal_vector_z = planek;

    /*********************************/
    /* Case 1 --> Null Normal Vector  */
    /*********************************/

    if (normal_vector_x == 0 && normal_vector_y == 0 && normal_vector_z == 0)
    {
        Covise::sendInfo("Warning: null normal vector detected");

        /**************************/
        /* Generate NULL Output  */
        /**************************/

        num_coord_out = 0; //intsec_vector.size();
        num_elem_out = 0; //capping_contour.ring_index.size();
        num_conn_out = 0; //capping_contour.ring.size();
    }

    /*************************************/
    /* Case 2 --> Regular Normal Vector  */
    /*************************************/

    else
    {
        /*****************************************/
        /* Sampling Plane Parameters Calculation */
        /*****************************************/

        /* Unit Normal Vector Coordinates */
        float len = sqrt((normal_vector_x * normal_vector_x) + (normal_vector_y * normal_vector_y) + (normal_vector_z * normal_vector_z));
        unit_normal_vector.x = normal_vector_x / len;
        unit_normal_vector.y = normal_vector_y / len;
        unit_normal_vector.z = normal_vector_z / len;

        /* Distance to Origin */
        p = (-1) * distance;

        /* Plane Equation "d" Parameter*/
        //d = p*len;

        // calculate two base vectors on the plane (orthogonal, length 1)
        EDGE_VECTOR plane_base_x;
        EDGE_VECTOR plane_base_y;
        if (fabs(unit_normal_vector.x) > fabs(unit_normal_vector.y))
        {
            plane_base_x.x = 0.0;
            plane_base_x.y = 1.0;
            plane_base_x.z = 0.0;
        }
        else
        {
            plane_base_x.x = 1.0;
            plane_base_x.y = 0.0;
            plane_base_x.z = 0.0;
        }
        plane_base_y = cross_product(plane_base_x, unit_normal_vector);
        len = (float)length(plane_base_y);
        plane_base_y.x /= len;
        plane_base_y.y /= len;
        plane_base_y.z /= len;
        plane_base_x = cross_product(plane_base_y, unit_normal_vector);

        if (grid_in->isType("UNSGRD"))
        {
            coDoUnstructuredGrid *grid = (coDoUnstructuredGrid *)grid_in;
            grid->getAddresses(&elem_in, &conn_in, &x_coord_in, &y_coord_in, &z_coord_in);
            grid->getGridSize(&num_elem_in, &num_conn_in, &num_coord_in);
            grid->getTypeList(&type_list);

            temp_elem_out.clear();
            temp_conn_out.clear();
            temp_s_data_out.clear();
            temp_u_data_out.clear();
            temp_v_data_out.clear();
            temp_w_data_out.clear();
            temp_x_coord_out.clear();
            temp_y_coord_out.clear();
            temp_z_coord_out.clear();

            new_elem_address = 0;
            new_conn_address = 0;
            first_sign = 0;
            current_sign = 2;

            for (elem_count = 0; elem_count < num_elem_in; elem_count++)
            {
                /*  Avoid additional calculations if the cell is not cut by the sampling plane */
                cell_intersection = false;
                next_elem_index = (elem_count < num_elem_in - 1) ? elem_in[elem_count + 1] : num_conn_in;
                for (i = elem_in[elem_count]; i < next_elem_index; i++)
                {
                    test_point.x = x_coord_in[conn_in[i]];
                    test_point.y = y_coord_in[conn_in[i]];
                    test_point.z = z_coord_in[conn_in[i]];

                    point_plane_distance = (float)dot_product(unit_normal_vector, test_point) + p;

                    if (point_plane_distance >= 0.0)
                    {
                        if (i == elem_in[elem_count])
                        {
                            first_sign = 1;
                        }

                        current_sign = 1;
                    }

                    if (point_plane_distance < 0.0)
                    {
                        if (i == elem_in[elem_count])
                        {
                            first_sign = -1;
                        }

                        current_sign = -1;
                    }

                    if (first_sign != current_sign)
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
                    temp_intersec_list.clear();
                    new_temp_elem_in.clear();
                    new_temp_conn_in.clear();
                    capping_contour.ring.clear();
                    capping_contour.ring_index.clear();
                    capping_contour.polyhedron_faces.clear();
                    intsec_vector.clear();
                    s_data.clear();
                    u_data_vector.clear();
                    v_data_vector.clear();
                    w_data_vector.clear();
                    start_vertex_set = false;

                    switch (type_list[elem_count])
                    {
                    case TYPE_POLYHEDRON:

                        /* Construct DO_Polygons Element and Connectivity Lists */
                        for (j = elem_in[elem_count]; j < next_elem_index; j++)
                        {
                            if (j == elem_in[elem_count] && start_vertex_set == false)
                            {
                                start_vertex = conn_in[elem_in[elem_count]];
                                temp_elem_in.push_back((int)temp_conn_in.size());
                                temp_conn_in.push_back(start_vertex);
                                start_vertex_set = true;
                            }

                            if (j > elem_in[elem_count] && start_vertex_set == true)
                            {
                                if (conn_in[j] != start_vertex)
                                {
                                    temp_conn_in.push_back(conn_in[j]);
                                }

                                else
                                {
                                    start_vertex_set = false;
                                    continue;
                                }
                            }

                            if (j > elem_in[elem_count] && start_vertex_set == false)
                            {
                                start_vertex = conn_in[j];
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
                        for (j = elem_in[elem_count]; j < next_elem_index; j++)
                        {
                            if (j == elem_in[elem_count])
                            {
                                temp_elem_in.push_back((int)temp_conn_in.size());
                            }

                            if ((j - elem_in[elem_count]) == 4)
                            {
                                temp_elem_in.push_back(4);
                            }

                            temp_conn_in.push_back(conn_in[j]);
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
                        for (j = elem_in[elem_count]; j < next_elem_index; j++)
                        {
                            if (j == elem_in[elem_count])
                            {
                                temp_elem_in.push_back((int)temp_conn_in.size());
                            }

                            if ((j - elem_in[elem_count]) == 3)
                            {
                                temp_elem_in.push_back(3);
                            }

                            temp_conn_in.push_back(conn_in[j]);
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
                        for (j = elem_in[elem_count]; j < next_elem_index; j++)
                        {
                            if (j == elem_in[elem_count])
                            {
                                temp_elem_in.push_back((int)temp_conn_in.size());
                            }

                            if ((j - elem_in[elem_count]) == 4)
                            {
                                temp_elem_in.push_back(4);
                            }

                            temp_conn_in.push_back(conn_in[j]);
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
                        for (j = elem_in[elem_count]; j < next_elem_index; j++)
                        {
                            if (j == elem_in[elem_count])
                            {
                                temp_elem_in.push_back((int)temp_conn_in.size());
                            }

                            if ((j - elem_in[elem_count]) == 3)
                            {
                                temp_elem_in.push_back(3);
                            }

                            temp_conn_in.push_back(conn_in[j]);
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

                    temp_elem_list = new int[temp_elem_in.size()];
                    temp_conn_list = new int[temp_conn_in.size()];
                    new_x_coord_in = new float[temp_vertex_list.size()];
                    new_y_coord_in = new float[temp_vertex_list.size()];
                    new_z_coord_in = new float[temp_vertex_list.size()];

                    for (i = 0; i < temp_elem_in.size(); i++)
                    {
                        temp_elem_list[i] = temp_elem_in[i];
                    }

                    for (i = 0; i < new_temp_conn_in.size(); i++)
                    {
                        temp_conn_list[i] = new_temp_conn_in[i];
                    }

                    /* Construct New Set of Coordinates */
                    for (i = 0; i < temp_vertex_list.size(); i++)
                    {
                        new_x_coord_in[i] = x_coord_in[temp_vertex_list[i]];
                        new_y_coord_in[i] = y_coord_in[temp_vertex_list[i]];
                        new_z_coord_in[i] = z_coord_in[temp_vertex_list[i]];
                    }

                    if ((Datatype == 1) || (Datatype == 2)) // scalar data or scalar & vector data
                    {
                        temp_s_data_in = new float[temp_vertex_list.size()];
                        if (bs_in)
                        {
                            for (i = 0; i < temp_vertex_list.size(); i++)
                            {
                                temp_s_data_in[i] = bs_in[temp_vertex_list[i]] / 255.f;
                            }
                        }
                        else
                        {
                            for (i = 0; i < temp_vertex_list.size(); i++)
                            {
                                temp_s_data_in[i] = s_in[temp_vertex_list[i]];
                            }
                        }
                    }
                    else
                    {
                        temp_u_data_in = new float[temp_vertex_list.size()];
                        temp_v_data_in = new float[temp_vertex_list.size()];
                        temp_w_data_in = new float[temp_vertex_list.size()];
                        for (i = 0; i < temp_vertex_list.size(); i++)
                        {
                            temp_u_data_in[i] = u_in[temp_vertex_list[i]];
                            temp_v_data_in[i] = v_in[temp_vertex_list[i]];
                            temp_w_data_in[i] = w_in[temp_vertex_list[i]];
                        }
                    }

                    create_contour(new_x_coord_in, new_y_coord_in, new_z_coord_in, temp_elem_list, temp_conn_list, (int)temp_vertex_list.size(), (int)temp_elem_in.size(), (int)new_temp_conn_in.size(), temp_s_data_in, temp_u_data_in, temp_v_data_in, temp_w_data_in, capping_contour, intsec_vector, s_data, u_data_vector, v_data_vector, w_data_vector, unit_normal_vector, plane_base_x, plane_base_y, p);

                    if (intsec_vector.size() >= 3)
                    {
                        /* Construct Partial Output */
                        if (temp_conn_out.size() == 0)
                        {
                            for (i = 0; i < capping_contour.ring_index.size(); i++)
                            {
                                temp_elem_out.push_back(capping_contour.ring_index[i] + new_elem_address);
                            }

                            for (i = 0; i < capping_contour.ring.size(); i++)
                            {
                                temp_conn_out.push_back(capping_contour.ring[i] + new_conn_address);
                            }

                            new_elem_address = (int)temp_conn_out.size();
                            new_conn_address += (int)intsec_vector.size();
                        }

                        else
                        {
                            for (i = 0; i < capping_contour.ring_index.size(); i++)
                            {
                                temp_elem_out.push_back(capping_contour.ring_index[i] + new_elem_address);
                            }

                            for (i = 0; i < capping_contour.ring.size(); i++)
                            {
                                temp_conn_out.push_back(capping_contour.ring[i] + new_conn_address);
                            }

                            new_elem_address = (int)temp_conn_out.size();
                            new_conn_address += (int)intsec_vector.size();
                        }

                        for (i = 0; i < s_data.size(); i++)
                        {
                            if ((Datatype == 1) || (Datatype == 2)) // scalar data or scalar & vector data
                            {
                                temp_s_data_out.push_back(s_data[i]);
                            }
                        }

                        for (i = 0; i < u_data_vector.size(); i++)
                        {
                            if ((Datatype == 0) || (Datatype == 2)) // vector data or scalar & vector data
                            {
                                temp_u_data_out.push_back(u_data_vector[i]);
                                temp_v_data_out.push_back(v_data_vector[i]);
                                temp_w_data_out.push_back(w_data_vector[i]);
                            }
                        }

                        for (i = 0; i < intsec_vector.size(); i++)
                        {
                            temp_x_coord_out.push_back((float)intsec_vector[i].intersection.x);
                            temp_y_coord_out.push_back((float)intsec_vector[i].intersection.y);
                            temp_z_coord_out.push_back((float)intsec_vector[i].intersection.z);
                        }
                    }

                    delete[] temp_elem_list;
                    delete[] temp_conn_list;
                    delete[] new_x_coord_in;
                    delete[] new_y_coord_in;
                    delete[] new_z_coord_in;
                    if ((Datatype == 1) || (Datatype == 2)) // scalar data or scalar & vector data
                        delete[] temp_s_data_in;
                    if ((Datatype == 0) || (Datatype == 2)) // vector data or scalar & vector data
                    {
                        delete[] temp_u_data_in;
                        delete[] temp_v_data_in;
                        delete[] temp_w_data_in;
                    }
                }
            }

            /* Generate Output */
            if (temp_conn_out.size() == 0)
            {
                /**************************/
                /* Generate NULL Output  */
                /**************************/

                num_coord_out = 0; //temp_data_out.size();
                num_elem_out = 0; //temp_elem_out.size();
                num_conn_out = 0; //temp_conn_out.size();
            }

            else
            {
                x_coord_out = new float[temp_x_coord_out.size()];
                y_coord_out = new float[temp_y_coord_out.size()];
                z_coord_out = new float[temp_z_coord_out.size()];

                elem_out = new int[temp_elem_out.size()];
                conn_out = new int[temp_conn_out.size()];

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

                if ((Datatype == 1) || (Datatype == 2))
                {
                    data_out_s = new float[temp_s_data_out.size()];
                    for (i = 0; i < temp_s_data_out.size(); i++)
                    {
                        data_out_s[i] = temp_s_data_out[i];
                    }
                }
                if ((Datatype == 0) || (Datatype == 2))
                {
                    data_out_u = new float[temp_u_data_out.size()];
                    data_out_v = new float[temp_v_data_out.size()];
                    data_out_w = new float[temp_w_data_out.size()];
                    for (i = 0; i < temp_u_data_out.size(); i++)
                    {
                        data_out_u[i] = temp_u_data_out[i];
                        data_out_v[i] = temp_v_data_out[i];
                        data_out_w[i] = temp_w_data_out[i];
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
                temp_s_data_out.clear();
                temp_u_data_out.clear();
                temp_v_data_out.clear();
                temp_w_data_out.clear();
            }
        }
    }

    return true;
}

// deprecated
coDistributedObject *POLYHEDRON_Plane::create_data_output(vector<float> &data_vector, const char *data_obj_name)
{
    (void)data_vector;
    (void)data_obj_name;
    return NULL;
}

void POLYHEDRON_Plane::createcoDistributedObjects(const char *Data_name_scal, const char *Data_name_vect, const char *Normal_name, const char *Triangle_name, AttributeContainer &gridAttrs, AttributeContainer &dataAttrs)
{
    float *u_out, *v_out, *w_out;

    s_data_out = NULL;
    v_data_out = NULL;
    polygons_out = NULL;
    strips_out = NULL;
    normals_out = NULL;

    (void)dataAttrs;
    (void)gridAttrs;

    // data per vertex is currently assumed
    if ((Data_name_scal) || (Data_name_vect))
    {
        if ((Datatype == 1) || (Datatype == 2)) // scalar data or scalar & vector data
        {
            if (num_coord_out == 0)
                s_data_out = new coDoFloat(Data_name_scal, 0);
            else
                s_data_out = new coDoFloat(Data_name_scal, num_coord_out, data_out_s);

            if (!s_data_out->objectOk())
            {
                int n = 0;
                const char **attr = NULL, **setting = NULL;
                if (grid_in)
                    n = grid_in->getAllAttributes(&attr, &setting);
                //             			else if(sgrid_in)
                //                				n = sgrid_in->getAllAttributes(&attr, &setting);
                //             			else if(ugrid_in)
                //                				n = ugrid_in->getAllAttributes(&attr, &setting);
                //             			else if(rgrid_in)
                //                				n = rgrid_in->getAllAttributes(&attr, &setting);
                if (n > 0)

                    s_data_out->addAttributes(n, attr, setting);

                Covise::sendError("ERROR: creation of data object 'dataOut' failed");
                return;
            }

            //          		dataAttrs.addAttributes(s_data_out);
        }
        if ((Datatype == 0) || (Datatype == 2)) // vector data or scalar & vector data
        {
            if (num_coord_out == 0)
                v_data_out = new coDoVec3(Data_name_vect, 0);
            else
                v_data_out = new coDoVec3(Data_name_vect, num_coord_out, data_out_u, data_out_v, data_out_w);

            if (!v_data_out->objectOk())
            {
                int n = 0;
                const char **attr, **setting;
                if (grid_in)
                    n = grid_in->getAllAttributes(&attr, &setting);
                //             			else if(sgrid_in)
                //                				n = sgrid_in->getAllAttributes(&attr, &setting);
                //             			else if(ugrid_in)
                //                				n = ugrid_in->getAllAttributes(&attr, &setting);
                //             			else if(rgrid_in)
                //                				n = rgrid_in->getAllAttributes(&attr, &setting);
                if (n > 0)
                    v_data_out->addAttributes(n, attr, setting);

                Covise::sendError("ERROR: creation of data object 'dataOut' failed");
                return;
            }

            //          		dataAttrs.addAttributes(v_data_out);
        }
    }

    polygons_out = new coDoPolygons(Triangle_name, num_coord_out, x_coord_out, y_coord_out, z_coord_out, num_conn_out, conn_out, num_elem_out, elem_out);

    if (polygons_out->objectOk())
    {
        //         	gridAttrs.addAttributes(polygons_out, Data_name);
        //          	if (num_coords)
        //          	{
        //                 	polygons_out->getAddresses(&u_out,&v_out,&w_out,&vl,&pl);
        //             		if(no_cut==0)                         // if there is a cut,
        //             		{
        //                			// otherwise coords_? have no meaning
        //                			// for the dummy...
        //                			memcpy(u_out,coords_x,num_coords*sizeof(float));
        //                			memcpy(v_out,coords_y,num_coords*sizeof(float));
        //                			memcpy(w_out,coords_z,num_coords*sizeof(float));
        //                			memcpy(vl,vertice_list,num_vertices*sizeof(int));
        //                			for(i=0;i<num_triangles;i++)
        //                   		   pl[i]=i*3;
        //             		}
        //          	}

        // 0?
        //          	polygons_out->addAttribute("vertexOrder","2");

        // we don't need this any more
        //char *DataIn 	=  Covise::get_object_name("dataIn");
        //polygons_out->addAttribute("DataObjectName",DataIn);

        if (polygons_out->getAttribute("COLOR") == 0)
            polygons_out->addAttribute("COLOR", "blue");

    } // objectOk
    else
    {
        Covise::sendError("ERROR: creation of polygonal object failed");
        return;
    }

    if (gennormals && Normal_name)
    {
        normals_out = new coDoVec3(Normal_name, num_coord_out);
        if (normals_out->objectOk())
        {
            normals_out->getAddresses(&u_out, &v_out, &w_out);

            // there is a real cut
            // and coords_? have physical meaning
            float pla[3], star[3];
            float rad = radius;
            pla[0] = planei;
            pla[1] = planej;
            pla[2] = planek;
            star[0] = startx;
            star[1] = starty;
            star[2] = startz;
            fill_normals(u_out, v_out, w_out,
                         coords_x, coords_y, coords_z, num_coord_out,
                         option, pla, rad, star);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'normalsOut' failed");
            return;
        }
    }
}

void POLYHEDRON_Plane::create_contour(float *x_coord_in, float *y_coord_in, float *z_coord_in, int *elem_in, int *conn_in,
                                      int num_coord_in, int num_elem_in, int num_conn_in,
                                      float *sdata_in, float *udata_in, float *vdata_in, float *wdata_in, CONTOUR &capping_contour, PLANE_EDGE_INTERSECTION_VECTOR &intsec_vector, vector<float> &s_data, vector<float> &u_data_vector, vector<float> &v_data_vector, vector<float> &w_data_vector, EDGE_VECTOR &unit_normal_vector, EDGE_VECTOR &plane_base_x, EDGE_VECTOR &plane_base_y, float p)
{

    (void)num_coord_in;
    (void)num_elem_in;

    /**************************************************/
    /* Calculation of  Edge Intersection with Slice Plane */
    /**************************************************/

    if (sdata_in)
        intsec_vector = calculate_intersections(sdata_in, vdata_in, wdata_in, num_elem_in, elem_in, num_conn_in, conn_in, x_coord_in, y_coord_in, z_coord_in, p, unit_normal_vector);
    else
        intsec_vector = calculate_intersections(udata_in, vdata_in, wdata_in, num_elem_in, elem_in, num_conn_in, conn_in, x_coord_in, y_coord_in, z_coord_in, p, unit_normal_vector);

    /***********************/
    /* Contour Generation  */
    /***********************/

    if (intsec_vector.size() >= 3)
    {
        //generate_capping_contour(capping_contour, intsec_vector, index_list, polygon_list, num_coord_in, num_conn_in, num_elem_in, elem_in, conn_in, data_vector_);
        if (sdata_in)
            generate_capping_contour(capping_contour, intsec_vector, plane_base_x, plane_base_y, s_data, v_data_vector, w_data_vector);
        else
            generate_capping_contour(capping_contour, intsec_vector, plane_base_x, plane_base_y, u_data_vector, v_data_vector, w_data_vector);
    }
}

//=============================== STR_Plane ===============================

bool STR_Plane::createPlane()
{
    int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int i;
    int node_list[8];
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    int *firstvertex;
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
                        num_triangles += numIntersections - 2;
                        firstvertex = vertex;
                        for (i = 0; i < numIntersections; i++)
                        {
                            n1 = node_list[*polygon_nodes];
                            n2 = node_list[*(polygon_nodes + 1)];
                            if (i > 2)
                            {
                                *vertex++ = *firstvertex;
                                *vertex = *(vertex - 2);
                                vertex++;
                            }
                            if (n1 < n2)
                                add_vertex(n1, n2);
                            else
                                add_vertex(n2, n1);
                            polygon_nodes += 2;
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

//============================= Isoline ================================

void Isoline::createIsoline(float isovalue)
{
    int i, *triangle, bitmap, n, n1, n2;
    node = plane->node_table;
    vertex = vertice_list;
    num_vertice = 0;
    num_coords = 0;
    coord_x = coords_x = new float[plane->num_triangles];
    coord_y = coords_y = new float[plane->num_triangles];
    coord_z = coords_z = new float[plane->num_triangles];
    for (i = 0; i < plane->num_coords; i++)
    {
        node->targets[0] = 0;
        // Calculate the myDistance of each node
        // to the Isoline
        node->dist = (plane->I_Data[i] - isovalue);
        node->side = (node->dist >= 0 ? 1 : 0);
        node++;
    }
    node = plane->node_table;
    for (i = 0; i < plane->num_triangles; i++)
    {
        triangle = plane->vertice_list + i * 3;
        bitmap = 0;
        for (n = 0; n < 3; n++)
            bitmap |= node[triangle[n]].side << n;
        if (Triangle_table[bitmap][3])
        {
            n1 = triangle[Triangle_table[bitmap][0]];
            n2 = triangle[Triangle_table[bitmap][1]];
            if (n1 < n2)
                add_vertex(n1, n2);
            else
                add_vertex(n2, n1);
            num_vertice++;
            n2 = triangle[Triangle_table[bitmap][2]];
            if (n1 < n2)
                add_vertex(n1, n2);
            else
                add_vertex(n2, n1);
            num_vertice++;
        }
    }
}

void Isoline::add_vertex(int n1, int n2)
{

    int *targets, *indices; // Pointers into the node_info structure
    float w1, w2;

    targets = node[n1].targets;
    indices = node[n1].vertice_list;

    int n = 0;
    while ((*targets) && (n < 11))
    {
        if (*targets == n2) // did we already calculate this vertex?
        {
            *vertex++ = *indices; // great! just put in the right index.
            return;
        }
        targets++;
        indices++;
        n++;
    }

    // remember the target we will calculate now

    *targets++ = n2;
    if (n < 11)
        *targets = 0;
    *indices = num_coords;
    *vertex++ = *indices;
    neighborlist[*indices] = 0;

    // Calculate the interpolation weights (linear interpolation)

    w2 = node[n1].dist / (node[n1].dist - node[n2].dist);
    w1 = 1.0f - w2;
    *coord_x++ = plane->coords_x[n1] * w1 + plane->coords_x[n2] * w2;
    *coord_y++ = plane->coords_y[n1] * w1 + plane->coords_y[n2] * w2;
    *coord_z++ = plane->coords_z[n1] * w1 + plane->coords_z[n2] * w2;
    num_coords++;
}

void Isoline::createcoDistributedObjects(const char *Line_name,
                                         coDistributedObject **Line_set,
                                         int currNumber,
                                         AttributeContainer &attribs)
{
    int i, n;
    coDistributedObject **Lines;
    coDoSet *ThisLines;
    char name[200];

    // create empty line objects if no lines are there
    if (num_isolines == 0)
    {
        Lines = new coDistributedObject *[2];

        sprintf(name, "%s_0", Line_name);
        Lines[0] = new coDoLines(name, 0, 0, 0);
        attribs.addAttributes(Lines[0]);
        if (!Lines[0]->getAttribute("COLOR"))
            Lines[0]->addAttribute("COLOR", "black");
        Lines[1] = NULL;
    }
    else
    {
        Lines = new coDistributedObject *[num_isolines + 1];
        Lines[num_isolines] = NULL;
    }

    for (i = 0; i < num_isolines; i++)
    {
        sprintf(name, "%s_%d", Line_name, plane->cur_line_elem);
        plane->cur_line_elem++;
        // add offset to isolines (z-buffer problem)
        if (option == 0)
        {
            for (n = 0; n < iso_numcoords[i]; n++)
            {
                iso_coords_x[i][n] += planei * offset;
                iso_coords_y[i][n] += planej * offset;
                iso_coords_z[i][n] += planek * offset;
            }
        }
        else if (option == 1)
        {
            for (n = 0; n < iso_numcoords[i]; n++)
            {
                iso_coords_x[i][n] += (iso_coords_x[i][n] - planei) * offset;
                iso_coords_y[i][n] += (iso_coords_y[i][n] - planej) * offset;
                iso_coords_z[i][n] += (iso_coords_z[i][n] - planek) * offset;
            }
        }
        else if (option == 2)
        {
            for (n = 0; n < iso_numcoords[i]; n++)
            {
                iso_coords_y[i][n] += (iso_coords_y[i][n]) * offset;
                iso_coords_z[i][n] += (iso_coords_z[i][n]) * offset;
            }
        }
        else if (option == 3)
        {
            for (n = 0; n < iso_numcoords[i]; n++)
            {
                iso_coords_x[i][n] += (iso_coords_x[i][n]) * offset;
                iso_coords_z[i][n] += (iso_coords_z[i][n]) * offset;
            }
        }
        else if (option == 4)
        {
            for (n = 0; n < iso_numcoords[i]; n++)
            {
                iso_coords_x[i][n] += (iso_coords_x[i][n]) * offset;
                iso_coords_y[i][n] += (iso_coords_y[i][n]) * offset;
            }
        }

        Lines[i] = new coDoLines(name, iso_numcoords[i], iso_coords_x[i], iso_coords_y[i], iso_coords_z[i],
                                 iso_numvert[i], iso_vl[i], iso_numlines[i], iso_ll[i]);
        attribs.addAttributes(Lines[i]);
        if (!Lines[i]->getAttribute("COLOR"))
            Lines[i]->addAttribute("COLOR", "black");

        Lines[i + 1] = NULL;
        delete[] iso_coords_x[i];
        delete[] iso_coords_y[i];
        delete[] iso_coords_z[i];
        delete[] iso_ll[i];
        delete[] iso_vl[i];

    } // numLines loop finished

    ThisLines = new coDoSet(Line_name, Lines);

    for (i = 0; i < num_isolines; i++)
        delete Lines[i];
    delete[] Lines;

    if (Line_set)
    {
        Line_set[currNumber] = ThisLines;
    }
    else
    {
        delete ThisLines;
    }
}

void Isoline::sortIsoline()
{
    int *vl, *ll, n, m, l, i;
    if (num_coords > 0)
    {
        iso_coords_x[num_isolines] = coords_x;
        iso_coords_y[num_isolines] = coords_y;
        iso_coords_z[num_isolines] = coords_z;
        iso_numcoords[num_isolines] = num_coords;
        iso_ll[num_isolines] = ll = new int[num_vertice / 2 + 2];
        iso_vl[num_isolines] = vl = new int[num_vertice];
        l = 0;
        ll[0] = 0;
        for (i = 0; i < num_vertice; i += 2)
        {
            if ((vertice_list[i] >= 0) && (vertice_list[i + 1] >= 0))
            {
                *vl++ = vertice_list[i];
                vertice_list[i] = -1; // just to avoid loops
                *vl++ = vertice_list[i + 1];
                n = i + 1;
                while ((vertice_list[n] >= 0) && ((m = neighborlist[vertice_list[n]]) > 0))
                {
                    neighborlist[vertice_list[n]] = 0;
                    if (m % 2)
                    {
                        if (vertice_list[m - 1] >= 0)
                        {
                            *vl++ = vertice_list[m - 1];
                            vertice_list[m] = -1;
                            n = m - 1;
                        }
                    }
                    else
                    {
                        if (vertice_list[m + 1] >= 0)
                        {
                            *vl++ = vertice_list[m + 1];
                            vertice_list[m] = -1;
                            n = m + 1;
                        }
                    }
                }
                l++;
                ll[l] = (int)(vl - iso_vl[num_isolines]);
            }
        }
        iso_numlines[num_isolines] = l;
        iso_numvert[num_isolines] = (int)(vl - iso_vl[num_isolines]);
        num_isolines++;
    }
    else
    {
        delete[] coords_x;
        coords_x = NULL;
        delete[] coords_y;
        coords_y = NULL;
        delete[] coords_z;
        coords_z = NULL;
    }
}
