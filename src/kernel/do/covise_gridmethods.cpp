/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covise_gridmethods.h"
#define CELL_TYPES_ONLY
#include <do/coDoUnstructuredGrid.h>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <time.h> // for time
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "Triangulate.h"

using namespace covise;

// Input:
// array_len: number of field values per point
// velos: array with the field defined at the nodes
// Output:
// fem_c: natural coordinates in the sense of FEM
// interp: interpolated field value
void
grid_methods::interpElem(float fem_c[3], float *interp, int array_len, const float *velos)
{
    int i;
    const float *p_velos;
    float multi[8];
    float val0_m = 1.0f - fem_c[0];
    float val0_p = 1.0f + fem_c[0];
    float val1_m = 1.0f - fem_c[1];
    float val1_p = 1.0f + fem_c[1];
    float val2_m = 1.0f - fem_c[2];
    float val2_p = 1.0f + fem_c[2];
    multi[0] = val0_m * val1_m * val2_m;
    multi[1] = val0_m * val1_m * val2_p;
    multi[2] = val0_m * val1_p * val2_m;
    multi[3] = val0_m * val1_p * val2_p;
    multi[4] = val0_p * val1_m * val2_m;
    multi[5] = val0_p * val1_m * val2_p;
    multi[6] = val0_p * val1_p * val2_m;
    multi[7] = val0_p * val1_p * val2_p;
    // i-th field component
    for (i = 0; i < array_len; ++i)
    {
        p_velos = velos + i;
        interp[i] = multi[0] * (*p_velos);
        p_velos += array_len;
        interp[i] += multi[1] * (*p_velos);
        p_velos += array_len;
        interp[i] += multi[2] * (*p_velos);
        p_velos += array_len;
        interp[i] += multi[3] * (*p_velos);
        p_velos += array_len;
        interp[i] += multi[4] * (*p_velos);
        p_velos += array_len;
        interp[i] += multi[5] * (*p_velos);
        p_velos += array_len;
        interp[i] += multi[6] * (*p_velos);
        p_velos += array_len;
        interp[i] += multi[7] * (*p_velos);
        interp[i] *= 0.125;
    }
}

// function used with rectangular grids for cell location
// its interface is adapted for binary searches
int
grid_methods::asc_compar_fp(const void *key, const void *fp)
{
    float pos = *(float *)(key);
    float fp0 = *(float *)(fp);
    float fp1 = *((float *)(fp) + 1);
    if (pos < fp0)
    {
        return -1;
    }
    else if (pos > fp1)
    {
        return 1;
    }
    return 0;
}

// function used with rectangular grids for cell location
// its interface is adapted for binary searches
int
grid_methods::desc_compar_fp(const void *key, const void *fp)
{
    float pos = *(float *)(key);
    float fp0 = *(float *)(fp);
    float fp1 = *((float *)(fp) + 1);
    if (pos < fp1)
    {
        return 1;
    }
    else if (pos > fp0)
    {
        return -1;
    }
    return 0;
}

// length of a bounding box
float
grid_methods::BoundBox::length()
{
    float x, y, z;
    x = x_max_ - x_min_;
    y = y_max_ - y_min_;
    z = z_max_ - z_min_;
    return sqrt(x * x + y * y + z * z);
}

// maximum of a velocity field at the element nodes for an element
float
grid_methods::getMaxVel(int no_v, const int *v_l,
                        const float *u, const float *v, const float *w)
{
    float ret = -FLT_MAX;
    float tmp;
    int i, vertex;
    for (i = 0; i < no_v; ++i)
    {
        vertex = v_l[i];
        tmp = u[vertex] * u[vertex];
        tmp += v[vertex] * v[vertex];
        tmp += w[vertex] * w[vertex];
        if (tmp > ret)
            ret = tmp;
    }
    return (ret > 0.0) ? sqrt(ret) : ret;
}

// bounding box for an element
void
grid_methods::getBoundBox(BoundBox &bbox, int no_v, const int *v_l,
                          const float *x_in, const float *y_in, const float *z_in)
{
    int i, vertex;
    bbox.x_max_ = (bbox.x_min_ = x_in[*v_l]);
    bbox.y_max_ = (bbox.y_min_ = y_in[*v_l]);
    bbox.z_max_ = (bbox.z_min_ = z_in[*v_l]);
    for (i = 1; i < no_v; ++i)
    {
        vertex = v_l[i];
        if (x_in[vertex] < bbox.x_min_)
            bbox.x_min_ = x_in[vertex];
        if (y_in[vertex] < bbox.y_min_)
            bbox.y_min_ = y_in[vertex];
        if (z_in[vertex] < bbox.z_min_)
            bbox.z_min_ = z_in[vertex];
        if (x_in[vertex] > bbox.x_max_)
            bbox.x_max_ = x_in[vertex];
        if (y_in[vertex] > bbox.y_max_)
            bbox.y_max_ = y_in[vertex];
        if (z_in[vertex] > bbox.z_max_)
            bbox.z_max_ = z_in[vertex];
    }
}

// Tetrahedronisation
void
grid_methods::hex2tet(int ind, const int *el, const int *cl,
                      int i, int *tel, int *tcl)
{
    int j;

    // fill the tel list
    for (j = 0; j < 5; j++)
    {
        tel[j] = j * 4;
    }

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 4];
        tcl[2] = cl[el[i] + 5];
        tcl[3] = cl[el[i] + 7];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 5];
        tcl[6] = cl[el[i] + 2];
        tcl[7] = cl[el[i] + 7];

        tcl[8] = cl[el[i]];
        tcl[9] = cl[el[i] + 1];
        tcl[10] = cl[el[i] + 2];
        tcl[11] = cl[el[i] + 5];

        tcl[12] = cl[el[i]];
        tcl[13] = cl[el[i] + 2];
        tcl[14] = cl[el[i] + 3];
        tcl[15] = cl[el[i] + 7];

        tcl[16] = cl[el[i] + 2];
        tcl[17] = cl[el[i] + 5];
        tcl[18] = cl[el[i] + 6];
        tcl[19] = cl[el[i] + 7];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i] + 3];
        tcl[1] = cl[el[i] + 4];
        tcl[2] = cl[el[i] + 6];
        tcl[3] = cl[el[i] + 7];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 3];
        tcl[6] = cl[el[i] + 4];
        tcl[7] = cl[el[i] + 6];

        tcl[8] = cl[el[i] + 1];
        tcl[9] = cl[el[i] + 2];
        tcl[10] = cl[el[i] + 3];
        tcl[11] = cl[el[i] + 6];

        tcl[12] = cl[el[i]];
        tcl[13] = cl[el[i] + 1];
        tcl[14] = cl[el[i] + 3];
        tcl[15] = cl[el[i] + 4];

        tcl[16] = cl[el[i] + 1];
        tcl[17] = cl[el[i] + 4];
        tcl[18] = cl[el[i] + 5];
        tcl[19] = cl[el[i] + 6];
    }
}

// Tetrahedronisation
void
grid_methods::prism2tet(int ind, const int *el, const int *cl,
                        int i, int *tel, int *tcl)
{
    int j;

    // fill the tel list
    for (j = 0; j < 3; j++)
    {
        tel[j] = j * 4;
    }

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 2];
        tcl[3] = cl[el[i] + 3];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 4];

        tcl[8] = cl[el[i] + 2];
        tcl[9] = cl[el[i] + 3];
        tcl[10] = cl[el[i] + 4];
        tcl[11] = cl[el[i] + 5];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 3];
        tcl[3] = cl[el[i] + 2];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 4];
        tcl[7] = cl[el[i] + 3];

        tcl[8] = cl[el[i] + 2];
        tcl[9] = cl[el[i] + 3];
        tcl[10] = cl[el[i] + 5];
        tcl[11] = cl[el[i] + 4];
    }
}

// Tetrahedronisation
void
grid_methods::pyra2tet(int ind, const int *el, const int *cl,
                       int i, int *tel, int *tcl)
{
    // fill the tel list
    tel[0] = 0;
    tel[1] = 4;

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 3];
        tcl[3] = cl[el[i] + 4];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 4];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 2];
        tcl[3] = cl[el[i] + 4];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 4];
    }
}

float
grid_methods::tri_surf(float *surf, const float *p0, const float *p1,
                       const float *p2)
{
    float v1[3];
    float v2[3];
    v1[0] = p1[0] - p0[0];
    v1[1] = p1[1] - p0[1];
    v1[2] = p1[2] - p0[2];
    v2[0] = p2[0] - p0[0];
    v2[1] = p2[1] - p0[1];
    v2[2] = p2[2] - p0[2];
    surf[0] = v1[1] * v2[2] - v1[2] * v2[1];
    surf[1] = v1[2] * v2[0] - v1[0] * v2[2];
    surf[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return sqrt(surf[0] * surf[0] + surf[1] * surf[1] + surf[2] * surf[2]);
}

//returns the volume of the tetrahedra cell
float
grid_methods::tetra_vol(const float p0[3], const float p1[3],
                        const float p2[3], const float p3[3])
{
    float vol;
    float diff1_0 = p1[0] - p0[0];
    float diff1_1 = p1[1] - p0[1];
    float diff1_2 = p1[2] - p0[2];
    float diff2_0 = p2[0] - p0[0];
    float diff2_1 = p2[1] - p0[1];
    float diff2_2 = p2[2] - p0[2];
    float diff3_0 = p3[0] - p0[0];
    float diff3_1 = p3[1] - p0[1];
    float diff3_2 = p3[2] - p0[2];

    vol = (diff2_1 * diff3_2 - diff3_1 * diff2_2) * diff1_0;
    vol += (diff2_2 * diff3_0 - diff3_2 * diff2_0) * diff1_1;
    vol += (diff2_0 * diff3_1 - diff3_0 * diff2_1) * diff1_2;
    vol *= 0.16666666666667f;

    return vol;
}

namespace covise
{

// get transposed matrix of the inverse
// .... efficiency !?
int
InvTrans(float out[3][3], const float in[3][3])
{
    float det = in[0][0] * in[1][1] * in[2][2];
    det += in[0][1] * in[1][2] * in[2][0];
    det += in[1][0] * in[2][1] * in[0][2];
    det -= in[0][2] * in[1][1] * in[2][0];
    det -= in[0][1] * in[1][0] * in[2][2];
    det -= in[0][0] * in[1][2] * in[2][1];
    if (det == 0.0) // this is a bit naive...
    {
        return -1;
    }
    det = 1.0f / det;
    out[0][0] = in[1][1] * in[2][2] - in[1][2] * in[2][1];
    out[0][1] = in[1][2] * in[2][0] - in[1][0] * in[2][2];
    out[0][2] = in[1][0] * in[2][1] - in[1][1] * in[2][0];
    out[1][0] = in[0][2] * in[2][1] - in[0][1] * in[2][2];
    out[1][1] = in[0][0] * in[2][2] - in[0][2] * in[2][0];
    out[1][2] = in[0][1] * in[2][0] - in[0][0] * in[2][1];
    out[2][0] = in[0][1] * in[1][2] - in[0][2] * in[1][1];
    out[2][1] = in[0][2] * in[1][0] - in[0][0] * in[1][2];
    out[2][2] = in[0][0] * in[1][1] - in[0][1] * in[1][0];
    return 0;
}
}

// derivative operators
int
grid_methods::derivativesAtCenter(float **v_interp[3],
                                  int no_points, int no_arrays, const float *const *velo,
                                  int no_el, int no_vert,
                                  const int *tl, const int *el, const int *conn,
                                  const float *x_in, const float *y_in, const float *z_in)
{
    (void)no_points;
    (void)no_vert;
    int elem;
    const int dzetaHex[8][3] = {
        { -1, -1, -1 },
        { 1, -1, -1 },
        { 1, 1, -1 },
        { -1, 1, -1 },
        { -1, -1, 1 },
        { 1, -1, 1 },
        { 1, 1, 1 },
        { -1, 1, 1 }
    };
    for (elem = 0; elem < no_el; ++elem)
    {
        int base = el[elem];
        switch (tl[elem])
        {
        case TYPE_HEXAEDER:
            float JacMal8[3][3] = {
                { 0.0, 0.0, 0.0 },
                { 0.0, 0.0, 0.0 },
                { 0.0, 0.0, 0.0 }
            };
            float ITJacMal8[3][3];
            int vert;
            for (vert = 0; vert < 8; ++vert)
            {
                JacMal8[0][0] += x_in[conn[base + vert]] * dzetaHex[vert][0];
                JacMal8[0][1] += x_in[conn[base + vert]] * dzetaHex[vert][1];
                JacMal8[0][2] += x_in[conn[base + vert]] * dzetaHex[vert][2];
                JacMal8[1][0] += y_in[conn[base + vert]] * dzetaHex[vert][0];
                JacMal8[1][1] += y_in[conn[base + vert]] * dzetaHex[vert][1];
                JacMal8[1][2] += y_in[conn[base + vert]] * dzetaHex[vert][2];
                JacMal8[2][0] += z_in[conn[base + vert]] * dzetaHex[vert][0];
                JacMal8[2][1] += z_in[conn[base + vert]] * dzetaHex[vert][1];
                JacMal8[2][2] += z_in[conn[base + vert]] * dzetaHex[vert][2];
            }
            if (InvTrans(ITJacMal8, JacMal8) != 0)
            {
                return -1;
            }
            int array;
            for (array = 0; array < no_arrays; ++array)
            {
                float b[3] = { 0.0, 0.0, 0.0 };
                for (vert = 0; vert < 8; ++vert)
                {
                    b[0] += velo[array][conn[base + vert]] * dzetaHex[vert][0];
                    b[1] += velo[array][conn[base + vert]] * dzetaHex[vert][1];
                    b[2] += velo[array][conn[base + vert]] * dzetaHex[vert][2];
                }
                // now get the solution
                v_interp[0][array][elem] = ITJacMal8[0][0] * b[0] + ITJacMal8[0][1] * b[1] + ITJacMal8[0][2] * b[2];
                v_interp[1][array][elem] = ITJacMal8[1][0] * b[0] + ITJacMal8[1][1] * b[1] + ITJacMal8[1][2] * b[2];
                v_interp[2][array][elem] = ITJacMal8[2][0] * b[0] + ITJacMal8[2][1] * b[1] + ITJacMal8[2][2] * b[2];
            }
            break;
        }
    }
    return 0;
}

// assume that point is in element cell
// use for hexaedra in an UNSGRD
int
grid_methods::interpolateInHexa(float *v_interp, const float *point,
                                int no_arrays, int array_dim, const float *const *velo,
                                const int *connl,
                                const float *x_in, const float *y_in, const float *z_in)
{
    if (!v_interp || !velo)
        return 0;
    // generate a microgrid with a single element
    float x_e[8];
    float y_e[8];
    float z_e[8];
    int unst2str[8] = { 0, 4, 6, 2, 1, 5, 7, 3 };
    int vert, coord, strind;
    for (vert = 0; vert < 8; ++vert)
    {
        coord = connl[vert];
        strind = unst2str[vert];
        x_e[strind] = x_in[coord];
        y_e[strind] = y_in[coord];
        z_e[strind] = z_in[coord];
    }
    // use cell3 to get natural coordinates
    float fem_c[3];
    float &a = fem_c[0];
    float &b = fem_c[1];
    float &g = fem_c[2];
    a = 0, b = 0, g = 0;
    float amat[3][3], bmat[3][3];
    int status;
    int cell_ind[3] = { 1, 1, 1 };
    cell3(2, 2, 2, x_e, y_e, z_e, cell_ind, cell_ind + 1, cell_ind + 2, &a, &b, &g,
          const_cast<float *>(point), amat, bmat, &status);
    // transform natural coordinates to my favourite form
    a -= 0.5;
    a += a;
    b -= 0.5;
    b += b;
    g -= 0.5;
    g += g;
    // prepare array velos as required by interpElem
    float *velos = new float[no_arrays * array_dim * 8];
    int array_num, comp_num;
    const float *velo_array_num;

    for (vert = 0; vert < 8; ++vert)
    {
        for (array_num = 0; array_num < no_arrays; ++array_num)
        {
            velo_array_num = velo[array_num];
            for (comp_num = 0; comp_num < array_dim; ++comp_num)
            {
                velos[unst2str[vert] * no_arrays * array_dim + array_dim * array_num + comp_num] = velo_array_num[connl[vert]];
            }
        }
    }
    interpElem(fem_c, v_interp, no_arrays * array_dim, velos);
    delete[] velos;
    return status;
}

// assume that point is in element cell
// use for hexaedra in an UNSGRD
int
grid_methods::interpolateVInHexa(float *v_interp, const float *point,
                                 const float *const *velo,
                                 const int *connl,
                                 const float *x_in, const float *y_in, const float *z_in)
{
    if (!v_interp || !velo)
        return 0;
    // generate a microgrid with a single element
    float x_e[8];
    float y_e[8];
    float z_e[8];
    int unst2str[8] = { 0, 4, 6, 2, 1, 5, 7, 3 };
    int vert, coord, strind;
    for (vert = 0; vert < 8; ++vert)
    {
        coord = connl[vert];
        strind = unst2str[vert];
        x_e[strind] = x_in[coord];
        y_e[strind] = y_in[coord];
        z_e[strind] = z_in[coord];
    }
    // use cell3 to get natural coordinates
    float fem_c[3];
    float &a = fem_c[0];
    float &b = fem_c[1];
    float &g = fem_c[2];
    a = 0, b = 0, g = 0;
    float amat[3][3], bmat[3][3];
    int status;
    int cell_ind[3] = { 1, 1, 1 };
    cell3(2, 2, 2, x_e, y_e, z_e, cell_ind, cell_ind + 1, cell_ind + 2, &a, &b, &g,
          const_cast<float *>(point), amat, bmat, &status);
    // transform natural coordinates to my favourite form
    a -= 0.5;
    a += a;
    b -= 0.5;
    b += b;
    g -= 0.5;
    g += g;
    // prepare array velos as required by interpElem
    float *velos = new float[24];
    const float *velo_0 = velo[0];
    const float *velo_1 = velo[1];
    const float *velo_2 = velo[2];

    int base;
    // corner 0
    base = unst2str[0] * 3;
    coord = connl[0];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];
    // corner 1
    base = unst2str[1] * 3;
    coord = connl[1];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];
    // corner 2
    base = unst2str[2] * 3;
    coord = connl[2];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];
    // corner 3
    base = unst2str[3] * 3;
    coord = connl[3];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];
    // corner 4
    base = unst2str[4] * 3;
    coord = connl[4];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];
    // corner 5
    base = unst2str[5] * 3;
    coord = connl[5];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];
    // corner 6
    base = unst2str[6] * 3;
    coord = connl[6];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];
    // corner 7
    base = unst2str[7] * 3;
    coord = connl[7];
    velos[base] = velo_0[coord];
    velos[base + 1] = velo_1[coord];
    velos[base + 2] = velo_2[coord];

    interpElem(fem_c, v_interp, 3, velos);
    delete[] velos;
    return status;
}

void
grid_methods::interpolateInTriangle(float *v_interp, const float *point,
                                    int no_arrays, int array_dim,
                                    const float *const *velo, int c0, int c1, int c2,
                                    const float *p0, const float *p1, const float *p2)
{
    float proj_point[3];
    int conn[3] = { 0, 1, 2 };
    float xin[3];
    float yin[3];
    float zin[3];
    xin[0] = p0[0];
    xin[1] = p1[0];
    xin[2] = p2[0];
    yin[0] = p0[1];
    yin[1] = p1[1];
    yin[2] = p2[1];
    zin[0] = p0[2];
    zin[1] = p1[2];
    zin[2] = p2[2];
    ProjectPoint(proj_point, point, conn, 0, 3, xin, yin, zin);
    float surfg[3];
    float surf0[3];
    float surf1[3];
    float surf2[3];
    float length = tri_surf(surfg, p0, p1, p2);
    length *= length;
    tri_surf(surf0, proj_point, p1, p2);
    tri_surf(surf1, p0, proj_point, p2);
    tri_surf(surf2, p0, p1, proj_point);
    float w0 = surf0[0] * surfg[0] + surf0[1] * surfg[1] + surf0[2] * surfg[2];
    float w1 = surf1[0] * surfg[0] + surf1[1] * surfg[1] + surf1[2] * surfg[2];
    float w2 = surf2[0] * surfg[0] + surf2[1] * surfg[1] + surf2[2] * surfg[2];
    w0 /= length;
    w1 /= length;
    w2 /= length;

    int array, no_dim;
    if (array_dim == 1) // scalar or vector
    {
        for (array = 0; array < no_arrays; ++array)
        {
            v_interp[array] = w0 * velo[array][c0];
            v_interp[array] += w1 * velo[array][c1];
            v_interp[array] += w2 * velo[array][c2];
        }
    } // general case
    else
    {
        int base = 0;
        for (array = 0; array < no_arrays; ++array)
        {
            for (no_dim = 0; no_dim < array_dim; ++no_dim, ++base)
            {
                v_interp[base] = w0 * velo[array][c0 * array_dim + no_dim];
                v_interp[base] += w1 * velo[array][c1 * array_dim + no_dim];
                v_interp[base] += w2 * velo[array][c2 * array_dim + no_dim];
            }
        }
    }
}

// interpolate the velo field in a tetrahedra (output in v_interp)
// c? mark the 4 nodes. p? mark the 4 tetraheder point coordinates.
void
grid_methods::interpolateInTetra(float *v_interp, const float *px,
                                 int no_arrays, int array_dim, const float *const *velo,
                                 int c0, int c1, int c2, int c3,
                                 const float *p0, const float *p1, const float *p2, const float *p3)
{
    // work out the weights
    float ivg, w0, w1, w2, w3;

    if (!v_interp || !velo)
        return;

    ivg = 1.0f / tetra_vol(p0, p1, p2, p3);
    w0 = tetra_vol(px, p1, p2, p3) * ivg;
    w1 = tetra_vol(p0, px, p2, p3) * ivg;
    w2 = tetra_vol(p0, p1, px, p3) * ivg;
    w3 = tetra_vol(p0, p1, p2, px) * ivg;

    int array, no_dim;
    if (array_dim == 1) // scalar or vector
    {
        for (array = 0; array < no_arrays; ++array)
        {
            v_interp[array] = w0 * velo[array][c0];
            v_interp[array] += w1 * velo[array][c1];
            v_interp[array] += w2 * velo[array][c2];
            v_interp[array] += w3 * velo[array][c3];
        }
    } // general case
    else
    {
        int base = 0;
        for (array = 0; array < no_arrays; ++array)
        {
            for (no_dim = 0; no_dim < array_dim; ++no_dim, ++base)
            {
                v_interp[base] = w0 * velo[array][c0 * array_dim + no_dim];
                v_interp[base] += w1 * velo[array][c1 * array_dim + no_dim];
                v_interp[base] += w2 * velo[array][c2 * array_dim + no_dim];
                v_interp[base] += w3 * velo[array][c3 * array_dim + no_dim];
            }
        }
    }
}

/*****************************************************/

double grid_methods::dot_product(POINT3D vector1, POINT3D vector2)
{
    double scalar;
    scalar = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;

    return scalar;
}

grid_methods::POINT3D grid_methods::cross_product(POINT3D vector1, POINT3D vector2)
{
    POINT3D normal;

    normal.x = (vector1.y * vector2.z) - (vector2.y * vector1.z);
    normal.y = (vector2.x * vector1.z) - (vector1.x * vector2.z);
    normal.z = (vector1.x * vector2.y) - (vector2.x * vector1.y);

    return normal;
}

void grid_methods::TesselatePolyhedron(TESSELATION &triangulated_polyhedron, int num_elem_in, int *elem_in, int num_conn_in, int *conn_in, float *xcoord_in, float *ycoord_in, float *zcoord_in)
{
    for (int i = 0; i < num_elem_in; ++i)
    {
        int conn = elem_in[i];
        int next_conn = (i < num_elem_in - 1) ? elem_in[i + 1] : num_conn_in;
        if (next_conn - conn >= 3) // valid polygon?
        {
            float len;
            bool axisFound;

            // get first axis
            POINT3D x_axis;
            int x_index = conn;
            axisFound = false;
            do
            {
                ++x_index;
                x_axis.x = xcoord_in[conn_in[x_index]] - xcoord_in[conn_in[conn]];
                x_axis.y = ycoord_in[conn_in[x_index]] - ycoord_in[conn_in[conn]];
                x_axis.z = zcoord_in[conn_in[x_index]] - zcoord_in[conn_in[conn]];
                len = (float)sqrt(dot_product(x_axis, x_axis));
                axisFound = (fabs(len) >= 0.00001);
            } while (!axisFound && (x_index < next_conn - 1));
            if (!axisFound)
                continue; // abort
            x_axis.x /= len;
            x_axis.y /= len;
            x_axis.z /= len;

            // get second axis
            POINT3D y_axis;
            int y_index = x_index; // we can skip all points rejected for the x_axis, since they wont work here either
            axisFound = false;
            do
            {
                ++y_index;
                y_axis.x = xcoord_in[conn_in[y_index]] - xcoord_in[conn_in[conn]];
                y_axis.y = ycoord_in[conn_in[y_index]] - ycoord_in[conn_in[conn]];
                y_axis.z = zcoord_in[conn_in[y_index]] - zcoord_in[conn_in[conn]];
                len = (float)sqrt(dot_product(y_axis, y_axis));
                axisFound = (fabs(len) >= 0.00001) && (fabs(dot_product(x_axis, y_axis)) <= 0.75 * len); // y_axis must not be (near) parallel to the x_axis
            } while (!axisFound && (y_index < next_conn - 1));
            if (!axisFound)
                continue; // abort
            y_axis.x /= len;
            y_axis.y /= len;
            y_axis.z /= len;

            // project vertices
            tr_vertexVector projection_vector;
            for (int c = conn; c < next_conn; ++c)
            {
                POINT3D currentVertex;
                currentVertex.x = xcoord_in[conn_in[c]];
                currentVertex.y = ycoord_in[conn_in[c]];
                currentVertex.z = zcoord_in[conn_in[c]];
                tr_vertex projected_vertex;
                projected_vertex.x = (float)dot_product(currentVertex, x_axis);
                projected_vertex.y = (float)dot_product(currentVertex, y_axis);
                projected_vertex.index = conn_in[c];
                projection_vector.push_back(projected_vertex);
            }

            // tesselate
            tr_intVector tesselation_vector;
            Triangulate::Process(projection_vector, tesselation_vector);

            // generate output
            for (int j = 0; j < tesselation_vector.size(); j += 3)
            {
                TRIANGLE tesselation_triangle;
                tesselation_triangle.vertex1 = tesselation_vector[j + 0];
                tesselation_triangle.vertex2 = tesselation_vector[j + 1];
                tesselation_triangle.vertex3 = tesselation_vector[j + 2];

                //              // reject bad triangles (TODO: clarify, what is bad)
                //              POINT3D a, b, c;
                //              float len;
                //              a.x = xcoord_in[tesselation_triangle.vertex2] - xcoord_in[tesselation_triangle.vertex1];
                //              a.y = ycoord_in[tesselation_triangle.vertex2] - ycoord_in[tesselation_triangle.vertex1];
                //              a.z = zcoord_in[tesselation_triangle.vertex2] - zcoord_in[tesselation_triangle.vertex1];
                //              len = sqrt(dot_product(a,a));
                //              a.x /= len;
                //              a.y /= len;
                //              a.z /= len;
                //              b.x = xcoord_in[tesselation_triangle.vertex3] - xcoord_in[tesselation_triangle.vertex1];
                //              b.y = ycoord_in[tesselation_triangle.vertex3] - ycoord_in[tesselation_triangle.vertex1];
                //              b.z = zcoord_in[tesselation_triangle.vertex3] - zcoord_in[tesselation_triangle.vertex1];
                //              len = sqrt(dot_product(b,b));
                //              b.x /= len;
                //              b.y /= len;
                //              b.z /= len;
                //              c.x = xcoord_in[tesselation_triangle.vertex3] - xcoord_in[tesselation_triangle.vertex2];
                //              c.y = ycoord_in[tesselation_triangle.vertex3] - ycoord_in[tesselation_triangle.vertex2];
                //              c.z = zcoord_in[tesselation_triangle.vertex3] - zcoord_in[tesselation_triangle.vertex2];
                //              len = sqrt(dot_product(c,c));
                //              c.x /= len;
                //              c.y /= len;
                //              c.z /= len;
                //              if ( (fabs(dot_product(a, b)) > 0.95) || (fabs(dot_product(b, c)) > 0.95) || (fabs(dot_product(c, a)) > 0.95) )
                //              {
                //                  continue;
                //              }

                triangulated_polyhedron.push_back(tesselation_triangle);
            }
        }
    }
}

void grid_methods::ComputeBoundingBox(int num_coord_in, float *x_coord_in, float *y_coord_in, float *z_coord_in, POINT3D &box_min, POINT3D &box_max, int &radius /*, vector<POINT3D> &box_vertices*/)
{
    int i;

    double diagonal;

    /* Initialize Bounding Box */
    box_min.x = x_coord_in[0];
    box_min.y = y_coord_in[0];
    box_min.z = z_coord_in[0];

    box_max.x = x_coord_in[0];
    box_max.y = y_coord_in[0];
    box_max.z = z_coord_in[0];

    for (i = 0; i < num_coord_in; i++)
    {
        /* Find Minimum Values */
        if (x_coord_in[i] < box_min.x)
        {
            box_min.x = x_coord_in[i];
        }

        if (y_coord_in[i] < box_min.y)
        {
            box_min.y = y_coord_in[i];
        }

        if (z_coord_in[i] < box_min.z)
        {
            box_min.z = z_coord_in[i];
        }

        /* Find Maximum Values  */
        if (x_coord_in[i] > box_max.x)
        {
            box_max.x = x_coord_in[i];
        }

        if (y_coord_in[i] > box_max.y)
        {
            box_max.y = y_coord_in[i];
        }

        if (z_coord_in[i] > box_max.z)
        {
            box_max.z = z_coord_in[i];
        }
    }

    /* Calculate Box Diagonal */
    diagonal = sqrt(pow((box_max.x - box_min.x), 2.0) + pow((box_max.y - box_min.y), 2.0) + pow((box_max.z - box_min.z), 2.0));

    /* Calculate Radius */
    radius = (int)(diagonal + 1);
}

bool grid_methods::InBox(POINT3D box_min, POINT3D box_max, POINT3D query_point)
{
    bool inside_box;

    if ((box_min.x <= query_point.x) && (query_point.x <= box_max.x) && (box_min.y <= query_point.y) && (query_point.y <= box_max.y) && (box_min.z <= query_point.z) && (query_point.z <= box_max.z))
    {
        inside_box = true;
    }

    else
    {
        inside_box = false;
    }

    return inside_box;
}

void grid_methods::RandomRay(POINT3D &end_point, int radius)
{
    double x;
    double y;
    double z;
    double w;
    double t;

    /****************************************************/
    /* Generate a random point on a sphere of radius 1  */
    /****************************************************/

    /* The sphere is sliced at z, and a random point at angle t is generated on the circle of intersection */
    z = 2.0 * (double)rand() / INT_MAX - 1.0;
    t = 2.0 * PI * (double)rand() / INT_MAX;
    w = sqrt(1 - z * z);
    x = w * cos(t);
    y = w * sin(t);

    end_point.x = (radius * x);
    end_point.y = (radius * y);
    end_point.z = (radius * z);
}

char grid_methods::RayBoxTest(POINT3D end_point, POINT3D query_point, POINT3D triangle_box_min, POINT3D triangle_box_max)
{
    double w;

    /************************************************************************/
    /* Test if ray lies entirely to one side of the six faces of the bounding box  */
    /************************************************************************/

    w = triangle_box_min.x;
    if ((end_point.x < w) && (query_point.x < w))
    {
        return '0';
    }

    w = triangle_box_min.y;
    if ((end_point.y < w) && (query_point.y < w))
    {
        return '0';
    }

    w = triangle_box_min.z;
    if ((end_point.z < w) && (query_point.z < w))
    {
        return '0';
    }

    w = triangle_box_max.x;
    if ((end_point.x > w) && (query_point.x > w))
    {
        return '0';
    }

    w = triangle_box_max.y;
    if ((end_point.y > w) && (query_point.y > w))
    {
        return '0';
    }

    w = triangle_box_max.z;
    if ((end_point.z > w) && (query_point.z > w))
    {
        return '0';
    }

    return '?';
}

int grid_methods::PlaneCoeff(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D &normal, double &distance)
{
    int component_index = 0;

    POINT3D triangle_vertex;

    /* Calculate a normal vector to the triangle plane */
    normal.x = (double)(triangle_z[2] - triangle_z[0]) * (triangle_y[1] - triangle_y[0]) - (triangle_z[1] - triangle_z[0]) * (triangle_y[2] - triangle_y[0]);
    normal.y = (double)(triangle_z[1] - triangle_z[0]) * (triangle_x[2] - triangle_x[0]) - (triangle_x[1] - triangle_x[0]) * (triangle_z[2] - triangle_z[0]);
    normal.z = (double)(triangle_x[1] - triangle_x[0]) * (triangle_y[2] - triangle_y[0]) - (triangle_y[1] - triangle_y[0]) * (triangle_x[2] - triangle_x[0]);

    /* Select a point on the plane */
    triangle_vertex.x = (double)triangle_x[0];
    triangle_vertex.y = (double)triangle_y[0];
    triangle_vertex.z = (double)triangle_z[0];

    /* Calculate distance to plane (parameter D of the plane equation) */
    distance = dot_product(triangle_vertex, normal);

    /* Find the largest component of the normal vector */
    double biggest = -DBL_MAX;

    double t = fabs(normal.x);

    if (t > biggest)
    {
        biggest = t;
        component_index = 0;
    }

    t = fabs(normal.y);

    if (t > biggest)
    {
        biggest = t;
        component_index = 1;
    }

    t = fabs(normal.z);

    if (t > biggest)
    {
        biggest = t;
        component_index = 2;
    }

    return component_index;
}

char grid_methods::RayPlaneIntersection(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D query_point, POINT3D end_point, POINT3D &int_point, int &component_index)
{
    double distance;
    double num;
    double denom;
    double t;

    POINT3D normal;
    POINT3D segment;

    component_index = PlaneCoeff(triangle_x, triangle_y, triangle_z, normal, distance);
    num = distance - dot_product(query_point, normal);

    segment.x = end_point.x - query_point.x;
    segment.y = end_point.y - query_point.y;
    segment.z = end_point.z - query_point.z;

    denom = dot_product(segment, normal);

    /* Segment is parallel to plane */
    if (denom == 0.0)
    {
        /* Query point is on the plane */
        if (num == 0.0)
        {
            return 'p';
        }

        else
        {
            return '0';
        }
    }

    else
    {
        t = num / denom;
    }

    /* Calculate intersection point with the plane */
    int_point.x = query_point.x + t * (end_point.x - query_point.x);
    int_point.y = query_point.y + t * (end_point.y - query_point.y);
    int_point.z = query_point.z + t * (end_point.z - query_point.z);

    /********************************************************************/
    /* Return Values:                                                                                  */
    /* 'p':  the segment lies wholly within the plane                                     */
    /* 'q':  the q endpoint is on the plane (but not 'p')                                 */
    /* 'r':  the r endpoint is on the plane (but not 'p')                                  */
    /* '0':  the segment lies strictly to one side or the other of the plane    */
    /* '1':  the segement intersects the plane, and 'p' does not hold           */
    /********************************************************************/

    if ((0.0 < t) && (t < 1.0))
    {
        return '1';
    }

    else if (num == 0.0)
    {
        /* t == 0 */
        return 'q';
    }

    else if (num == denom)
    {
        /* t == 1 */
        return 'r';
    }

    else
    {
        return '0';
    }
}

int grid_methods::AreaSign(POINT3D new_vertex_1, POINT3D new_vertex_2, POINT3D new_vertex_3)
{
    double area2;

    POINT3D A;
    POINT3D B;

    A.x = new_vertex_2.x - new_vertex_1.x;
    A.y = new_vertex_2.y - new_vertex_1.y;
    A.z = new_vertex_2.z - new_vertex_1.z;

    B.x = new_vertex_3.x - new_vertex_1.x;
    B.y = new_vertex_3.y - new_vertex_1.y;
    B.z = new_vertex_3.z - new_vertex_1.z;

    area2 = (A.y * B.z - A.z * B.y) + (A.z * B.x - A.x * B.z) + (A.x * B.y - A.y * B.x);

    /* The area should be an integer. */
    //  if(area2 > 0.5)
    //  {
    //      return  1;
    //  }
    //
    //  else if(area2 < -0.5)
    //  {
    //      return -1;
    //  }
    //
    //  else
    //  {
    //      return  0;
    //  }

    if (area2 > 0.0)
    {
        return 1;
    }

    else if (area2 < 0.0)
    {
        return -1;
    }

    else
    {
        return 0;
    }
}

char grid_methods::InTri2D(POINT3D new_vertex_1, POINT3D new_vertex_2, POINT3D new_vertex_3, POINT3D projected_int_point)
{
    int area0;
    int area1;
    int area2;

    /* Compute three AreaSign() values for projected_int_point  w.r.t. each edge of the face in 2D */
    area0 = AreaSign(projected_int_point, new_vertex_1, new_vertex_2);
    area1 = AreaSign(projected_int_point, new_vertex_2, new_vertex_3);
    area2 = AreaSign(projected_int_point, new_vertex_3, new_vertex_1);

    /*********************************************************************/
    /* Return Values:                                                                                   */
    /* 'V':  int_point coincides with a vertex of the triangle                           */
    /* 'E':  int_point is in the relative interior of an edge of the triangle         */
    /* 'F':  int_point is in the relative interior of a face of the triangle             */
    /* '0':  int_point does not intersect the triangle                                       */
    /*********************************************************************/

    if (((area0 == 0) && (area1 > 0) && (area2 > 0))
        || ((area1 == 0) && (area0 > 0) && (area2 > 0))
        || ((area2 == 0) && (area0 > 0) && (area1 > 0)))
    {
        return 'E';
    }

    if (((area0 == 0) && (area1 < 0) && (area2 < 0))
        || ((area1 == 0) && (area0 < 0) && (area2 < 0))
        || ((area2 == 0) && (area0 < 0) && (area1 < 0)))
    {
        return 'E';
    }

    if (((area0 > 0) && (area1 > 0) && (area2 > 0))
        || ((area0 < 0) && (area1 < 0) && (area2 < 0)))
    {
        return 'F';
    }

    if ((area0 == 0) && (area1 == 0) && (area2 == 0))
    {
        printf("Error in grid_methods::InTri2D\n");
    }

    if (((area0 == 0) && (area1 == 0))
        || ((area0 == 0) && (area2 == 0))
        || ((area1 == 0) && (area2 == 0)))
    {
        return 'V';
    }

    else
    {
        return '0';
    }
}

char grid_methods::InTri3D(float *triangle_x, float *triangle_y, float *triangle_z, int component_index, POINT3D int_point)
{
    POINT3D projected_int_point;
    POINT3D new_vertex_1;
    POINT3D new_vertex_2;
    POINT3D new_vertex_3;

    assert(component_index >= 0);
    assert(component_index < 3);

    switch (component_index)
    {
    /* Project out coordinate X in both int_point and the triangular face */
    case 0:
        projected_int_point.x = 0.0;
        projected_int_point.y = int_point.y;
        projected_int_point.z = int_point.z;
        new_vertex_1.x = 0.0;
        new_vertex_1.y = triangle_y[0];
        new_vertex_1.z = triangle_z[0];
        ;
        new_vertex_2.x = 0.0;
        new_vertex_2.y = triangle_y[1];
        new_vertex_2.z = triangle_z[1];
        new_vertex_3.x = 0.0;
        new_vertex_3.y = triangle_y[2];
        new_vertex_3.z = triangle_z[2];
        break;

    /* Project out coordinate Y in both int_point and the triangular face */
    case 1:
        projected_int_point.x = int_point.x;
        projected_int_point.y = 0.0;
        projected_int_point.z = int_point.y;
        new_vertex_1.x = triangle_x[0];
        new_vertex_1.y = 0.0;
        new_vertex_1.z = triangle_z[0];
        new_vertex_2.x = triangle_x[1];
        new_vertex_2.y = 0.0;
        new_vertex_2.z = triangle_z[1];
        new_vertex_3.x = triangle_x[2];
        new_vertex_3.y = 0.0;
        new_vertex_3.z = triangle_z[2];
        break;

    /* Project out coordinate Z in both int_point and the triangular face */
    case 2:
        projected_int_point.x = int_point.x;
        projected_int_point.y = int_point.y;
        projected_int_point.z = 0.0;
        new_vertex_1.x = triangle_x[0];
        new_vertex_1.y = triangle_y[0];
        new_vertex_1.z = 0.0;
        new_vertex_2.x = triangle_x[1];
        new_vertex_2.y = triangle_y[1];
        new_vertex_2.z = 0.0;
        new_vertex_3.x = triangle_x[2];
        new_vertex_3.y = triangle_y[2];
        new_vertex_3.z = 0.0;
        break;
    default:
        abort();
    }

    return (InTri2D(new_vertex_1, new_vertex_2, new_vertex_3, projected_int_point));
}

char grid_methods::InPlane(/*float *triangle_x, float *triangle_y, float *triangle_z, int component_index, POINT3D query_point, POINT3D end_point, POINT3D int_point*/)
{
    /* Not Implemented */
    return 'p';
}

int grid_methods::VolumeSign(POINT3D a, POINT3D b, POINT3D c, POINT3D d)
{
    double vol;
    double ax;
    double ay;
    double az;
    double bx;
    double by;
    double bz;
    double cx;
    double cy;
    double cz;
    double dx;
    double dy;
    double dz;
    double bxdx;
    double bydy;
    double bzdz;
    double cxdx;
    double cydy;
    double czdz;

    ax = a.x;
    ay = a.y;
    az = a.z;
    bx = b.x;
    by = b.y;
    bz = b.z;
    cx = c.x;
    cy = c.y;
    cz = c.z;
    dx = d.x;
    dy = d.y;
    dz = d.z;
    bxdx = bx - dx;
    bydy = by - dy;
    bzdz = bz - dz;
    cxdx = cx - dx;
    cydy = cy - dy;
    czdz = cz - dz;

    vol = (az - dz) * (bxdx * cydy - bydy * cxdx) + (ay - dy) * (bzdz * cxdx - bxdx * czdz) + (ax - dx) * (bydy * czdz - bzdz * cydy);

    /* The volume should be an integer */
    //  if(vol > 0.5)
    //  {
    //      return  1;
    //  }
    //
    //  else if(vol < -0.5)
    //  {
    //      return -1;
    //  }
    //
    //  else
    //  {
    //      return  0;
    //  }

    if (vol > 0.0)
    {
        return 1;
    }

    else if (vol < 0.0)
    {
        return -1;
    }

    else
    {
        return 0;
    }
}

char grid_methods::RayTriangleCrossing(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D query_point, POINT3D end_point)
{
    int vol0;
    int vol1;
    int vol2;

    POINT3D new_vertex_1;
    POINT3D new_vertex_2;
    POINT3D new_vertex_3;

    new_vertex_1.x = (double)triangle_x[0];
    new_vertex_1.y = (double)triangle_y[0];
    new_vertex_1.z = (double)triangle_z[0];

    new_vertex_2.x = (double)triangle_x[1];
    new_vertex_2.y = (double)triangle_y[1];
    new_vertex_2.z = (double)triangle_z[1];

    new_vertex_3.x = (double)triangle_x[2];
    new_vertex_3.y = (double)triangle_y[2];
    new_vertex_3.z = (double)triangle_z[2];

    vol0 = VolumeSign(query_point, new_vertex_1, new_vertex_2, end_point);
    vol1 = VolumeSign(query_point, new_vertex_2, new_vertex_3, end_point);
    vol2 = VolumeSign(query_point, new_vertex_3, new_vertex_1, end_point);

    /*******************************************************************************************/
    /* Return Values:                                                                                                                       */
    /* 'v':  the open segment includes a vertex of the triangle                                                         */
    /* 'e':  the open segment includes a point in the relative interior of an edge of the triangle        */
    /* 'f':  the open segment includes a point in the relative interior of a face of the triangle            */
    /* '0':  the open segment does not intersect the triangle                                                           */
    /*******************************************************************************************/

    /* Same sign: segment intersects interior of triangle */
    if (((vol0 > 0) && (vol1 > 0) && (vol2 > 0)) || ((vol0 < 0) && (vol1 < 0) && (vol2 < 0)))
    {
        return 'f';
    }

    /* Opposite sign: no intersection between segment and triangle */
    if (((vol0 > 0) || (vol1 > 0) || (vol2 > 0)) && ((vol0 < 0) || (vol1 < 0) || (vol2 < 0)))
    {
        return '0';
    }

    else if ((vol0 == 0) && (vol1 == 0) && (vol2 == 0))
    {
        printf("Error in grid_methods::RayTriangleCrossing\n");
        return '0';
    }

    /* Two zeros: segment intersects vertex */
    else if (((vol0 == 0) && (vol1 == 0)) || ((vol0 == 0) && (vol2 == 0)) || ((vol1 == 0) && (vol2 == 0)))
    {
        return 'v';
    }

    /* One zero: segment intersects edge */
    else if ((vol0 == 0) || (vol1 == 0) || (vol2 == 0))
    {
        return 'e';
    }

    else
    {
        printf("Error in grid_methods::RayTriangleCrossing\n");
        return '0';
    }
}

char grid_methods::RayTriangleIntersection(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D query_point, POINT3D end_point, POINT3D &int_point)
{
    char code;

    int component_index;

    //code = '?';
    //component_index = -1;

    code = RayPlaneIntersection(triangle_x, triangle_y, triangle_z, query_point, end_point, int_point, component_index);

    /*******************************************************************************************/
    /* Return Values:                                                                                                                       */
    /* '0':  the closed segment does not intersect the triangle                                                         */
    /* 'p':  the segment lies wholly within the plane                                                                          */
    /* 'V':  an end point of the segment coincides with a vertex of the triangle                                */
    /* 'E':  an end point of the segment is in the relative interior of an edge of the triangle              */
    /* 'F':  an end point of the segment is in the relative interior of a face of the triangle                 */
    /* 'v':  the open segment includes a vertex of the triangle                                                         */
    /* 'e':  the open segment includes a point in the relative interior of an edge of the triangle        */
    /* 'f':  the open segment includes a point in the relative interior of a face of the triangle            */
    /* '0':  the open segment does not intersect the triangle                                                           */
    /*******************************************************************************************/

    if (code == '0')
    {
        return '0';
    }

    else if (code == 'q')
    {
        return InTri3D(triangle_x, triangle_y, triangle_z, component_index, query_point);
    }

    else if (code == 'r')
    {
        return InTri3D(triangle_x, triangle_y, triangle_z, component_index, end_point);
    }

    else if (code == 'p')
    {
        return InPlane(/*triangle_x, triangle_y, triangle_z, component_index, query_point, end_point, int_point*/);
    }

    else if (code == '1')
    {
        return RayTriangleCrossing(triangle_x, triangle_y, triangle_z, query_point, end_point);
    }

    else
    {
        /* Error */
        return code;
    }
}

char grid_methods::InPolyhedron(float *x_coord_in, float *y_coord_in, float *z_coord_in, POINT3D box_min, POINT3D box_max, POINT3D query_point, POINT3D &end_point, int radius, TESSELATION triangulated_polyhedron)
{
    bool degenerate_ray;

    char test;
    char code = '?';

    int j;
    int crossings;
    int triangle_box_radius;
    int numTries;

    float *triangle_x;
    float *triangle_y;
    float *triangle_z;

    //POINT3D end_point;
    POINT3D int_point;
    POINT3D triangle_box_min;
    POINT3D triangle_box_max;

    std::vector<POINT3D> triangle_box_vertices;

    srand((unsigned int)time(NULL));

    triangle_x = new float[3];
    triangle_y = new float[3];
    triangle_z = new float[3];

    /* Point-in-Bounding-Box Test */
    if (!InBox(box_min, box_max, query_point))
    {
        /* If point is not inside the box then it is also not inside the polyhedron */
        delete[] triangle_x;
        delete[] triangle_y;
        delete[] triangle_z;
        return 'o';
    }

    numTries = 0;

    do
    {
        degenerate_ray = false;
        crossings = 0;

        /* Calculate a random point on a unitary sphere */
        RandomRay(end_point, radius);

        /* Calculate endpoint of ray */
        end_point.x = query_point.x + end_point.x;
        end_point.y = query_point.y + end_point.y;
        end_point.z = query_point.z + end_point.z;

        /* Test ray crossings for each face of the polyhedron */
        for (j = 0; j < triangulated_polyhedron.size(); j++)
        {
            if (!degenerate_ray)
            {
                triangle_x[0] = x_coord_in[triangulated_polyhedron[j].vertex1];
                triangle_x[1] = x_coord_in[triangulated_polyhedron[j].vertex2];
                triangle_x[2] = x_coord_in[triangulated_polyhedron[j].vertex3];

                triangle_y[0] = y_coord_in[triangulated_polyhedron[j].vertex1];
                triangle_y[1] = y_coord_in[triangulated_polyhedron[j].vertex2];
                triangle_y[2] = y_coord_in[triangulated_polyhedron[j].vertex3];

                triangle_z[0] = z_coord_in[triangulated_polyhedron[j].vertex1];
                triangle_z[1] = z_coord_in[triangulated_polyhedron[j].vertex2];
                triangle_z[2] = z_coord_in[triangulated_polyhedron[j].vertex3];

                /* Calculate bounding boxes of each face (triangle) */
                ComputeBoundingBox(3, triangle_x, triangle_y, triangle_z, triangle_box_min, triangle_box_max, triangle_box_radius /*, triangle_box_vertices*/);

                /* Test if the ray lies completely to one side of the six faces of the bounding box */
                test = RayBoxTest(end_point, query_point, triangle_box_min, triangle_box_max);

                if (test == '0')
                {
                    code = '0';
                }

                if (test == '?')
                {
                    /* Test for degeneracies */
                    code = RayTriangleIntersection(triangle_x, triangle_y, triangle_z, query_point, end_point, int_point);
                }

                /* If the ray is degenerate, then exit for-loop and create a new ray */
                if (code == 'p' || code == 'v' || code == 'e')
                {
                    degenerate_ray = true;
                }

                /* If the ray hits a face at an interior point, increment crossings */
                else if (code == 'f')
                {
                    crossings++;
                }

                /* If query point sits on a vertex/edge/face, return that code */
                else if (code == 'V' || code == 'E' || code == 'F')
                {
                    return (code);
                }

                /* If ray misses triangle, do nothing. */
                else if (code == '0')
                {
                    /* Do Nothing */
                }

                else
                {
                    printf("Error in grid_methods::InPolyhedron\n");
                }
            }
        }

        numTries++;
    }
    // Only try a couple of times, it might be a degenerated face and not a degenerated ray; a degenerated face will cause an infinite loop as it will allways be degenerate
    while (degenerate_ray && numTries < 10);

    /* Query point is strictly interior to the polyhedron if and only if the number of crossings is odd */
    if ((crossings % 2) == 1)
    {
        delete[] triangle_x;
        delete[] triangle_y;
        delete[] triangle_z;
        return 'i';
    }

    else
    {
        delete[] triangle_x;
        delete[] triangle_y;
        delete[] triangle_z;
        return 'o';
    }
}

double grid_methods::InterpolateCellData(int num_coord_in, float *x_coord_in, float *y_coord_in, float *z_coord_in, float *data_in, POINT3D query_point)
{
    int i;

    double sum_w;
    double sum_wf;

    std::vector<double> weights;
    std::vector<double> radial_distances;

    POINT3D distance;

    sum_w = 0;
    sum_wf = 0;

    for (i = 0; i < num_coord_in; i++)
    {
        distance.x = query_point.x - x_coord_in[i];
        distance.y = query_point.y - y_coord_in[i];
        distance.z = query_point.z - z_coord_in[i];

        //Avoid division by zero; this implies that the query point is also a vertex of the cell
        if (distance.x == 0 && distance.y == 0 && distance.z == 0)
        {
            return data_in[i];
        }

        radial_distances.push_back(sqrt(pow(distance.x, 2) + pow(distance.y, 2) + pow(distance.z, 2)));
        weights.push_back(1 / pow(radial_distances[i], 2));

        sum_wf += weights[i] * data_in[i];
        sum_w += weights[i];
    }

    return sum_wf / sum_w;
}

/**********************************************************/

namespace covise
{

inline void vectorProduct(float *normal, const float *v1, const float *v2)
{
    normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
    normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
    normal[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

inline float dotProduct(const float *v1, const float *v2)
{
    return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
}

inline void createVector(float *result, const float *org, const float *end)
{
    result[0] = end[0] - org[0];
    result[1] = end[1] - org[1];
    result[2] = end[2] - org[2];
}
}

int
grid_methods::isin_triangle(const float *point,
                            const float *pointa,
                            const float *pointb,
                            const float *pointc,
                            float tolerance)
{
    float ab[3]; // AB
    float bc[3];
    float ca[3];
    createVector(ab, pointa, pointb);
    createVector(bc, pointb, pointc);
    createVector(ca, pointc, pointa);

    // find area scale
    float areaVector[3];
    vectorProduct(areaVector, ab, bc);
    float areaScale2 = dotProduct(areaVector, areaVector);
    float areaScale = sqrt(areaScale2);

    // find height over triangle
    if (areaScale == 0.0)
    {
        return 0;
    }
    float iareaScale = 1.0f / areaScale;

    float pa[3]; // PA
    float pb[3];
    float pc[3];
    createVector(pa, point, pointa);
    createVector(pb, point, pointb);
    createVector(pc, point, pointc);

    float volume = dotProduct(pa, areaVector);
    float height = volume * iareaScale;

    float pointP[3];
    pointP[0] = point[0] + height * areaVector[0] * iareaScale;
    pointP[1] = point[1] + height * areaVector[1] * iareaScale;
    pointP[2] = point[2] + height * areaVector[2] * iareaScale;

    // correct pa,pb,pc
    createVector(pa, pointP, pointa);
    createVector(pb, pointP, pointb);
    createVector(pc, pointP, pointc);

    float pab[3]; // PAxPB
    float pbc[3];
    float pca[3];
    vectorProduct(pab, pa, pb);
    vectorProduct(pbc, pb, pc);
    vectorProduct(pca, pc, pa);

    float mua = dotProduct(pca, pab);
    float mub = dotProduct(pab, pbc);
    float muc = dotProduct(pbc, pca);

    areaScale2 *= tolerance;
    if (mua > -areaScale2
        && mub > -areaScale2
        && muc > -areaScale2)
    {
        return 1;
    }
    return 0;
}

/*
int
grid_methods::isin_triangle(const float *point,
                            const float *point0,
                            const float *point1,
                            const float *point2,
                            float tolerance)
{
   // project point onto the plane defined by point0, point1, point2
   float proj_point[3];
   int conn[3] = {0,1,2};
float x_in[3],y_in[3],z_in[3];
x_in[0] = point0[0];
x_in[1] = point1[0];
x_in[2] = point2[0];
y_in[0] = point0[1];
y_in[1] = point1[1];
y_in[2] = point2[1];
z_in[0] = point0[2];
z_in[1] = point1[2];
z_in[2] = point2[2];
ProjectPoint(proj_point,point,conn,0,3,x_in,y_in,z_in);
float ag,a0,a1,a2;
float sg[3],s0[3],s1[3],s2[3];
ag = tri_surf(sg,point0,point1,point2);
a0 = tri_surf(s0,proj_point,point1,point2);
a1 = tri_surf(s1,point0,proj_point,point2);
a2 = tri_surf(s2,point0,point1,proj_point);

if(a0 + a1 + a2 <= ag*(1.0 + tolerance) )
{
return 1;
}
return 0;
}
*/

void
grid_methods::ExtractNormal(float *normal, int base, int second, int third,
                            const float *x_in, const float *y_in, const float *z_in)
{
    float v1[3];
    float v2[3];
    v1[0] = x_in[second] - x_in[base];
    v1[1] = y_in[second] - y_in[base];
    v1[2] = z_in[second] - z_in[base];
    v2[0] = x_in[third] - x_in[base];
    v2[1] = y_in[third] - y_in[base];
    v2[2] = z_in[third] - z_in[base];
    vectorProduct(normal, v1, v2);
}

void
grid_methods::ProjectPoint(float *proj_point, const float *point,
                           const int *conn, int elem_cell, int num_of_vert,
                           const float *x_in, const float *y_in, const float *z_in)
{
    // compute an average normal vector...
    // ... and an average point
    float avNormal[3] = { 0.0, 0.0, 0.0 };
    float avCentre[3];
    avCentre[0] = x_in[conn[elem_cell]];
    avCentre[1] = y_in[conn[elem_cell]];
    avCentre[2] = z_in[conn[elem_cell]];
    avCentre[0] += x_in[conn[elem_cell + num_of_vert - 1]];
    avCentre[1] += y_in[conn[elem_cell + num_of_vert - 1]];
    avCentre[2] += z_in[conn[elem_cell + num_of_vert - 1]];
    int triangle;
    for (triangle = 1; triangle < num_of_vert - 1; ++triangle)
    {
        int second = conn[elem_cell + triangle];
        int third = conn[elem_cell + triangle + 1];
        float normal[3];
        grid_methods::ExtractNormal(normal, conn[elem_cell],
                                    second, third, x_in, y_in, z_in);
        avNormal[0] += normal[0];
        avNormal[1] += normal[1];
        avNormal[2] += normal[2];
        avCentre[0] += x_in[conn[elem_cell + triangle]];
        avCentre[1] += y_in[conn[elem_cell + triangle]];
        avCentre[2] += z_in[conn[elem_cell + triangle]];
    }
    avCentre[0] /= num_of_vert;
    avCentre[1] /= num_of_vert;
    avCentre[2] /= num_of_vert;
    float length = sqrt(avNormal[0] * avNormal[0] + avNormal[1] * avNormal[1] + avNormal[2] * avNormal[2]);
    avNormal[0] /= length;
    avNormal[1] /= length;
    avNormal[2] /= length;
    // the average normal and point define the plane for projection
    float proj = avNormal[0] * (point[0] - avCentre[0]) + avNormal[1] * (point[1] - avCentre[1]) + avNormal[2] * (point[2] - avCentre[2]);
    proj_point[0] = point[0] - proj * avNormal[0];
    proj_point[1] = point[1] - proj * avNormal[1];
    proj_point[2] = point[2] - proj * avNormal[2];
}

int
grid_methods::isin_tetra(const float px[3], const float p0[3],
                         const float p1[3], const float p2[3], const float p3[3],
                         float rel_tol)
{
    //returns 1 if point px is inside the tetrahedra cell, else 0
    float vg, w0, w1, w2, w3;

    vg = fabsf(tetra_vol(p0, p1, p2, p3));
    w0 = fabsf(tetra_vol(px, p1, p2, p3));
    w1 = fabsf(tetra_vol(p0, px, p2, p3));
    w2 = fabsf(tetra_vol(p0, p1, px, p3));
    w3 = fabsf(tetra_vol(p0, p1, p2, px));

    if (w0 + w1 + w2 + w3 <= vg * (1. + rel_tol))
        return 1;
    else
        return 0;
}

#ifdef _USING_STL_
// Oct-tree stuff
const int grid_methods::constants::NO_OF_BITS = 10;
const int grid_methods::constants::out_of_domain = 0x80000000;

class grid_methods::lists
{
public:
    vector<int> cellList_;
    vector<int> macroCellList_;
    lists(int num_grid_cells)
        : cellList_(1, 0)
        , macroCellList_(1, 1)
    {
        cellList_.reserve(num_grid_cells);
        // !!!!!!!!!!!!
        macroCellList_.reserve(num_grid_cells / 20 + 1);
    }
};

grid_methods::octTree::octTree(int num_grid_cells, const int *keyBBoxes)
    : num_grid_cells_(num_grid_cells)
    , keyBBoxes_(keyBBoxes)
{
    lists_ = new lists(num_grid_cells);
}

void grid_methods::octTree::ModifyLists(int num, int *elements, int offset)
{
    int i;
    int how_many = 0;
    vector<int> &cellList = lists_->cellList_;
    vector<int> &macroCellList = lists_->macroCellList_;
    if (num > 0)
    {
        // changes in cellList
        how_many = cellList.size();
        cellList.push_back(num);
        for (i = 0; i < num; ++i)
        {
            cellList.push_back(*elements);
            ++elements;
        }
    }
    // changes in macroCellList
    macroCellList[offset] = -how_many;
}

inline int grid_methods::octTree::fill_son_share(oct_tree_key MacroCell, int son,
                                                 int i, int level, unsigned char *son_share)
{
    if (key_bbox_intersection((MacroCell << 3) | son,
                              reinterpret_cast<keyBoundBox *>(
                                  const_cast<int *>(keyBBoxes_) + 2 * i),
                              level + 1))
    {
        *son_share |= ('\1' << son);
        return 1;
    }
    return 0;
}

inline int grid_methods::octTree::maxOfCountSons(int *count_sons)
{
    int ret = count_sons[0];
    if (count_sons[1] > ret)
        ret = count_sons[1];
    if (count_sons[2] > ret)
        ret = count_sons[2];
    if (count_sons[3] > ret)
        ret = count_sons[3];
    if (count_sons[4] > ret)
        ret = count_sons[4];
    if (count_sons[5] > ret)
        ret = count_sons[5];
    if (count_sons[6] > ret)
        ret = count_sons[6];
    if (count_sons[7] > ret)
        ret = count_sons[7];
    return ret;
}

// offset refers to the position in macroCellList that is
// used by the actual macrocell in the macroCellList.
// I assume that the caller of DivideOctTree has made room
// in macroCellList for a reference to this offset to be meaningful.
void grid_methods::octTree::DivideOctTree(oct_tree_key MacroCell,
                                          int *list_cells, int num, int level, int offset)
{
    int son, i;
    int count_sons_l[8];

    //   cout.setf(ios::oct,ios::basefield);
    //   cout << "Population in "<<MacroCell<<endl;
    //   cout<<"     ";
    //   cout.setf(ios::dec,ios::basefield);
    //   for(i=0;i<num;++i) cout << list_cells[i] <<' ';
    //   cout<<endl;

    // no more divisions if the population is small enough or if
    // the maximum supported level has been achieved
    if (num <= SMALL_ENOUGH || level == constants::NO_OF_BITS)
    {
        ModifyLists(num, list_cells, offset);
        delete[] list_cells;
        return;
    }
    // determine how the cell population of this macrocell
    // would be shared by its 8 sons
    unsigned char *son_share = new unsigned char[num];
    memset(count_sons_l, '\0', sizeof(int) * 8);
    memset(son_share, '\0', num);
    for (son = 0; son < 8; ++son)
    {
        for (i = 0; i < num; ++i)
        {
            count_sons_l[son] += fill_son_share(MacroCell, son,
                                                list_cells[i], level, &son_share[i]);
        }
    }
    // if we are beyond some critical level and one son inherits the entire
    // cell population, divide no further
    if (level >= CRIT_LEVEL && num == maxOfCountSons(count_sons_l))
    {
        ModifyLists(num, list_cells, offset);
        delete[] son_share;
        delete[] list_cells;
        return;
    }
    // else divide the population,
    // write in macroCellList_,
    // and prepare room for the 8 sons
    vector<int> &macroCellList = lists_->macroCellList_;
    int macroCellListSize = macroCellList.size();
    macroCellList[offset] = macroCellListSize;
    macroCellList.resize(macroCellListSize + 8, 0);
    for (son = 0; son < 8; ++son)
    {
        oct_tree_key son_key;
        son_key = MacroCell << 3;
        son_key |= son;
        int *son_elements = SonList(son, list_cells, num, son_share, count_sons_l);
        DivideOctTree(son_key, son_elements, count_sons_l[son], level + 1,
                      macroCellListSize + son);
    }
    delete[] son_share;
    delete[] list_cells;
}

int *grid_methods::octTree::SonList(int son, int *list_cells,
                                    int num, unsigned char *son_share, int *count_sons)
{
    unsigned char mask = ('\1' << son);
    int *ret = new int[count_sons[son]];
    int *point = ret;
    int i;
    for (i = 0; i < num; ++i)
    {
        if (mask & son_share[i])
        {
            *point = list_cells[i];
            ++point;
        }
    }
    return ret;
}

void grid_methods::octTree::treePrint(ostream &outfile, int level,
                                      oct_tree_key key, int offset)
{
    vector<int> &cellList = lists_->cellList_;
    vector<int> &macroCellList = lists_->macroCellList_;
    int i;
    int entry = macroCellList[offset];
    if (entry == 0)
    {
        // no sons, no cells
        outfile << "Macrocell " << key << " is empty." << endl;
    }
    else if (entry > 0)
    {
        // there are 8 sons
        outfile << "Macrocell " << key << " has 8 sons." << endl;
        for (i = 0; i < 8; ++i)
        {
            oct_tree_key son_key = (key << 3);
            son_key |= i;
            treePrint(outfile, level + 1, son_key, entry + i);
        }
    }
    else if (entry < 0)
    {
        // this macrocell has a cell population, it is a leave
        outfile << "Macrocell " << key << " has " << cellList[-entry] << " cells:" << endl;
        outfile << "   ";
        for (i = -entry + 1; i < -entry + 1 + cellList[-entry]; ++i)
        {
            outfile << cellList[i] << ' ';
        }
        outfile << endl;
    }
}

ostream &operator<<(ostream &outfile, grid_methods::octTree &tree)
{
    int level = 0;
    int key = 1;
    int offset = 0;
    tree.treePrint(outfile, level, key, offset);
    return outfile;
}

grid_methods::octTree::~octTree()
{
    delete lists_;
}

void grid_methods::get_oct_tree_key(oct_tree_key &key,
                                    const BoundBox &bbox, float point[3], int exc)
{
    static const int Xcoord = (1 << 27);
    static const int Ycoord = (1 << 28);
    static const int Zcoord = (1 << 29);
    if (point[0] < bbox.x_min_ || point[0] > bbox.x_max_ || point[1] < bbox.y_min_ || point[1] > bbox.y_max_ || point[2] < bbox.z_min_ || point[2] > bbox.z_max_)
    {
        key = constants::out_of_domain;
        return;
    }
    key = 0;
    int i;
    float c_min, c_max, c_c;
    // these operations are equivalent to (x-x_min)/(x_max-x_min)*(2^10-1)
    // except for bit interleaving for the 3 coordinates...
    // X coordinate
    for (i = 0, c_min = bbox.x_min_, c_max = bbox.x_max_; i < constants::NO_OF_BITS; ++i)
    {
        c_c = 0.5 * (c_min + c_max);
        if ((exc && point[0] < c_c) || (!exc && point[0] <= c_c))
        {
            c_max = c_c;
        }
        else
        {
            c_min = c_c;
            key |= (Xcoord >> (3 * i));
        }
    }
    // Y coordinate
    for (i = 0, c_min = bbox.y_min_, c_max = bbox.y_max_; i < constants::NO_OF_BITS; ++i)
    {
        c_c = 0.5 * (c_min + c_max);
        if ((exc && point[1] < c_c) || (!exc && point[1] <= c_c))
        {
            c_max = c_c;
        }
        else
        {
            c_min = c_c;
            key |= (Ycoord >> (3 * i));
        }
    }
    // Z coordinate
    for (i = 0, c_min = bbox.z_min_, c_max = bbox.z_max_; i < constants::NO_OF_BITS; ++i)
    {
        c_c = 0.5 * (c_min + c_max);
        if ((exc && point[2] < c_c) || (!exc && point[2] <= c_c))
        {
            c_max = c_c;
        }
        else
        {
            c_min = c_c;
            key |= (Zcoord >> (3 * i));
        }
    }
    key |= 0x40000000; // bit 30 is set if the point is in the grid bounding box
}

// 1 if the element is in a macroelement
// assumptions: the element bounding box is contained in the grid
//              bounding box
int grid_methods::key_bbox_intersection(oct_tree_key macroEl, const keyBoundBox *element, int level)
{
    oct_tree_key tmp1, tmp2;
    oct_tree_key tMp1, tMp2;
    oct_tree_key macroElMask;
    // oct_tree_key level_key=(1<<(3*level));
    static const int MaskX = 01111111111;
    static const int MaskY = 02222222222;
    static const int MaskZ = 04444444444;

    if (!level)
        return 1;

    tmp1 = (element->min_ >> ((constants::NO_OF_BITS - level) * 3));
    tMp1 = (element->max_ >> ((constants::NO_OF_BITS - level) * 3));

    // X projection
    tmp2 = MaskX & tmp1;
    tMp2 = MaskX & tMp1;
    /*
      tmp2 |= level_key;
      tMp2 |= level_key;
   */
    macroElMask = (macroEl & MaskX);
    if ((tmp2 < macroElMask && tMp2 < macroElMask) || (tmp2 > macroElMask && tMp2 > macroElMask))
    {
        return 0;
    }

    // Y projection
    tmp2 = MaskY & tmp1;
    tMp2 = MaskY & tMp1;
    /*
      tmp2 |= level_key;
      tMp2 |= level_key;
   */
    macroElMask = (macroEl & MaskY);
    if ((tmp2 < macroElMask && tMp2 < macroElMask) || (tmp2 > macroElMask && tMp2 > macroElMask))
    {
        return 0;
    }

    // Z projection
    tmp2 = MaskZ & tmp1;
    tMp2 = MaskZ & tMp1;
    /*
      tmp2 |= level_key;
      tMp2 |= level_key;
   */
    macroElMask = (macroEl & MaskZ);
    if ((tmp2 < macroElMask && tMp2 < macroElMask) || (tmp2 > macroElMask && tMp2 > macroElMask))
    {
        return 0;
    }

    return 1;
}
#endif

// NASA interpolation functions

// ----------------------
// Interpolationsfunktion
// ----------------------

#ifndef MAX
#define MAX(v1, v2) ((v1) > (v2) ? (v1) : (v2))
#endif
#ifndef MIN
#define MIN(v1, v2) ((v1) < (v2) ? (v1) : (v2))
#endif
#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif
#define SIGN(a, b) ((b) < 0.0 ? -(ABS((a))) : ABS((a)))

/*++ ptran3
 *****************************************************************************
 * PURPOSE: Transform a vector from physical coordinates to computational coordinates
 *          or vice-versa.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

void grid_methods::ptran3(float amat[3][3], float v[3], float vv[3])
{
    vv[0] = v[0] * amat[0][0] + v[1] * amat[1][0] + v[2] * amat[2][0];
    vv[1] = v[0] * amat[0][1] + v[1] * amat[1][1] + v[2] * amat[2][1];
    vv[2] = v[0] * amat[0][2] + v[1] * amat[1][2] + v[2] * amat[2][2];
    return;
}

/*++ cell3
 *****************************************************************************
 * PURPOSE: Find the point (x,y,z) in the grid XYZ and return its (i,j,k) cell number.
 *          We ASSUME that the given (i,j,k), (a,b,g), and subset are valid.  These
 *          can be checked with STRTxx.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: status=1 - Unable to find the point without going out of the
 *          computational domain or active subet.  The computational
 *          point returned indicates the direction to look...
 * RETURNS: None
 *          PROCEDURES CALLED IN THIS ROUTINE :
 *          None
 * NOTES:   None
 *****************************************************************************
--*/
// sl: be careful, *i, *j and *k may start from 1 instead of 0
// sl: be careful, the domain of *a, *b and *g extends from 0 to 1
//                 instead of from -1 to 1

void grid_methods::cell3(int idim, int jdim, int kdim,
                         float *x_in, float *y_in, float *z_in,
                         int *i, int *j, int *k,
                         float *a, float *b, float *g,
                         float x[3], float amat[3][3], float bmat[3][3],
                         int *status)
{
#define MITER 5
/*
      defines for readability
   */
#ifdef PI
#undef PI
#endif

#define PA (*a)
#define PB (*b)
#define PG (*g)
#define PI (*i)
#define PJ (*j)
#define PK (*k)

    float x0, x1, x2, x3, x4, x5, x6, x7, x8,
        xa, xb, xg, xag, xbg, xabg, xab,

        /* y0 and y1 cannot conflict in math.h */
        Y0, Y1, y2, y3, y4, y5, y6, y7, y8,
        ya, yb, yg, yag, ybg, yabg, yab,

        z0, z1, z2, z3, z4, z5, z6, z7, z8,
        za, zb, zg, zag, zbg, zabg, zab;

    int nstep, msteps;
    int iter;
    float dx, dy, dz, da, db, dg, as, bs, gs;
    int i1, J1, k1, in, jN, kn;
    float xh, yh, zh;
    int dialog = 0, istat;
    float err2;
    int di, dj, dk;

    int isav1 = 0, jsav1 = 0, ksav1 = 0,
        isav2 = 0, jsav2 = 0, ksav2 = 0;
    float pab, pbg, pag, pabg;

    *status = 0;

    /*
       Reset saved points used for checking if we're stuck on the same point (or
       ping-ponging back and forth) when searching for the right cell.  This
       would happen while following a point outside the computational domain.
   */

    isav1 = 0;
    isav2 = 0;
    di = 1;
    dj = 1;
    dk = 1;

    /*
     Maximum number of steps before we'll give up is maximum of (idim,jdim,kdim).
   */

    msteps = MAX(idim, jdim);
    msteps = MAX(msteps, kdim);
    msteps *= 50;
    nstep = 0;

    /*
      beginning of loop
   */
    while (1)
    {
        nstep++;
        i1 = PI - 1;
        J1 = PJ - 1;
        k1 = PK - 1;

        x1 = x_in[i1 * jdim * kdim + J1 * kdim + k1];
        Y1 = y_in[i1 * jdim * kdim + J1 * kdim + k1];
        z1 = z_in[i1 * jdim * kdim + J1 * kdim + k1];
        x2 = x_in[PI * jdim * kdim + J1 * kdim + k1];
        y2 = y_in[PI * jdim * kdim + J1 * kdim + k1];
        z2 = z_in[PI * jdim * kdim + J1 * kdim + k1];
        x3 = x_in[i1 * jdim * kdim + PJ * kdim + k1];
        y3 = y_in[i1 * jdim * kdim + PJ * kdim + k1];
        z3 = z_in[i1 * jdim * kdim + PJ * kdim + k1];
        x4 = x_in[PI * jdim * kdim + PJ * kdim + k1];
        y4 = y_in[PI * jdim * kdim + PJ * kdim + k1];
        z4 = z_in[PI * jdim * kdim + PJ * kdim + k1];
        x5 = x_in[i1 * jdim * kdim + J1 * kdim + PK];
        y5 = y_in[i1 * jdim * kdim + J1 * kdim + PK];
        z5 = z_in[i1 * jdim * kdim + J1 * kdim + PK];
        x6 = x_in[PI * jdim * kdim + J1 * kdim + PK];
        y6 = y_in[PI * jdim * kdim + J1 * kdim + PK];
        z6 = z_in[PI * jdim * kdim + J1 * kdim + PK];
        x7 = x_in[i1 * jdim * kdim + PJ * kdim + PK];
        y7 = y_in[i1 * jdim * kdim + PJ * kdim + PK];
        z7 = z_in[i1 * jdim * kdim + PJ * kdim + PK];
        x8 = x_in[PI * jdim * kdim + PJ * kdim + PK];
        y8 = y_in[PI * jdim * kdim + PJ * kdim + PK];
        z8 = z_in[PI * jdim * kdim + PJ * kdim + PK];

        x0 = x1;
        xa = x2 - x1;
        xb = x3 - x1;
        xg = x5 - x1;
        xab = x4 - x3 - xa;
        xag = x6 - x5 - xa;
        xbg = x7 - x5 - xb;
        xabg = x8 - x7 - x6 + x5 - x4 + x3 + xa;

        Y0 = Y1;
        ya = y2 - Y1;
        yb = y3 - Y1;
        yg = y5 - Y1;
        yab = y4 - y3 - ya;
        yag = y6 - y5 - ya;
        ybg = y7 - y5 - yb;
        yabg = y8 - y7 - y6 + y5 - y4 + y3 + ya;

        z0 = z1;
        za = z2 - z1;
        zb = z3 - z1;
        zg = z5 - z1;
        zab = z4 - z3 - za;
        zag = z6 - z5 - za;
        zbg = z7 - z5 - zb;
        zabg = z8 - z7 - z6 + z5 - z4 + z3 + za;

        PA = .5;
        PB = .5;
        PG = .5;

        iter = 0;
        while (1)
        {
            iter++;
            /*
         These next 4 lines of code reduce the number of
         multiplications performed in the succeding 12 lines
         of code.
         */
            pab = PA * PB;
            pag = PA * PG;
            pbg = PB * PG;
            pabg = pab * PG;

            xh = x0 + xa * PA + xb * PB + xg * PG + xab * pab + xag * pag + xbg * pbg + xabg * pabg;
            yh = Y0 + ya * PA + yb * PB + yg * PG + yab * pab + yag * pag + ybg * pbg + yabg * pabg;
            zh = z0 + za * PA + zb * PB + zg * PG + zab * pab + zag * pag + zbg * pbg + zabg * pabg;

            amat[0][0] = xa + xab * PB + xag * PG + xabg * pbg;
            amat[0][1] = ya + yab * PB + yag * PG + yabg * pbg;
            amat[0][2] = za + zab * PB + zag * PG + zabg * pbg;

            amat[1][0] = xb + xab * PA + xbg * PG + xabg * pag;
            amat[1][1] = yb + yab * PA + ybg * PG + yabg * pag;
            amat[1][2] = zb + zab * PA + zbg * PG + zabg * pag;

            amat[2][0] = xg + xag * PA + xbg * PB + xabg * pab;
            amat[2][1] = yg + yag * PA + ybg * PB + yabg * pab;
            amat[2][2] = zg + zag * PA + zbg * PB + zabg * pab;

            inv3x3(amat, bmat, &istat);
            if (istat)
            {
                if (dialog)
                    printf("Degenerate volume at index %d %d %d\n", PI, PJ, PK);
                /*
              See if we're at the edge of the cell.
            If so, move away from the (possibly degenerate)
            edge and recompute the matrix.
            */
                as = PA;
                bs = PB;
                gs = PG;

                if (PA == 0.)
                    PA = .01f;
                if (PA == 1.)
                    PA = .99f;
                if (PB == 0.)
                    PB = .01f;
                if (PB == 1.)
                    PB = .99f;
                if (PG == 0.)
                    PG = .01f;
                if (PG == 1.)
                    PG = .99f;
                if (PA != as || PB != bs || PG != gs)
                    continue;
                else
                {
                    /*
                 We're inside a cell and the transformation
                 matrix is singular.  Move to the next cell and try again.
               */
                    PA = di + .5f;
                    PB = dj + .5f;
                    PG = dk + .5f;
                    break;
                }
            }
            dx = x[0] - xh;
            dy = x[1] - yh;
            dz = x[2] - zh;
            da = dx * bmat[0][0] + dy * bmat[1][0] + dz * bmat[2][0];
            db = dx * bmat[0][1] + dy * bmat[1][1] + dz * bmat[2][1];
            dg = dx * bmat[0][2] + dy * bmat[1][2] + dz * bmat[2][2];
            PA += da;
            PB += db;
            PG += dg;

            /*
           If we're WAY off, don't bother with the error test.
           In fact, go ahead and try another cell.
         */

            if (ABS(PA - .5) > 3.)
                break;
            else if (ABS(PB - .5) > 3.)
                break;
            else if (ABS(PG - .5) > 3.)
                break;
            else
            {
                err2 = da * da + db * db + dg * dg;
                /*
              Check iteration error and branch out if it's small enough.
            */
                if (err2 <= 1.e-4)
                    break;
            }
            if (iter >= MITER)
                break;

        } /* end of edge while */

        /*
        The point is in this cell.
      */

        if (ABS(PA - .5) <= .50005 && ABS(PB - .5) <= .50005 && ABS(PG - .5) <= .50005)
        {
            if (dialog)
                printf("match\n");
            return;
        }
        /*
         We've taken more steps then we're willing to wait...
      */
        else if (nstep > msteps)
        {
            *status = 1;
            if (dialog)
                printf("more than %d steps\n", msteps);
            /*
           Update our (i,j,k) guess, keeping it inbounds.
         */
        }
        else
        {
            in = PI;
            jN = PJ;
            kn = PK;
            if (PA < 0.)
                in = MAX(in - 1, 1);
            if (PA > 1.)
                in = MIN(in + 1, idim - 1);
            if (PB < 0.)
                jN = MAX(jN - 1, 1);
            if (PB > 1.)
                jN = MIN(jN + 1, jdim - 1);
            if (PG < 0.)
                kn = MAX(kn - 1, 1);
            if (PG > 1.)
                kn = MIN(kn + 1, kdim - 1);
            if (dialog)
                printf("try cell index %d %d %d\n", in, jN, kn);
            /*
           Not repeating a previous point.  Use the
           new (i,j,k) and try again.
         */
            if ((in != isav1 || jN != jsav1 || kn != ksav1)
                && (in != isav2 || jN != jsav2 || kn != ksav2))
            {
                isav2 = isav1;
                jsav2 = jsav1;
                ksav2 = ksav1;
                isav1 = in;
                jsav1 = jN;
                ksav1 = kn;
                di = (int)SIGN(1, in - PI);
                dj = (int)SIGN(1, jN - PJ);
                dk = (int)SIGN(1, kn - PK);
                PI = in;
                PJ = jN;
                PK = kn;
                continue;
            }
            else
            {
                /*
              It seems to be outside the domain.
            We would have to extrapolate to find it.
            */
                *status = 1;
                if (dialog)
                    printf("extrapolate 2\n");
                return;
            }
        }
        /* exit */
        break;

    } /* end main while */
}

/*++ padv3
 *****************************************************************************
 * PURPOSE: Advance a particle through a vector field V, roughly CELLFR fraction of a
 *          computational cell.  Use 2-step Runge-Kutta time-advance:
 *          x(*)   = x(n) + dt*f[x(n)]
 *          x(n+1) = x(n) + dt*(1/2)*{f[x(n)]+f[x(*)]}
 *          = x(*) + dt*(1/2)*{f[x(*)]-f[x(n)]}
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: STATUS=1 - Particle has come to rest.
 *          STATUS=2 - Particle has left the (valid) computational domain.
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

void grid_methods::padv3(int *first, float cellfr, int direction,
                         int idim, int jdim, int kdim,
                         float *x_in, float *y_in, float *z_in,
                         float *u_in, float *v_in, float *w_in,
                         int *i, int *j, int *k,
                         float *a, float *b, float *g, float x[4],
                         float min_velo, int *status, float *ovel, float *nvel)
{
    //float asav, bsav, gsav;
    float u[3], uu[3], ustar[3], uustar[3], xstar[3];
    float uumax, dt;
    static float amat[3][3], bmat[3][3];
    int idegen, istat;
    *status = 0;

    /*  First time through, compute the metric transformation matrices.     */

    if (*first)
    {
        metr3(idim, jdim, kdim, x_in, y_in, z_in, *i, *j, *k, *a, *b, *g, amat, bmat,
              &idegen, &istat);
        xstar[0] = x[0];
        xstar[1] = x[1];
        xstar[2] = x[2];
        cell3(idim, jdim, kdim, x_in, y_in, z_in, i, j, k, a, b, g, xstar, amat, bmat, &istat);
        x[0] = xstar[0];
        x[1] = xstar[1];
        x[2] = xstar[2];
        metr3(idim, jdim, kdim, x_in, y_in, z_in, *i, *j, *k, *a, *b, *g, amat, bmat,
              &idegen, &istat);
    }

    /*  Interpolate for the velocity.                                       */

    intp3(idim, jdim, kdim, u_in, v_in, w_in, *i, *j, *k, *a, *b, *g, u);
    if (*first)
    {
        // *ovel=sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        ovel[0] = u[0];
        ovel[1] = u[1];
        ovel[2] = u[2];
        *first = 0;
    }

    /*  Transform velocity to computational space.                          */

    ptran3(bmat, u, uu);

    /*  Limit the timestep so we move only CELLFR fraction of a cell (at    */
    /*  least in the predictor step).                                       */

    uumax = MAX(ABS(uu[0]), ABS(uu[1]));
    uumax = MAX(uumax, ABS(uu[2]));
    if (uumax <= min_velo)
    {
        *status = 1;
        return;
    }

    do
    {
        dt = cellfr / uumax;
        if (direction == 2)
            dt = -dt;

        /*  Predictor step.                                                     */

        xstar[0] = x[0] + dt * u[0];
        xstar[1] = x[1] + dt * u[1];
        xstar[2] = x[2] + dt * u[2];
        // x[3]    = x[3] + dt;   not here!!!....

        /*  Find the new point in computational space.                          */

        cell3(idim, jdim, kdim, x_in, y_in, z_in, i, j, k, a, b, g, xstar, amat, bmat, &istat);
        if (istat == 1)
        {
            *status = 2;
            x[0] = xstar[0];
            x[1] = xstar[1];
            x[2] = xstar[2];
            x[3] = x[3] + dt; // but here, or....
            return;
        }

        /*  Interpolate for velocity at the new point.                          */

        intp3(idim, jdim, kdim, u_in, v_in, w_in, *i, *j, *k, *a, *b, *g, ustar);

        /*  Transform velocity to computational space.                          */

        ptran3(bmat, ustar, uustar);

        /*  Check that our timestep is still reasonable.                        */

        uu[0] = .5f * (uu[0] + uustar[0]);
        uu[1] = .5f * (uu[1] + uustar[1]);
        uu[2] = .5f * (uu[2] + uustar[2]);
        uumax = MAX(ABS(uu[0]), ABS(uu[1]));
        uumax = MAX(uumax, ABS(uu[2]));
        if (uumax <= min_velo)
        {
            *status = 1;
            break;
        }
    } while (ABS(dt * uumax) > 1.5 * (cellfr));

    x[3] = x[3] + dt; // here.

    /*  Corrector step.                                                     */
    // *nvel=sqrt(ustar[0]*ustar[0]+ustar[1]*ustar[1]+ustar[2]*ustar[2]);
    nvel[0] = ustar[0];
    nvel[1] = ustar[1];
    nvel[2] = ustar[2];

    x[0] = x[0] + dt * .5f * (u[0] + ustar[0]);
    x[1] = x[1] + dt * .5f * (u[1] + ustar[1]);
    x[2] = x[2] + dt * .5f * (u[2] + ustar[2]);

    /*  Find the corresponding point in computational space.                */

    cell3(idim, jdim, kdim, x_in, y_in, z_in, i, j, k, a, b, g, x, amat, bmat, &istat);
    if (istat == 1)
        *status = 2;

    return;
}

/*++ intp3
 *****************************************************************************
 * PURPOSE: Interpolate to find the value of F at point (I+A,J+B,K+G).  Use trilinear
 *          interpolation.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   F4 is a series of pointers that takes advantage of the way
 *          C dereferences pointers.  It allows the referencing
 *          of the array "f" as an array of dimension 4, but speeds
 *          array reference calculation as the addresses are stored
 *          in the pointer array.  It also allows array dimensions to
 *          be changed during runtime.  The pointer addresses are
 *          In this particular routine it is more slightly more efficient
 *          to calculate array references than to use the multiple
 *          indirect method with F4, so f is assigned the value of the
 *          address for the first element of the array.
 *          The variable "skip" has been added to allow the routine to
 *          ignore the first value in the solution array.  this routine
 *          for the solution array.
 *****************************************************************************
--*/

/*idim*/
void grid_methods::intp3(int, int jdim, int kdim,
                         float *u_in, float *v_in, float *w_in,
                         int i, int j, int k,
                         float a, float b, float g,
                         float *fi)

{
#define NDIM 5

    int i1, J1, k1; //, nx, itmp;
    float a1, b1, g1, c1, c2, c3, c4, c5, c6, c7, c8;

    a1 = 1.0f - a;
    b1 = 1.0f - b;
    g1 = 1.0f - g;

    c1 = a1 * b1 * g1;
    c2 = a * b1 * g1;
    c3 = a1 * b * g1;
    c4 = a * b * g1;
    c5 = a1 * b1 * g;
    c6 = a * b1 * g;
    c7 = a1 * b * g;
    c8 = a * b * g;

    /*   get subscripts for current array element (original code started
        numbering from 1, whereas here we start from 0 */

    i1 = i - 1;
    J1 = j - 1;
    k1 = k - 1;

    /*------
       the computation of fi[nx] consumed more time than any other in the
       program.  Array reference calculation is a significant part of this.
       The array reference calculations are removed here so that we can
       use manually optimized code.  Repetitive parts of the array reference
       are computed outside of the loop, and the offset for the first subsrcipt
       is calculated ONCE.  The number of multiplies is reduced substantially,
       and execution time is improved by ~30%.  SHPhilipson	--------*/

    fi[0] = c1 * u_in[i1 * jdim * kdim + J1 * kdim + k1] + c2 * u_in[i * jdim * kdim + J1 * kdim + k1] + c3 * u_in[i1 * jdim * kdim + j * kdim + k1] + c4 * u_in[i * jdim * kdim + j * kdim + k1] + c5 * u_in[i1 * jdim * kdim + J1 * kdim + k] + c6 * u_in[i * jdim * kdim + J1 * kdim + k] + c7 * u_in[i1 * jdim * kdim + j * kdim + k] + c8 * u_in[i * jdim * kdim + j * kdim + k];
    fi[1] = c1 * v_in[i1 * jdim * kdim + J1 * kdim + k1] + c2 * v_in[i * jdim * kdim + J1 * kdim + k1] + c3 * v_in[i1 * jdim * kdim + j * kdim + k1] + c4 * v_in[i * jdim * kdim + j * kdim + k1] + c5 * v_in[i1 * jdim * kdim + J1 * kdim + k] + c6 * v_in[i * jdim * kdim + J1 * kdim + k] + c7 * v_in[i1 * jdim * kdim + j * kdim + k] + c8 * v_in[i * jdim * kdim + j * kdim + k];
    fi[2] = c1 * w_in[i1 * jdim * kdim + J1 * kdim + k1] + c2 * w_in[i * jdim * kdim + J1 * kdim + k1] + c3 * w_in[i1 * jdim * kdim + j * kdim + k1] + c4 * w_in[i * jdim * kdim + j * kdim + k1] + c5 * w_in[i1 * jdim * kdim + J1 * kdim + k] + c6 * w_in[i * jdim * kdim + J1 * kdim + k] + c7 * w_in[i1 * jdim * kdim + j * kdim + k] + c8 * w_in[i * jdim * kdim + j * kdim + k];
    return;
}

/*++ metr3
 *****************************************************************************
 * PURPOSE: Compute the metric transformations for the point (I+A,J+B,K+G).  If the
 *          cell is degenerate, return transformations for a "nearby" point (if a cell
 *          edge or face is collapsed), or return the singular transformation and its
 *          pseudo-inverse (if an entire side is collapsed). Indicate the degenerate
 *          coordinate direction(s) in IDEGEN.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: STATUS=1 - The cell is degenerate.  Nearby transformations are
 *          returned.
 *          STATUS=2 - The cell is degenerate.  Singular trasformations are
 *          returned and IDEGEN indicates the degenerate direction(s).
 *          IDEGEN=1 - The cell is degenerate in the i-direction.
 *          IDEGEN=2 - The cell is degenerate in the j-direction.
 *          IDEGEN=3 - The cell is degenerate in the i- and j-directions.
 *          IDEGEN=4 - The cell is degenerate in the k-direction.
 *          IDEGEN=5 - The cell is degenerate in the i- and k-directions.
 *          IDEGEN=6 - The cell is degenerate in the j- and k-directions.
 *          IDEGEN=7 - The cell is degenerate in the i-, j-, and k-directions.
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

/*idim*/
void grid_methods::metr3(int, int jdim, int kdim,
                         float *x_in, float *y_in, float *z_in,
                         int i, int j, int k,
                         float a, float b, float g,
                         float amat[3][3], float bmat[3][3],
                         int *idegen, int *status)
{
    /* local variables						*/

    int i1, J1, k1, istat;
    float aa, bb, gg, as, bs, gs;
    float x1, x2, x3, x4, x5, x6, x7, x8;
    float Y1, y2, y3, y4, y5, y6, y7, y8;
    float z1, z2, z3, z4, z5, z6, z7, z8;
    float xa, xb, xg, xab, xag, xbg, xabg;
    float ya, yb, yg, yab, yag, ybg, yabg;
    float za, zb, zg, zab, zag, zbg, zabg;

    *status = 0;
    *idegen = 0;

    aa = a;
    bb = b;
    gg = g;

    i1 = i - 1;
    J1 = j - 1;
    k1 = k - 1;

    x1 = x_in[i1 * jdim * kdim + J1 * kdim + k1];
    x2 = x_in[i * jdim * kdim + J1 * kdim + k1];
    x3 = x_in[i1 * jdim * kdim + j * kdim + k1];
    x4 = x_in[i * jdim * kdim + j * kdim + k1];
    x5 = x_in[i1 * jdim * kdim + J1 * kdim + k];
    x6 = x_in[i * jdim * kdim + J1 * kdim + k];
    x7 = x_in[i1 * jdim * kdim + j * kdim + k];
    x8 = x_in[i * jdim * kdim + j * kdim + k];
    Y1 = y_in[i1 * jdim * kdim + J1 * kdim + k1];
    y2 = y_in[i * jdim * kdim + J1 * kdim + k1];
    y3 = y_in[i1 * jdim * kdim + j * kdim + k1];
    y4 = y_in[i * jdim * kdim + j * kdim + k1];
    y5 = y_in[i1 * jdim * kdim + J1 * kdim + k];
    y6 = y_in[i * jdim * kdim + J1 * kdim + k];
    y7 = y_in[i1 * jdim * kdim + j * kdim + k];
    y8 = y_in[i * jdim * kdim + j * kdim + k];
    z1 = z_in[i1 * jdim * kdim + J1 * kdim + k1];
    z2 = z_in[i * jdim * kdim + J1 * kdim + k1];
    z3 = z_in[i1 * jdim * kdim + j * kdim + k1];
    z4 = z_in[i * jdim * kdim + j * kdim + k1];
    z5 = z_in[i1 * jdim * kdim + J1 * kdim + k];
    z6 = z_in[i * jdim * kdim + J1 * kdim + k];
    z7 = z_in[i1 * jdim * kdim + j * kdim + k];
    z8 = z_in[i * jdim * kdim + j * kdim + k];

    xa = x2 - x1;
    xb = x3 - x1;
    xg = x5 - x1;
    xab = x4 - x3 - x2 + x1;
    xag = x6 - x5 - x2 + x1;
    xbg = x7 - x5 - x3 + x1;
    xabg = x8 - x7 - x6 + x5 - x4 + x3 + x2 - x1;

    ya = y2 - Y1;
    yb = y3 - Y1;
    yg = y5 - Y1;
    yab = y4 - y3 - y2 + Y1;
    yag = y6 - y5 - y2 + Y1;
    ybg = y7 - y5 - y3 + Y1;
    yabg = y8 - y7 - y6 + y5 - y4 + y3 + y2 - Y1;

    za = z2 - z1;
    zb = z3 - z1;
    zg = z5 - z1;
    zab = z4 - z3 - z2 + z1;
    zag = z6 - z5 - z2 + z1;
    zbg = z7 - z5 - z3 + z1;
    zabg = z8 - z7 - z6 + z5 - z4 + z3 + z2 - z1;

    while (1)
    {
        amat[0][0] = xa + xab * bb + xag * gg + xabg * (bb * gg);
        amat[0][1] = ya + yab * bb + yag * gg + yabg * (bb * gg);
        amat[0][2] = za + zab * bb + zag * gg + zabg * (bb * gg);

        amat[1][0] = xb + xab * aa + xbg * gg + xabg * (aa * gg);
        amat[1][1] = yb + yab * aa + ybg * gg + yabg * (aa * gg);
        amat[1][2] = zb + zab * aa + zbg * gg + zabg * (aa * gg);

        amat[2][0] = xg + xag * aa + xbg * bb + xabg * (aa * bb);
        amat[2][1] = yg + yag * aa + ybg * bb + yabg * (aa * bb);
        amat[2][2] = zg + zag * aa + zbg * bb + zabg * (aa * bb);

        inv3x3(amat, bmat, &istat);

        if (istat >= 1)

        /*  See if we're at the edge of the cell.  If so, move away from the 	*/
        /*  (possibly degenerate) edge and recompute the matrix.		*/

        {
            as = aa;
            bs = bb;
            gs = gg;
            if (aa == 0.)
                aa = .01f;
            if (aa == 1.)
                aa = .99f;
            if (bb == 0.)
                bb = .01f;
            if (bb == 1.)
                bb = .99f;
            if (gg == 0.)
                gg = .01f;
            if (gg == 1.)
                gg = .99f;
            if (aa != as || bb != bs || gg != gs)
            {
                *status = 1;
                continue;
            }

            /*   We're inside a cell and the transformation matrix is singular.	*/
            /*      Determine which directions are degenerate.			*/

            else
            {
                *status = 2;
                if (amat[0][0] == 0. && amat[0][1] == 0.
                    && amat[0][2] == 0.)
                    *idegen = *idegen + 1;
                if (amat[1][0] == 0. && amat[1][1] == 0.
                    && amat[1][2] == 0.)
                    *idegen = *idegen + 2;
                if (amat[2][0] == 0. && amat[2][1] == 0.
                    && amat[2][2] == 0.)
                    *idegen = *idegen + 4;
            } /* endif */
        } /* endif */

        return;

    } /* endwhile */
}

/*++ inv3x3
 *****************************************************************************
 * PURPOSE: Invert the 3x3 matrix A.  If A is singular, do our best to find the
 *          pseudo-inverse.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: STATUS = 1   A has one dependent column.
 *          = 2   A has two dependent columns.
 *          = 3   A is zero.
 * RETURNS: None
 * NOTES:   there may be problems in the port of this code to c, as both
 *          array subscripts and indexing had to be changed.  In
 *          particular, the value of "info" is suspect, as it is set in
 *          the routine "ssvdc" to a loop limit value.  It is then
 *          checked in this routine.  The values may be off by 1.  It
 *          used to report the location of singularities.  In FORTRAN,
 *          the value of zero was used to report singularities.  In C,
 *          since the first element of an array is element zero, we
 *          may be precluded from reporting this element as a singularity.
 *          Verification of this problem requires analysis of the use of
 *          this variable.  Initial analysis seems to indicate that the
 *          value contains an array bound and is only used for checking
 *          limits, and it is NOT used as an index.  Thus the code should
 *          work as orignially desired.
 *          SHP  9/7/89
 *****************************************************************************
--*/

void grid_methods::inv3x3(float a[3][3], float ainv[3][3], int *status)
{
    float tmp[3][3], work[3], s[3], e[3], u[3][3], v[3][3], siu[3][3];
    float det;
    int info;

    *status = 0;

    ainv[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    ainv[1][0] = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    ainv[2][0] = a[1][0] * a[2][1] - a[1][1] * a[2][0];

    ainv[0][1] = a[0][2] * a[2][1] - a[0][1] * a[2][2];
    ainv[1][1] = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    ainv[2][1] = a[0][1] * a[2][0] - a[0][0] * a[2][1];

    ainv[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    ainv[1][2] = a[0][2] * a[1][0] - a[0][0] * a[1][2];
    ainv[2][2] = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    det = a[0][0] * ainv[0][0] + a[0][1] * ainv[1][0] + a[0][2] * ainv[2][0];

    /*   Matrix is nonsingular.  Finish up AINV.		*/

    if (det != 0.0)
    {
        det = 1.0f / det;
        ainv[0][0] = ainv[0][0] * det;
        ainv[0][1] = ainv[0][1] * det;
        ainv[0][2] = ainv[0][2] * det;
        ainv[1][0] = ainv[1][0] * det;
        ainv[1][1] = ainv[1][1] * det;
        ainv[1][2] = ainv[1][2] * det;
        ainv[2][0] = ainv[2][0] * det;
        ainv[2][1] = ainv[2][1] * det;
        ainv[2][2] = ainv[2][2] * det;
    }

    /*   Matrix is singular.  Do a singular value decomposition to construct*/
    /*   the pseudo-inverse.  Use LINPACK routine SSVDC.			*/

    else
    {
        memcpy((char *)tmp, (char *)a, sizeof(tmp));
        ssvdc(&tmp[0][0], 3, 3, s, e, &u[0][0], &v[0][0], work, 11, &info);
        if (s[0] == 0.0)
        {
            *status = 3;
            memset((char *)ainv, 0, sizeof(tmp));
            return;
        }

        /*              -1 T			*/
        /*   Compute V S  U .			*/

        s[0] = 1.0f / s[0];
        if (s[2] * s[0] < 1.e-5)
        {
            *status = 1;
            s[2] = 0.;
        }
        else
            s[2] = 1.0f / s[2];

        if (s[1] * s[0] < 1.e-5)
        {
            *status = 2;
            s[1] = 0.;
        }
        else
            s[1] = 1.0f / s[1];

        /*   Start out assuming S is a diagonal matrix.	*/

        siu[0][0] = s[0] * u[0][0];
        siu[0][1] = s[1] * u[1][0];
        siu[0][2] = s[2] * u[2][0];

        siu[1][0] = s[0] * u[0][1];
        siu[1][1] = s[1] * u[1][1];
        siu[1][2] = s[2] * u[2][1];

        siu[2][0] = s[0] * u[0][2];
        siu[2][1] = s[1] * u[1][2];
        siu[2][2] = s[2] * u[2][2];

        /*   S is upper bidiagonal, with E as the super diagonal.	*/

        if (info >= 1)
        {
            siu[0][0] = siu[0][0] - (e[0] * s[0] * s[1]) * u[0][1];
            siu[1][0] = siu[1][0] - (e[0] * s[0] * s[1]) * u[1][1];
            siu[2][0] = siu[2][0] - (e[0] * s[0] * s[1]) * u[2][1];
        }
        if (info >= 2)
        {
            siu[0][0] = siu[0][0] + (e[0] * e[1] * s[0] * powf(s[1], 2.f) * s[2]) * u[0][2];
            siu[1][0] = siu[1][0] + (e[0] * e[1] * s[0] * powf(s[1], 2.f) * s[2]) * u[1][2];
            siu[2][0] = siu[2][0] + (e[0] * e[1] * s[0] * powf(s[1], 2.f) * s[2]) * u[2][2];

            siu[0][1] = siu[0][1] - (e[1] * s[1] * s[2]) * u[0][2];
            siu[1][1] = siu[1][1] - (e[1] * s[1] * s[2]) * u[1][2];
            siu[2][1] = siu[2][1] - (e[1] * s[1] * s[2]) * u[2][2];
        }

        /*               +       -1 T			*/
        /*   Finish up  A   = V S  U .			*/

        ainv[0][0] = v[0][0] * siu[0][0] + v[1][0] * siu[0][1] + v[2][0] * siu[0][2];
        ainv[0][1] = v[0][1] * siu[0][0] + v[1][1] * siu[0][1] + v[2][1] * siu[0][2];
        ainv[0][2] = v[0][2] * siu[0][0] + v[1][2] * siu[0][1] + v[2][2] * siu[0][2];

        ainv[1][0] = v[0][0] * siu[1][0] + v[1][0] * siu[1][1] + v[2][0] * siu[1][2];
        ainv[1][1] = v[0][1] * siu[1][0] + v[1][1] * siu[1][1] + v[2][1] * siu[1][2];
        ainv[1][2] = v[0][2] * siu[1][0] + v[1][2] * siu[1][1] + v[2][2] * siu[1][2];

        ainv[2][0] = v[0][0] * siu[2][0] + v[1][0] * siu[2][1] + v[2][0] * siu[2][2];
        ainv[2][1] = v[0][1] * siu[2][0] + v[1][1] * siu[2][1] + v[2][1] * siu[2][2];
        ainv[2][2] = v[0][2] * siu[2][0] + v[1][2] * siu[2][1] + v[2][2] * siu[2][2];
    }

    return;
}

/*++ ssvdc
 *****************************************************************************
 * PURPOSE: ssvdc is a subroutine to reduce a real nxp matrix x by
 *          orthogonal transformations u and v to diagonal form.  the
 *          diagonal elements s[i] are the singular values of x.  the
 *          columns of u are the corresponding left singular vectors,
 *          and the columns of v the right singular vectors.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  on entry
 *          x        *float.
 *          x contains the matrix whose singular value
 *          decomposition is to be computed.  x is
 *          destroyed by ssvdc.
 *          n        int .
 *          n is the number of rows of the matrix x.
 *          p        int .
 *          p is the number of columns of the matrix x.
 *          work     float(n).
 *          work is a scratch array.
 *          job      int .
 *          job controls the computation of the singular
 *          vectors.  it has the decimal expansion ab
 *          with the following meaning
 *          a = 0    do not compute the left singular
 *          vectors.
 *          a = 1    return the n left singular vectors
 *          in u.
 *          a > 2    return the first min(n,p) singular
 *          vectors in u.
 *          b = 0    do not compute the right singular
 *          vectors.
 *          b = 1    return the right singular vectors
 * OUTPUTS: on return
 *          s         float(mm), where mm=min(n+1,p).
 *          the first min(n,p) entries of s contain the
 *          singular values of x arranged in descending
 *          order of magnitude.
 *          e         float(p).
 *          e ordinarily contains zeros.  however see the
 *          discussion of info for exceptions.
 *          u         float.  if joba = 1 then
 *          k = n, if joba > 2 then
 *          k = min(n,p).
 *          u contains the matrix of left singular vectors.
 *          u is not referenced if joba = 0.  if n < p
 *          or if joba = 2, then u may be identified with x
 *          in the subroutine call.
 *          v         float..
 *          v contains the matrix of right singular vectors.
 *          v is not referenced if job = 0.  if p < n,
 *          then v may be identified with x in the
 *          subroutine call.
 *          info      int .
 *          the singular values (and their corresponding
 *          singular vectors) s(info+1),s(info+2),...,s(m)
 *          are correct (here m=min(n,p)).  thus if
 *          info = 0, all the singular values and their
 *          vectors are correct.  in any event, the matrix
 *          b = trans(u)*x*v is the bidiagonal matrix
 *          with the elements of s on its diagonal and the
 *          elements of e on its super-diagonal (trans(u)
 *          is the transpose of u).  thus the singular
 *          values of x and b are the same.
 * RETURNS: None
 *          PROCEDURES CALLED FROM THIS ROUTINE :
 *          srot, saxpy,sdot,sscal,sswap,snrm2,srotg
 * NOTES:   None
 *****************************************************************************
--*/

#define X_ARR(a, b) (*(x + ((a)*n) + (b)))
#define U_ARR(a, b) (*(u + ((a)*p) + (b)))
#define V_ARR(a, b) (*(v + ((a)*p) + (b)))

void grid_methods::ssvdc(float *x,
                         int n, int p, float *s, float *e,
                         float *u, float *v, float *work,
                         int job, int *info)
{
    /*
      LOCAL VARIABLES
   */
    int i, iter, j, jobu, k, kase, kk, l, ll, lls, lm1, lp1, ls, lu, m, maxit,
        mm, mm1, mp1, nct, nctp1, ncu, nrt, nrtp1;
    float b, c, cs, el, emm1, f, g, scale_val, shift, sl, sm, sn, smm1, t1, test,
        t, ztest;
    int wantu = 0, wantv = 0;

    // ma 10/2003 added initialisation
    l = 0;
    ls = 0;

    // we 08/2002 added initialisation
    sn = 0;
    cs = 0;

    /*
     set the maximum number of iterations.
   */
    maxit = 30;

    /*
     determine what is to be computed.
   */

    jobu = (job % 100) / 10;
    ncu = n;
    if (jobu > 1)
        ncu = MIN(n, p);
    if (jobu != 0)
        wantu = 1;
    if ((job % 10) != 0)
        wantv = 1;

    /*
     reduce x to bidiagonal form, storing the diagonal elements
     in s and the super-diagonal elements in e.
   */

    *info = 0;
    nct = MIN(n - 1, p);
    nrt = MAX(0, MIN(p - 2, n));
    lu = MAX(nct, nrt);
    if (lu >= 1)
    {
        for (l = 1; l <= lu; l++)
        {
            lp1 = l + 1;
            if (l <= nct)
            {
                /*
                   compute the transformation for the l-th column and
                   place the l-th diagonal in s[l].
            */
                s[l - 1] = snrm2(n - l + 1, &X_ARR(l - 1, l - 1), 1);
                if (s[l - 1] != 0.0e0)
                {
                    if (X_ARR(l - 1, l - 1) != 0.0e0)
                        s[l - 1] = SIGN(s[l - 1], X_ARR(l - 1, l - 1));
                    /* !alert +1 to +2 */
                    sscal(n - l + 1, (1.0f / s[l - 1]), &X_ARR(l - 1, l - 1), 1);
                    X_ARR(l - 1, l - 1) = 1.0e0f + X_ARR(l - 1, l - 1);
                }
                s[l - 1] = -s[l - 1];
            }
            if (p >= lp1)
            {
                for (j = lp1; j <= p; j++)
                {
                    if (l <= nct)
                    {
                        if (s[l - 1] != 0.0e0)
                        {
                            /*
                            apply the transformation.
                     */
                            t = -sdot(n - l + 1, &X_ARR(l - 1, l - 1), 1,
                                      &X_ARR(j - 1, l - 1), 1) / X_ARR(l - 1, l - 1);
                            saxpy(n - l + 1, t, &X_ARR(l - 1, l - 1), 1, &X_ARR(j - 1, l - 1), 1);
                        }
                    }
                    /*
                         place the l-th row of x into  e for the
                         subsequent calculation of the row transformation.
               */
                    e[j - 1] = X_ARR(j - 1, l - 1);
                }
            }
            if (wantu && l <= nct)
            {
                /*
                   place the transformation in u for subsequent back
                   multiplication.
            */
                for (i = 1; i <= n; i++)
                    U_ARR(l - 1, i - 1) = X_ARR(l - 1, i - 1);
            }
            if (l <= nrt)
            {
                /*
                   compute the l-th row transformation and place the
                   l-th super-diagonal in e[l].
            */
                e[l - 1] = snrm2(p - l, &e[lp1 - 1], 1);
                if (e[l - 1] != 0.0e0)
                {
                    if (e[lp1 - 1] != 0.0e0)
                        e[l - 1] = SIGN(e[l - 1], e[lp1 - 1]);
                    sscal(p - l, 1.0f / e[l - 1], &e[lp1 - 1], 1);
                    e[lp1 - 1] = 1.0f + e[lp1 - 1];
                }
                e[l - 1] = -e[l - 1];
                if (lp1 <= n && e[l - 1] != 0.0e0)
                {
                    /*
                        apply the transformation.
                     */
                    for (i = lp1; i <= n; i++)
                        work[i - 1] = 0.0e0;
                    for (j = lp1; j <= p; j++)
                        saxpy(n - 1, e[j - 1], &X_ARR(j - 1, lp1 - 1), 1, &work[lp1 - 1], 1);
                    for (j = lp1; j <= p; j++)
                        saxpy(n - l, -e[j - 1] / e[lp1 - 1],
                              &work[lp1 - 1], 1, &X_ARR(j - 1, lp1 - 1), 1);
                }
                if (wantv)
                {
                    /*
                      place the transformation in v for subsequent
                      back multiplication.
               */
                    for (i = lp1; i <= p; i++)
                        V_ARR(l - 1, i - 1) = e[i - 1];
                }
            }
        } /* end of 'for loop' */
    } /* end of 'if lu' */

    /*
     set up the final bidiagonal matrix or order m.
   */

    m = MIN(p, n + 1);
    nctp1 = nct + 1;
    nrtp1 = nrt + 1;
    if (nct < p)
        s[nctp1 - 1] = X_ARR(nctp1 - 1, nctp1 - 1);
    if (n < m)
        s[m - 1] = 0.0e0;
    if (nrtp1 < m)
        e[nrtp1 - 1] = X_ARR(m - 1, nrtp1 - 1);
    e[m - 1] = 0.0e0;

    /*
     if required, generate u.
   */

    if (wantu)
    {
        if (ncu >= nctp1)
        {
            for (j = nctp1; j < ncu; j++)
            {
                for (i = 1; i <= n; i++)
                    U_ARR(j - 1, i - 1) = 0.0e0;
                U_ARR(j - 1, j - 1) = 1.0e0;
            }
        }
        if (nct >= 1)
        {
            for (ll = 1; ll <= nct; ll++)
            {
                l = nct - ll + 1;
                if (s[l - 1] != 0.0e0)
                {
                    lp1 = l + 1;
                    if (ncu >= lp1)
                    {
                        for (j = lp1; j <= ncu; j++)
                        {
                            t = -sdot(n - l + 1, &U_ARR(l - 1, l - 1), 1,
                                      &U_ARR(j - 1, l - 1), 1) / U_ARR(l - 1, l - 1);
                            saxpy(n - l + 1, t, &U_ARR(l - 1, l - 1), 1, &U_ARR(j - 1, l - 1), 1);
                        }
                    }
                    sscal(n - l + 1, -1.0e0, &U_ARR(l - 1, l - 1), 1);
                    U_ARR(l - 1, l - 1) = 1.0e0f + U_ARR(l - 1, l - 1);
                    lm1 = l - 1;
                    if (lm1 >= 1)
                    {
                        for (i = 1; i <= lm1; i++)
                            U_ARR(l - 1, i - 1) = 0.0e0;
                    }
                    continue;
                }
                for (i = 1; i <= n; i++)
                    U_ARR(l - 1, i - 1) = 0.0e0;
                U_ARR(l - 1, l - 1) = 1.0e0;
            }
        }

        /*
        if it is required, generate v.
      */

        if (wantv)
        {
            for (ll = 1; ll <= p; ll++)
            {
                l = p - ll + 1;
                lp1 = l + 1;
                if (l <= nrt)
                {
                    if (e[l - 1] != 0.0e0)
                    {
                        for (j = lp1; j <= p; j++)
                        {
                            t = -sdot(p - l, &V_ARR(l - 1, lp1 - 1), 1,
                                      &V_ARR(j - 1, lp1 - 1), 1) / V_ARR(l - 1, lp1 - 1);
                            saxpy(p - l, t, &V_ARR(l - 1, lp1 - 1), 1, &V_ARR(j - 1, lp1 - 1), 1);
                        }
                    }
                }
                for (i = 1; i <= p; i++)
                    V_ARR(l - 1, i - 1) = 0.0e0;
                V_ARR(l - 1, l - 1) = 1.0e0;
            }
        }

        /*
        main iteration loop for the singular values.
      */

        mm = m;
        iter = 0;
        while (1)
        {
            /*
                quit if all the singular values have been found.
         */
            if (m == 0)
                break;

            /*
                if too many iterations have been performed, set
                flag and return.
         */

            if (iter >= maxit)
            {
                *info = m;
                /*
            exit
            */
                break;
            }
            /*
                this section of the program inspects for
                negligible elements in the s and e arrays.  on
                completion the variables kase and l are set as follows.

                   kase = 1     if s[m) and e(l-1) are negligible and l.lt.m
                   kase = 2     if s[l) is negligible and l.lt.m
                   kase = 3     if e(l-1) is negligible, l.lt.m, and
                                s[l), ..., s[m) are not negligible (qr step).
                   kase = 4     if e(m-1) is negligible (convergence).
         */

            for (ll = 1; ll <= m; ll++)
            {
                l = m - ll;
                /*
                   exit
            */
                if (l == 0)
                    break;
                test = ABS(s[l - 1]) + ABS(s[l + 1 - 1]);
                ztest = test + ABS(e[l - 1]);
                if (ztest == test)
                {
                    e[l - 1] = 0.0e0;
                    /*
                 exit
               */
                    break;
                }
            }
            while (1)
            {
                if (l == m - 1)
                {
                    kase = 4;
                    break;
                }
                lp1 = l + 1;
                mp1 = m + 1;
                for (lls = lp1; lls <= mp1; lls++)
                {
                    ls = m - lls + lp1;
                    /*
                      exit
               */
                    if (ls == l)
                        break;
                    test = 0.0e0;
                    if (ls != m)
                        test = test + ABS(e[ls - 1]);
                    if (ls != l + 1)
                        test += ABS(e[ls - 2]);
                    ztest = test + ABS(s[ls - 1]);
                    if (ztest == test)
                    {
                        s[ls - 1] = 0.0e0;
                        /*
                         exit
                  */
                        break;
                    }
                }
                if (ls == l)
                {
                    kase = 3;
                    break;
                }
                if (ls == m)
                {
                    kase = 1;
                    break;
                }
                kase = 2;
                l = ls;
                break;
            } /* end of while */
            l = l + 1;

            /*
                perform the task indicated by kase.
         */

            switch (kase)
            {

            /*
                   deflate negligible s[m].
            */
            case 1:
                mm1 = m - 1;
                f = e[m - 2];
                e[m - 2] = 0.0e0;
                for (kk = l; kk <= mm1; kk++)
                {
                    k = mm1 - kk + l;
                    t1 = s[k - 1];
                    srotg(t1, f, cs, sn);
                    s[k - 1] = t1;
                    if (k != l)
                    {
                        f = -sn * e[k - 2];
                        e[k - 2] = cs * e[k - 2];
                    }
                    if (wantv)
                        srot(p, &V_ARR(k - 1, 0), 1, &V_ARR(m - 1, 0), 1, cs, sn);
                }
                continue;
            //break;

            /*
                      split at negligible s[l].
               */

            case 2:
                f = e[l - 2];
                e[l - 2] = 0.0e0;
                for (k = l; k <= m; k++)
                {
                    t1 = s[k - 1];
                    srotg(t1, f, cs, sn);
                    s[k - 1] = t1;
                    f = -sn * e[k - 1];
                    e[k - 1] = cs * e[k - 1];
                    if (wantu)
                        srot(n, &U_ARR(k - 1, 0), 1, &U_ARR(l - 2, 0), 1, cs, sn);
                }
                continue;
            //break;

            /*
                      perform one qr step.
               */
            case 3:

                /*
                      calculate the shift.
               */
                scale_val = MAX(ABS(s[m - 1]), ABS(s[m - 2]));
                scale_val = MAX(scale_val, ABS(e[m - 2]));
                scale_val = MAX(scale_val, ABS(s[l - 1]));
                scale_val = MAX(scale_val, ABS(e[l - 1]));

                sm = s[m - 1] / scale_val;
                smm1 = s[m - 2] / scale_val;
                emm1 = e[m - 2] / scale_val;
                sl = s[l - 1] / scale_val;
                el = e[l - 1] / scale_val;
                b = ((smm1 + sm) * (smm1 - sm) + powf(emm1, 2)) / 2.0e0f;
                c = powf(sm * emm1, 2);
                shift = 0.0e0;
                if (b != 0.0e0 || c != 0.0e0)
                {
                    shift = sqrt(powf(b, 2) + c);
                    if (b < 0.0e0)
                        shift = -shift;
                    shift = c / (b + shift);
                }
                f = (sl + sm) * (sl - sm) + shift;
                g = sl * el;

                /*
                      chase zeros.
               */

                mm1 = m - 1;
                for (k = l; k <= mm1; k++)
                {
                    srotg(f, g, cs, sn);
                    if (k != l)
                        e[k - 2] = f;
                    f = cs * s[k - 1] + sn * e[k - 1];
                    e[k - 1] = cs * e[k - 1] - sn * s[k - 1];
                    g = sn * s[k + 1 - 1];
                    s[k + 1 - 1] = cs * s[k + 1 - 1];
                    if (wantv)
                        srot(p, &V_ARR(k - 1, 0), 1, &V_ARR(k + 1 - 1, 0), 1, cs, sn);
                    srotg(f, g, cs, sn);
                    s[k - 1] = f;
                    f = cs * e[k - 1] + sn * s[k + 1 - 1];
                    s[k + 1 - 1] = -sn * e[k - 1] + cs * s[k + 1 - 1];
                    g = sn * e[k + 1 - 1];
                    e[k + 1 - 1] = cs * e[k + 1 - 1];
                    if (wantu && k < n)
                        srot(n, &U_ARR(k - 1, 0), 1, &U_ARR(k + 1 - 1, 0), 1, cs, sn);
                }
                e[m - 2] = f;
                iter++;
                continue;
            //break;

            /*
                      convergence.
               */
            case 4:
                /*
                      make the singular value  positive.
               */
                if (s[l - 1] >= 0.0e0)
                {
                    s[l - 1] = -s[l - 1];
                    if (wantv)
                        sscal(p, -1.0e0, &V_ARR(l - 1, 0), 1);
                }

                /*
                      order the singular value.
               */
                while (l != mm)
                {
                    /*
                              exit
                  */
                    if (s[l - 1] >= s[l + 1 - 1])
                        break;
                    t = s[l - 1];
                    s[l - 1] = s[l + 1 - 1];
                    s[l + 1 - 1] = t;
                    if (wantv && l < p)
                        sswap(p, &V_ARR(l - 1, 0), 1, &V_ARR(l + 1 - 1, 0), 1);
                    if (wantu && l < n)
                        sswap(n, &U_ARR(l - 1, 0), 1, &U_ARR(l + 1 - 1, 0), 1);
                    l++;
                }
                iter = 0;
                m--;
            } /* end of switch */
        }
    }
}

/*++ srot
 *****************************************************************************
 * PURPOSE: this routine applies a plane rotation.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:
 *****************************************************************************
 --*/

void grid_methods::srot(int n, float *sx, int incx, float *sy,
                        int incy, float c, float s)
{
    float stemp;
    int i, ix, iy;
    if (n < 0)
        return;

    if (incx == 1 && incy == 1)
    {
        /*
            code for both increments equal to 1
      */
        for (i = 0; i < n; i++)
        {
            stemp = c * sx[i] + s * sy[i];
            sy[i] = c * sy[i] - s * sx[i];
            sx[i] = stemp;
        }
        return;
    }
    /*
     code for unequal increments or equal increments not equal to 1
   */
    ix = 1;
    iy = 1;
    if (incx < 0)
        ix = (-n + 1) * incx + 1;
    if (incy < 0)
        iy = (-n + 1) * incy + 1;
    for (i = 0; i < n; i++)
    {
        stemp = c * sx[ix] + s * sy[iy];
        sy[iy] = c * sy[iy] - s * sx[ix];
        sx[ix] = stemp;
        ix = ix + incx;
        iy = iy + incy;
    }
}

/*++ srotg
 *****************************************************************************
 * PURPOSE: This routine constructs given plane rotation.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   Uses unrolled loops for increment equal to 1.
 *****************************************************************************
 --*/

void grid_methods::srotg(float sa, float sb, float c, float s)
{
    float roe, scale_val, r, z;

    roe = sb;
    if (ABS(sa) > ABS(sb))
        roe = sa;
    scale_val = ABS(sa) + ABS(sb);
    if (scale_val != 0.0)
    {
        r = scale_val * sqrt(powf((sa / scale_val), 2.f) + powf((sb / scale_val), 2.f));
        r = SIGN(1.0f, roe) * r;
        c = sa / r;
        s = sb / r;
    }
    else
    {
        c = 1.0;
        s = 0.0;
        r = 0.0;
    }
    z = 1.0;
    if (ABS(sa) > ABS(sb))
        z = s;
    if (ABS(sb) >= ABS(sa) && c != 0.0)
        z = 1.0f / c;
    sa = r;
    sb = z;
}

/*++ sscal
 *****************************************************************************
 * PURPOSE: sscal scales a vector by a constant.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   Uses unrolled loops for increment equal to 1.
 *****************************************************************************
 --*/

void grid_methods::sscal(int n, float sa, float *sx, int incx)
{
    int i, m, mp1, nincx;
    if (n < 0)
        return;
    if (incx == 1)
    {
        /*
             increment equal to 1
      */
        m = n % 5;
        if (m == 0)
            ;
        else
        {
            for (i = 0; i < m; i++)
                sx[i] = sa * sx[i];
            if (n < 5)
                return;
        }

        mp1 = m + 1;
        for (i = mp1; i < n; i += 5)
        {
            sx[i] = sa * sx[i];
            sx[i + 1] = sa * sx[i + 1];
            sx[i + 2] = sa * sx[i + 2];
            sx[i + 3] = sa * sx[i + 3];
            sx[i + 4] = sa * sx[i + 4];
        }
    }
    else
    {
        /*
             code for increment not equal to 1
      */
        nincx = n * incx;
        for (i = 0; i < nincx; i += incx)
            sx[i] = sa * sx[i];
    }
}

/*++ sswap
 *****************************************************************************
 * PURPOSE: Interchanges two vectors.
 *          uses unrolled loops for increments equal to 1.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

void grid_methods::sswap(int n, float *sx, int incx, float *sy, int incy)
{
    float stemp;
    int i, ix, iy, m, mp1;
    if (n < 0)
        return;
    if (incx == 1 && incy == 1)
        ;
    else
    {
        /*
        code for unequal increments or equal increments not equal to 1
      */
        ix = 1;
        iy = 1;
        if (incx < 0)
            ix = (-n + 1) * incx + 1;
        if (incy < 0)
            iy = (-n + 1) * incy + 1;
        for (i = 0; i < n; i++)
        {
            stemp = sx[ix];
            sx[ix] = sy[iy];
            sy[iy] = stemp;
            ix = ix + incx;
            iy = iy + incy;
        }
        return;
    }
    /*
     code for both increments equal to 1
     clean-up loop
   */
    m = n % 3;
    if (m == 0)
        ;
    else
    {
        for (i = 0; i < m; i++)
        {
            stemp = sx[i];
            sx[i] = sy[i];
            sy[i] = stemp;
        }
        if (n < 3)
            return;
    }
    mp1 = m + 1;
    for (i = mp1; i < n; i += 3)
    {
        stemp = sx[i];
        sx[i] = sy[i];
        sy[i] = stemp;
        stemp = sx[i + 1];
        sx[i + 1] = sy[i + 1];
        sy[i + 1] = stemp;
        stemp = sx[i + 2];
        sx[i + 2] = sy[i + 2];
        sy[i + 2] = stemp;
    }
}

/*++ saxpy
 *****************************************************************************
 * PURPOSE: Constant times a vector plus a vector.
 *          uses unrolled loop for increments equal to one.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

void grid_methods::saxpy(int n, float sa, float *sx, int incx, float *sy, int incy)
{
    int i, ix, iy, m, mp1;
    if (n < 0)
        return;
    if (sa == 0.0)
        return;
    if (incx == 1 && incy == 1)
        ;
    else
    {
        /*
         code for unequal increments or equal increments
         not equal to 1
      */
        ix = 1;
        iy = 1;
        if (incx < 0)
            ix = (-n + 1) * incx + 1;
        if (incy < 0)
            iy = (-n + 1) * incy + 1;
        for (i = 0; i < n; i++)
        {
            sy[iy] = sy[iy] + sa * sx[ix];
            ix = ix + incx;
            iy = iy + incy;
        }
        return;
    }
    /*
      code for both increments equal to 1
      clean-up loop
   */
    m = n % 4;
    if (m == 0)
        ;
    else
    {
        for (i = 0; i < m; i++)
            sy[i] = sy[i] + sa * sx[i];
        if (n < 4)
            return;
    }
    mp1 = m + 1;
    for (i = mp1; i < n; i += 4)
    {
        sy[i] = sy[i] + sa * sx[i];
        sy[i + 1] = sy[i + 1] + sa * sx[i + 1];
        sy[i + 2] = sy[i + 2] + sa * sx[i + 2];
        sy[i + 3] = sy[i + 3] + sa * sx[i + 3];
    }
}

/*++ sdot
 *****************************************************************************
 * PURPOSE: Forms the dot product of two vectors.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   uses unrolled loops for increments equal to one.
 *****************************************************************************
 --*/

float grid_methods::sdot(int n, float *sx, int incx, float *sy, int incy)
{
    float stemp = 0.0;
    int i, ix, iy, m, mp1;
    if (n < 0)
        return (0.0);
    if (incx != 1 || incy != 1)
    {
        /*
         code for unequal increments or equal increments not equal to 1
      */
        ix = 1;
        iy = 1;
        if (incx < 0)
            ix = (-n + 1) * incx + 1;
        if (incy < 0)
            iy = (-n + 1) * incy + 1;
        for (i = 0; i < n; i++)
        {
            stemp += sx[ix] * sy[iy];
            ix = ix + incx;
            iy = iy + incy;
        }
        return (stemp);
    }
    /*
      code for both increments equal to 1
      clean-up loop
   */
    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            stemp = stemp + sx[i] * sy[i];
        if (n < 5)
        {
            return (stemp);
        }
    }
    mp1 = m + 1;
    for (i = mp1; i < n; i += 5)
    {
        stemp += sx[i] * sy[i] + sx[i + 1] * sy[i + 1] + sx[i + 2] * sy[i + 2] + sx[i + 3] * sy[i + 3] + sx[i + 4] * sy[i + 4];
    }
    return (stemp);
}

/*++ snrm2
 *****************************************************************************
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   euclidean norm of the n-vector stored in sx[] with storage increment incx .
 *          if    n .le. 0 return with result = 0.
 *          if n .ge. 1 then incx must be .ge. 1
 *          four phase method     using two built-in constants that are
 *          hopefully applicable to all machines.
 *          cutlo = maximum of  sqrt(u/eps)  over all known machines.
 *          cuthi = minimum of  sqrt(v)      over all known machines.
 *          where
 *          eps = smallest no. such that eps + 1. .gt. 1.
 *          u   = smallest positive no.   (underflow limit)
 *          v   = largest  no.            (overflow  limit)
 *          brief outline of algorithm..
 *          phase 1    scans zero components.
 *          move to phase 2 when a component is nonzero and .le. cutlo
 *          move to phase 3 when a component is .gt. cutlo
 *          move to phase 4 when a component is .ge. cuthi/m
 *          where m = n for x() real and m = 2*n for complex.
 *          values for cutlo and cuthi..
 *          from the environmental parameters listed in the imsl converter
 *          document the limiting values are as follows..
 *          cutlo, s.p.   u/eps = 2**(-102) for  honeywell.  close seconds are
 *          univac and dec at 2**(-103)
 *          thus cutlo = 2**(-51) = 4.44089e-16
 *          cuthi, s.p.   v = 2**127 for univac, honeywell, and dec.
 *          thus cuthi = 2**(63.5) = 1.30438e19
 *          cutlo, d.p.   u/eps = 2**(-67) for honeywell and dec.
 *          thus cutlo = 2**(-33.5) = 8.23181d-11
 *          cuthi, d.p.   same as s.p.  cuthi = 1.30438d19
 *          data cutlo, cuthi / 8.232d-11,  1.304d19 /
 *          data cutlo, cuthi / 4.441e-16,  1.304e19 /
 *          data cutlo, cuthi / 4.441e-16,  1.304e19 /
 * NOTES:
 *****************************************************************************
--*/

float grid_methods::snrm2(int n, float *sx, int incx)
{
    int next, nn;
    float cutlo = 4.441e-16f, cuthi = 1.304e19f, hitest, xmax = 0.0f;
    float sum;
    float zero = 0.0, one = 1.0;
    int i, j;

    if (n <= 0)
        return (zero);

    next = 30;
    sum = zero;
    nn = n * incx;

    /*
     begin main loop
   */

    i = 0;
    while (i < nn)
    {
        switch (next)
        {
        case 30:
            if (ABS(sx[i]) > cutlo)
            {
                hitest = cuthi / (float)(n);

                /*
                      phase 3.  sum is mid-range.  no scaling.
               */

                for (j = i; j < nn; j += incx)
                {
                    if (ABS(sx[j]) >= hitest)
                    {
                        i = j;
                        next = 110;
                        sum = (sum / sx[i]) / sx[i];
                        xmax = ABS(sx[i]);
                        sum += powf(sx[i] / xmax, 2);
                        i += incx;
                        continue;
                    }
                    sum += powf(sx[j], 2.f);
                }
#ifdef __sgi
                return (fsqrt(sum));
#else
                return (sqrt(sum));
#endif
            }
            next = 50;
            xmax = zero;
            break;

        case 50:
            /*
              phase 1.  sum is zero
            */

            if (sx[i] == zero)
            {
                i += incx;
                continue;
            }
            if (ABS(sx[i]) > cutlo)
            {
                hitest = cuthi / (float)(n);

                /*
                      phase 3.  sum is mid-range.  no scaling.
               */

                for (j = i; j < nn; j += incx)
                {
                    if (ABS(sx[j]) >= hitest)
                    {
                        i = j;
                        next = 110;
                        sum = (sum / sx[i]) / sx[i];
                        xmax = ABS(sx[i]);
                        sum += powf((sx[i] / xmax), 2);
                        i += incx;
                        continue;
                    }
                    sum += powf(sx[j], 2.f);
                }
                return (sqrt(sum));
            }

            /*
              prepare for phase 2.
            */

            next = 70;
            xmax = ABS(sx[i]);
            sum += powf((sx[i] / xmax), 2);
            i += incx;
            break;

        case 70:

            /*
              phase 2.  sum is small.
              scale to avoid destructive underflow.
            */

            if (ABS(sx[i]) > cutlo)
            {

                /*
                 prepare for phase 3
               */

                sum = (sum * xmax) * xmax;

                /*
                 for real or d.p. set hitest = cuthi/n
                 for complex      set hitest = cuthi/(2*n)
               */

                hitest = cuthi / (float)(n);

                /*
                 phase 3.  sum is mid-range.  no scaling.
               */

                for (j = i; j < nn; j += incx)
                {
                    if (ABS(sx[j]) >= hitest)
                    {
                        i = j;
                        next = 110;
                        sum = (sum / sx[i]) / sx[i];
                        xmax = ABS(sx[i]);
                        sum += powf((sx[i] / xmax), 2);
                        i += incx;
                        continue;
                    }
                    sum += powf(sx[j], 2);
                }
                return (sqrt(sum));
            }

            next = 110;
            break;

        case 110:

            /*
                        common code for phases 2 and 4.
                        in phase 4 sum is large.  scale to avoid overflow.
            */

            if (ABS(sx[i]) > xmax)
            {
                sum = one + sum * powf(xmax / sx[i], 2);
                xmax = ABS(sx[i]);
                i += incx;
                continue;
            }
            sum += powf(sx[i] / xmax, 2);
            i += incx;
            break;
        }
    } /* end of while */

    /*
        compute square root and adjust for scaling.
   */
    return (xmax * sqrt(sum));
}
