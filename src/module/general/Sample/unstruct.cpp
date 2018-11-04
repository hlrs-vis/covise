/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "unstruct.h"
#include "Sample.h"

int unstruct_grid::is_in(int el)
{
    int i;
    element = el;

    switch (tl[cur_block][element])
    {
    case TYPE_HEXAEDER:
        for (i = 0; i < 5; i++)
            if (tetra_search(h_tetras[i], lambda))
            {
                tetras = h_tetras[i];
                return 1;
            }
        return 0;
        break;
    case TYPE_PYRAMID:
        for (i = 0; i < 2; i++)
            if (tetra_search(py_tetras[i], lambda))
            {
                tetras = py_tetras[i];
                return 1;
            }
        return 0;
        break;
    case TYPE_TETRAHEDER:
        if (tetra_search(t_tetras, lambda))
        {
            tetras = t_tetras;
            return 1;
        }
        return 0;
        break;
    case TYPE_PRISM:
        for (i = 0; i < 3; i++)
            if (tetra_search(p_tetras[i], lambda))
            {
                tetras = p_tetras[i];
                return 1;
            }
        return 0;
        break;
    }
    return 0;
}

int unstruct_grid::interpolate(float x, float y, float z,
                               int ele,
                               coDoFloat *data,
                               float *result)
{
    punkt[0] = x;
    punkt[1] = y;
    punkt[2] = z;
    if (is_in(ele))
    {
        //if(!fancy)
        // to be implemented
        {

            *result = 0.0;
            float current;

            for (int i = 0; i < 4; i++)
            {
                int elem_index = cl[cur_block][el[cur_block][element] + tetras[i]];
                if (elem_index >= ncoord[cur_block])
                {
                    printf("index ueberlauf: %d\n", elem_index);
                    elem_index = ncoord[cur_block] - 1;
                }
                data->getPointValue(elem_index, &current);
                *result += lambda[i] * current;
                ////*result= 100.0; // wirklich drin
            }
        }
        return 1;
    }
    ////*result= 40.0; // verworfen, weil doch ausserhalb
    ////return 1;
    return 0;
}

int unstruct_grid::interpolate(float x, float y, float z,
                               int ele,
                               coDoVec3 *data,
                               float *result)
{
    punkt[0] = x;
    punkt[1] = y;
    punkt[2] = z;
    if (is_in(ele))
    {
        //if(!fancy)
        // to be implemented
        {
            int i, j;
            for (j = 0; j < 3; j++)
                result[j] = 0.0;
            float current[3];
            for (i = 0; i < 4; i++)
            {
                data->getPointValue(cl[cur_block][el[cur_block][element] + tetras[i]], current);
                for (j = 0; j < 3; j++)
                    result[j] += lambda[i] * current[j];
            }
        }
        return 1;
    }
    return 0;
}

#define SWAP(a, b)  \
    {               \
        temp = (a); \
        (a) = (b);  \
        (b) = temp; \
    }
// Gauss Jordan Elimination aus "Numerical Recipes"
int unstruct_grid::gausj(float *a[], int n, float *b[], int m)
{
    int indxc[10], indxr[10], ipiv[10];
    int i, icol, irow, j, k, l, ll;
    float big, dum, pivinv, temp;
    //	indxc=new int[n];
    /* The integer arrays ipiv, indxr, andindxc are
   used for bookkeeping on the pivoting.*/

    //	indxr=new int[n];
    //	ipiv=new int[n];
    for (j = 0; j < n; j++)
        ipiv[j] = 0;
    for (i = 0; i < n; i++)
    {
        icol = 0;
        irow = 0;
        // This is the main loop over the columns to be reduced.
        big = 0.0;
        for (j = 0; j < n; j++)
            //This is the outer loop of the search for a pivot
            //element.
            if (ipiv[j] != 1)
                for (k = 0; k < n; k++)
                {
                    if (ipiv[k] == 0)
                    {
                        if (fabs(a[j][k]) >= big)
                        {
                            big = fabs(a[j][k]);
                            irow = j;
                            icol = k;
                        }
                    }
                    else if (ipiv[k] > 1)
                    {
                        //printf("gaussj: Singular Matrix-1\n");
                        return -1;
                    }
                }
        ++(ipiv[icol]);

        /*We now have the pivot element, so we interchange rows, if needed, to put the pivot
      element on the diagonal. The columns are not physically interchanged, only relabeled:
      indxc[i], the column of the ith pivot element, is the ith column that is reduced, while
      indxr[i] is the row in which that pivot element was originally located. If indxr[i]
      6 =
      indxc[i] there is an implied column interchange. With this form of bookkeeping, the
      solution b's will end up in the correct order, and the inverse matrix will be scrambled
      by columns.*/

        if (irow != icol)
        {
            for (l = 0; l < n; l++)
                SWAP(a[irow][l], a[icol][l])
            for (l = 0; l < m; l++)
                SWAP(b[irow][l], b[icol][l])
        }
        indxr[i] = irow;
        //We are now ready to divide the pivot row by the
        //pivot element, located at irow and icol.
        indxc[i] = icol;
        if (a[icol][icol] == 0.0)
        {
            // printf("gaussj: Singular Matrix-2");
            return -1;
        }
        pivinv = 1.0f / a[icol][icol];
        a[icol][icol] = 1.0;
        for (l = 0; l < n; l++)
            a[icol][l] *= pivinv;
        for (l = 0; l < m; l++)
            b[icol][l] *= pivinv;
        for (ll = 0; ll < icol; ll++)
        // Next, we reduce the rows...
        { // ...except for the pivot one, of course.
            dum = a[ll][icol];
            a[ll][icol] = 0.0;
            for (l = 0; l < n; l++)
                a[ll][l] -= a[icol][l] * dum;
            for (l = 0; l < m; l++)
                b[ll][l] -= b[icol][l] * dum;
        }
        for (ll = icol + 1; ll < n; ll++)
        // Next, we reduce the rows...
        { // ...except for the pivot one, of course.
            dum = a[ll][icol];
            a[ll][icol] = 0.0;
            for (l = 0; l < n; l++)
                a[ll][l] -= a[icol][l] * dum;
            for (l = 0; l < m; l++)
                b[ll][l] -= b[icol][l] * dum;
        }
    }
    //This is the end of the main loop over columns of the reduction. It only remains to unscram-
    //ble the solution in view of the column interchanges. We do this by interchanging pairs of
    //columns in the reverse order that the permutation was built up.
    for (l = n - 1; l >= 0; l--)
    {
        if (indxr[l] != indxc[l])
            for (k = 0; k < n; k++)
                SWAP(a[k][indxr[l]], a[k][indxc[l]]);
    }
    //And we are done.
    //delete[]ipiv;
    //delete[]indxr;
    //delete[]indxc;
    return 0;
}

int unstruct_grid::tetra_search(int tetra[4], float *coeff)
{
    int i;
    float *x_cl, *y_cl, *z_cl;
    int *cll, *ell;
    x_cl = x_c[cur_block];
    y_cl = y_c[cur_block];
    z_cl = z_c[cur_block];
    cll = cl[cur_block];
    ell = el[cur_block];
    for (i = 0; i < 4; i++)
    {
        matrix[0][i] = x_cl[cll[ell[element] + tetra[i]]];
        matrix[1][i] = y_cl[cll[ell[element] + tetra[i]]];
        matrix[2][i] = z_cl[cll[ell[element] + tetra[i]]];
        matrix[3][i] = 1.0;
    }
    for (int j = 0; j < 3; j++)
        coeff[j] = punkt[j];

    coeff[3] = 1.0;

    float *b[4];
    b[0] = coeff;
    b[1] = coeff + 1;
    b[2] = coeff + 2;
    b[3] = coeff + 3;

    if (gausj(matrix, 4, b, 1) == -1)
        return 0;

    for (i = 0; i < 4; i++)
        if (coeff[i] < (0.0 - eps) || coeff[i] > (1.0 + eps))
            return 0;

    return 1;
}

void unstruct_grid::sample_structured(const coDistributedObject **in_data,
                                      const char *grid_name, coDistributedObject **uni_grid_o,
                                      const char *data_name, coDistributedObject **out_data_o,
                                      int x_s, int y_s, int z_s)
{
    x_size = x_s;
    y_size = y_s;
    z_size = z_s;

    if (flagVector != SCALAR && flagVector != VECTOR)
    {
        fprintf(stderr, "unstruct_grid::sample_structured: USTSDT or USTVDT required\n");
        return;
    }

    // build structured grid according to specification and build dataset too
    coDoUniformGrid *uni_grid = new coDoUniformGrid(grid_name, x_size, y_size, z_size,
                                                    reg_min[0], reg_max[0], reg_min[1],
                                                    reg_max[1], reg_min[2], reg_max[2]);
    *uni_grid_o = uni_grid;

    float *in_val[3] = { NULL, NULL, NULL };
    float *out_val[3] = { NULL, NULL, NULL };
    coDoFloat *in_scal = NULL, *out_scal = NULL;
    coDoVec3 *in_vect = NULL, *out_vect = NULL;
    int no_arrays = 1;
    if (flagVector == SCALAR)
    {
        out_scal = new coDoFloat(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        out_scal->getAddress(&out_val[0]);
        *out_data_o = out_scal;
        in_scal = (coDoFloat *)in_data[0];
        in_scal->getAddress(&in_val[0]);
    }
    else if (flagVector == VECTOR)
    {
        no_arrays = 3;
        out_vect = new coDoVec3(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        out_vect->getAddresses(&out_val[0], &out_val[1], &out_val[2]);
        *out_data_o = out_vect;
        in_vect = (coDoVec3 *)in_data[0];
        in_vect->getAddresses(&in_val[0], &in_val[1], &in_val[2]);
    }

    if (!noDummy)
        return;

    float sx = (reg_max[0] - reg_min[0]) / float(x_s);
    float sy = (reg_max[1] - reg_min[1]) / float(y_s);
    float sz = (reg_max[2] - reg_min[2]) / float(z_s);
    float fill = FLT_MAX;
    if (!nan_flag)
        fill = fill_value;

    coMatrix m;
    m.unity();
    if (transform_inv[0] == NULL)
        transform_inv[0] = &m;
    coVector dx(*(transform_inv[0]) * coVector(sx, 0, 0));
    coVector dy(*(transform_inv[0]) * coVector(0, sy, 0));
    coVector dz(*(transform_inv[0]) * coVector(0, 0, sz));
    coVector base(*(transform_inv[0]) * coVector(reg_min[0], reg_min[1], reg_min[2]));

    int cell[3] = { -1, -1, -1 };
    for (int x = 0; x < x_s; x++)
    {
        for (int y = 0; y < y_s; y++)
        {
            coVector u = base + dx * x + dy * y;
            for (int z = 0; z < z_s; z++)
            {
                int dst_idx = x * y_size * z_size + y * z_size + z;
                float val[3];
                float point[3] = { (float)u[0], (float)u[1], (float)u[2] };
                if (!str_grid[0] || str_grid[0]->interpolateField(val, point, cell, no_arrays, 1, in_val))
                {
                    val[0] = val[1] = val[2] = fill;
                }
                out_val[0][dst_idx] = val[0];
                if (flagVector == VECTOR)
                {
                    out_val[1][dst_idx] = val[1];
                    out_val[2][dst_idx] = val[2];
                }

                u = u + dz;
            }
        }
    }
}

void unstruct_grid::sample_points(const coDistributedObject **in_data,
                                  const char *grid_name, coDistributedObject **uni_grid_o,
                                  const char *data_name, coDistributedObject **out_data_o,
                                  int x_s, int y_s, int z_s, int algo)
{
    x_size = x_s;
    y_size = y_s;
    z_size = z_s;

    if (flagVector != POINT)
    {
        fprintf(stderr, "unstruct_grid::sample_points: POINT required\n");
        return;
    }

    // build structured grid according to specification and build dataset too
    coDoUniformGrid *uni_grid = new coDoUniformGrid(grid_name, x_size, y_size, z_size,
                                                    reg_min[0], reg_max[0], reg_min[1],
                                                    reg_max[1], reg_min[2], reg_max[2]);
    *uni_grid_o = uni_grid;

    coDoFloat *out_data = new coDoFloat(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
    *out_data_o = out_data;

    if (!noDummy)
        return;

    // get address to be able to write into data
    float *scalars;
    out_data->getAddress(&scalars);

    for (int i = 0; i < x_size * y_size * z_size; i++)
    {
        float value = FLT_MAX;
        if (!nan_flag)
            value = fill_value;
        else
            value = FLT_MAX;
        scalars[i] = value;
    }

    float *countHits = new float[x_size * y_size * z_size];
    for (int i = 0; i < x_size * y_size * z_size; i++)
    {
        countHits[i] = 0.f;
    }

    for (int block = 0; block < num_blocks; block++)
    {
        float *x_cl = x_c[block];
        float *y_cl = y_c[block];
        float *z_cl = z_c[block];
        float *data = NULL;

        if (in_data && in_data[block])
        {
            ((coDoFloat *)in_data[block])->getAddress(&data);
        }

        for (int i = 0; i < ncoord[block]; ++i)
        {
            int x_index, y_index, z_index;
            findIndexRint(x_cl[i], y_cl[i], z_cl[i],
                          &x_index, &y_index, &z_index);
            if (x_index < 0 || y_index < 0 || z_index < 0 || x_index >= x_size || y_index >= y_size || z_index >= z_size)
            {
                continue;
            }

            int uniIndex = x_index * y_size * z_size + y_index * z_size + z_index;
            if (data)
            {
                countHits[uniIndex] += data[i];
            }
            else
            {
                countHits[uniIndex] += 1.f;
            }
        }
    }

    float normalization = 1.0;
    if (algo == Sample::POINTS_LOGARITHMIC_NORMALIZED || algo == Sample::POINTS_LINEAR_NORMALIZED)
    {
        normalization = x_size * y_size * z_size / (reg_max[0] - reg_min[0]) / (reg_max[1] - reg_min[1]) / (reg_max[2] - reg_min[2]);
    }

    switch (algo)
    {
    case Sample::POINTS_LOGARITHMIC_NORMALIZED:
    case Sample::POINTS_LOGARITHMIC:
        for (int i = 0; i < x_size * y_size * z_size; ++i)
        {
            if (countHits[i] > 0)
                scalars[i] = log(normalization * countHits[i]);
            else
                scalars[i] = 0.0;
        }
        break;
    case Sample::POINTS_LINEAR:
    case Sample::POINTS_LINEAR_NORMALIZED:
    default:
        for (int i = 0; i < x_size * y_size * z_size; ++i)
        {
            scalars[i] = normalization * countHits[i];
        }
        break;
    }

    delete[] countHits;
}

void unstruct_grid::sample_accu(const coDistributedObject **in_data,
                                const char *grid_name, coDistributedObject **uni_grid_o,
                                const char *data_name, coDistributedObject **out_data_o,
                                int x_s, int y_s, int z_s, float e)
{
    x_size = x_s;
    y_size = y_s;
    z_size = z_s;
    eps = e;
    int i_coor[8], j_coor[8], k_coor[8];
    int min_i, max_i, min_j, max_j, min_k, max_k;
    float *scalars = NULL;
    float result[3];
    float *u_vectors = NULL, *v_vectors = NULL, *w_vectors = NULL;
    int j_start, j_end;

    // build structured grid according to specification and build dataset too
    coDoUniformGrid *uni_grid;
    coDoFloat *out_data = NULL;
    coDoVec3 *out_data_v = NULL;
    uni_grid = new coDoUniformGrid(grid_name, x_size, y_size, z_size, reg_min[0], reg_max[0], reg_min[1], reg_max[1],
                                   reg_min[2], reg_max[2]);
    *uni_grid_o = uni_grid;

    if (flagVector == VECTOR)
    {
        out_data_v = new coDoVec3(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        *out_data_o = out_data_v;
    }
    else
    {
        out_data = new coDoFloat(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        *out_data_o = out_data;
    }

    if (!noDummy)
        return;

    // get adress to be able to write into data
    if (flagVector == VECTOR)
        out_data_v->getAddresses(&u_vectors, &v_vectors, &w_vectors);
    else
        out_data->getAddress(&scalars);

    for (int i = 0; i < x_size * y_size * z_size; i++)
    {
        float value = FLT_MAX;
        if (!nan_flag)
            value = fill_value;
        if (flagVector != VECTOR)
        {
            scalars[i] = value;
        }
        else
        {
            u_vectors[i] = value;
            v_vectors[i] = value;
            w_vectors[i] = value;
        }
    }

    // traverse all elements
    //for (int curr_elem=0; curr_elem < nelem-1; curr_elem++)
    for (int block = 0; block < num_blocks; block++)
    {
        cur_block = block;
        float *x_cl, *y_cl, *z_cl;
        int *cll, *ell;
        x_cl = x_c[block];
        y_cl = y_c[block];
        z_cl = z_c[block];
        cll = cl[block];
        ell = el[block];
        for (int curr_elem = 0; curr_elem < nelem[block]; curr_elem++)
        {
            //if (!(curr_elem%1000))
            //    printf("current element: %d of %d\n", curr_elem, nelem);

            // amount of indices
            int indices = 0;
            // flag if indices are different - different cells are hit
            int different = 0;
            min_i = x_size - 1;
            min_j = y_size - 1;
            min_k = z_size - 1;
            max_i = max_j = max_k = 0;

            // element list is index into vertex list
            // j_start is index of first vertex of cell
            // j_end is index last vertex of cell
            j_start = ell[curr_elem];
            if (curr_elem != nelem[block] - 1)
                j_end = ell[curr_elem + 1];
            else
                j_end = nconn[block];

            for (int j = j_start; j < j_end; j++)
            {

                // calculate the indices of the unigrid which surround
                // this element
                // simply round the coordinates to the next smaller unigrid coordinate
                index(x_cl[cll[j]], y_cl[cll[j]], z_cl[cll[j]],
                      i_coor + indices, j_coor + indices, k_coor + indices);

                if (i_coor[indices] < 0 || j_coor[indices] < 0 || k_coor[indices] < 0 || i_coor[indices] >= x_size || j_coor[indices] >= y_size || k_coor[indices] >= z_size)
                {
                    different = 0;
                    break;
                }

                if (i_coor[indices] >= max_i)
                {
                    max_i = i_coor[indices];
                    if (indices > 0)
                        different = 1;
                }
                if (j_coor[indices] >= max_j)
                {
                    max_j = j_coor[indices];
                    if (indices > 0)
                        different = 1;
                }
                if (k_coor[indices] >= max_k)
                {
                    max_k = k_coor[indices];
                    if (indices > 0)
                        different = 1;
                }

                if (i_coor[indices] <= min_i)
                {
                    min_i = i_coor[indices];
                    if (indices > 0)
                        different = 1;
                }
                if (j_coor[indices] <= min_j)
                {
                    min_j = j_coor[indices];
                    if (indices > 0)
                        different = 1;
                }
                if (k_coor[indices] <= min_k)
                {
                    min_k = k_coor[indices];
                    if (indices > 0)
                        different = 1;
                }
                indices++;
            }

            // we have a candidate for interpolation, if indices are different
            if (different)
            {

                float edge_x, edge_y, edge_z;

                for (int c_i = min_i; c_i <= max_i; c_i++)
                    for (int c_j = min_j; c_j <= max_j; c_j++)
                        for (int c_k = min_k; c_k <= max_k; c_k++)
                        {
                            if (c_i < x_size && c_j < y_size && c_k < z_size)
                            {
                                edge_coordinate(c_i, c_j, c_k, &edge_x, &edge_y, &edge_z);

                                //if ( (curr_elem == 132290) && (c_i == 9)  && (c_j == 27) && (c_k == 19))
                                //    fprintf(stderr,"HIER break\n");

                                if (flagVector != VECTOR)
                                {
                                    if (interpolate(edge_x, edge_y, edge_z, curr_elem,
                                                    (coDoFloat *)(in_data[cur_block]), &result[0]))
                                    {

                                        //// if (scalars[c_i*y_size*z_size+c_j*z_size+c_k] !=100.0)
                                        scalars[c_i * y_size * z_size + c_j * z_size + c_k] = result[0];
                                    }
                                }
                                else
                                {
                                    if (interpolate(edge_x, edge_y, edge_z, curr_elem,
                                                    (coDoVec3 *)(in_data[cur_block]), &result[0]))
                                    {

                                        //// if (scalars[c_i*y_size*z_size+c_j*z_size+c_k] !=100.0)
                                        u_vectors[c_i * y_size * z_size + c_j * z_size + c_k] = result[0];
                                        v_vectors[c_i * y_size * z_size + c_j * z_size + c_k] = result[1];
                                        w_vectors[c_i * y_size * z_size + c_j * z_size + c_k] = result[2];
                                    }
                                }
                            }
                            //else
                            //    fprintf(stderr,"ci cj ck=[%d %d %d]\n", c_i, c_k, c_k);
                        }
            }
        }
    }
}

// the fastest method
void unstruct_grid::sample_holes(const coDistributedObject **in_data,
                                 const char *grid_name, coDistributedObject **uni_grid_o,
                                 const char *data_name, coDistributedObject **out_data_o,
                                 int x_s, int y_s, int z_s)
{
    x_size = x_s;
    y_size = y_s;
    z_size = z_s;
    float *scalars[3] = { NULL, NULL, NULL };

    // build structured grid according to specification and build dataset too
    coDoUniformGrid *uni_grid;
    coDoFloat *out_data = NULL;
    coDoVec3 *out_data_v = NULL;
    uni_grid = new coDoUniformGrid(grid_name, x_size, y_size, z_size, reg_min[0], reg_max[0], reg_min[1], reg_max[1],
                                   reg_min[2], reg_max[2]);
    *uni_grid_o = uni_grid;

    if (flagVector == VECTOR)
    {
        out_data_v = new coDoVec3(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        *out_data_o = out_data_v;
    }
    else
    {
        out_data = new coDoFloat(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        *out_data_o = out_data;
    }

    if (!noDummy)
        return;

    // get adress to be able to write into data
    if (flagVector == VECTOR)
        out_data_v->getAddresses(&scalars[0], &scalars[1], &scalars[2]);
    else
        out_data->getAddress(&scalars[0]);

    for (int i = 0; i < x_size * y_size * z_size; i++)
    {
        float value = FLT_MAX;
        if (!nan_flag)
            value = fill_value;
        scalars[0][i] = value;
        if (flagVector == VECTOR)
        {
            scalars[1][i] = value;
            scalars[2][i] = value;
        }
    }

    int x_index, y_index, z_index;
    int uniIndex;
    int *countHits;
    float *weights;
    float *inSData[3] = { NULL, NULL, NULL };

    countHits = new int[x_size * y_size * z_size];
    memset(countHits, 0, x_size * y_size * z_size * sizeof(int));
    weights = new float[x_size * y_size * z_size];
    for (int i = 0; i < x_size * y_size * z_size; ++i)
    {
        weights[i] = 0.0;
    }

    for (int block = 0; block < num_blocks; block++)
    {
        cur_block = block;
        float *x_cl, *y_cl, *z_cl;
        x_cl = x_c[cur_block];
        y_cl = y_c[cur_block];
        z_cl = z_c[cur_block];

        if (flagVector != VECTOR)
            ((coDoFloat *)(in_data[cur_block]))->getAddress(&inSData[0]);
        else
            ((coDoVec3 *)(in_data[cur_block]))->getAddresses(&inSData[0], &inSData[1], &inSData[2]);

        float i_length;
        for (int i = 0; i < ncoord[cur_block]; ++i)
        {
            findIndexRint(x_cl[i], y_cl[i], z_cl[i],
                          &x_index, &y_index, &z_index);
            if (x_index < 0 || y_index < 0 || z_index < 0 || x_index >= x_size || y_index >= y_size || z_index >= z_size)
                continue;
            i_length = i_Length(x_cl[i], y_cl[i], z_cl[i], x_index, y_index, z_index);
            uniIndex = x_index * y_size * z_size + y_index * z_size + z_index;
            if (countHits[uniIndex] != -1)
            {
                if (i_length == FLT_MAX) // geometric coincidence
                {
                    scalars[0][uniIndex] = inSData[0][i];
                    if (flagVector == VECTOR)
                    {
                        scalars[1][uniIndex] = inSData[1][i];
                        scalars[2][uniIndex] = inSData[2][i];
                    }

                    countHits[uniIndex] = -1;
                }
                else
                {
                    weights[uniIndex] += i_length;
                    if (countHits[uniIndex] == 0)
                    {
                        scalars[0][uniIndex] = i_length * inSData[0][i];
                    }
                    else
                    {
                        scalars[0][uniIndex] += i_length * inSData[0][i];
                    }
                    if (flagVector == VECTOR)
                    {
                        if (countHits[uniIndex] == 0)
                        {
                            scalars[1][uniIndex] = i_length * inSData[1][i];
                            scalars[2][uniIndex] = i_length * inSData[2][i];
                        }
                        else
                        {
                            scalars[1][uniIndex] += i_length * inSData[1][i];
                            scalars[2][uniIndex] += i_length * inSData[2][i];
                        }
                    }
                    ++countHits[uniIndex];
                }
            } // else we have the exact value becaue of a geometric coincidence
        }
    }
    for (int i = 0; i < x_size * y_size * z_size; ++i)
    {
        if (countHits[i] > 0)
            scalars[0][i] /= weights[i];
        if (flagVector == VECTOR)
        {
            if (countHits[i] > 0)
            {
                scalars[1][i] /= weights[i];
                scalars[2][i] /= weights[i];
            }
        }
    }

    // There may be interstitial holes. These are dangerous!!!!
    delete[] weights;
    delete[] countHits;
}

void unstruct_grid::findIndexFloor(float x, float y, float z, int *xi, int *yi, int *zi)
{
    *xi = (int)floor((x_size - 1) * (x - reg_min[0]) / (reg_max[0] - reg_min[0]));
    *yi = (int)floor((y_size - 1) * (y - reg_min[1]) / (reg_max[1] - reg_min[1]));
    *zi = (int)floor((z_size - 1) * (z - reg_min[2]) / (reg_max[2] - reg_min[2]));
}

void unstruct_grid::findIndexCeil(float x, float y, float z, int *xi, int *yi, int *zi)
{
    *xi = (int)ceil((x_size - 1) * (x - reg_min[0]) / (reg_max[0] - reg_min[0]));
    *yi = (int)ceil((y_size - 1) * (y - reg_min[1]) / (reg_max[1] - reg_min[1]));
    *zi = (int)ceil((z_size - 1) * (z - reg_min[2]) / (reg_max[2] - reg_min[2]));
}

void unstruct_grid::findIndexRint(float x, float y, float z, int *xi, int *yi, int *zi)
{
#ifdef _WIN32
    *xi = int((x_size - 1) * (x - reg_min[0]) / (reg_max[0] - reg_min[0]));
    *yi = int((y_size - 1) * (y - reg_min[1]) / (reg_max[1] - reg_min[1]));
    *zi = int((z_size - 1) * (z - reg_min[2]) / (reg_max[2] - reg_min[2]));
#else
    *xi = (int)rint((x_size - 1) * (x - reg_min[0]) / (reg_max[0] - reg_min[0]));
    *yi = (int)rint((y_size - 1) * (y - reg_min[1]) / (reg_max[1] - reg_min[1]));
    *zi = (int)rint((z_size - 1) * (z - reg_min[2]) / (reg_max[2] - reg_min[2]));
#endif
}

float unstruct_grid::i_Length(float x, float y, float z, int i, int j, int k)
{
    float xp, yp, zp, length;

    if (i < 0 || i >= x_size)
        return 0.0;
    if (j < 0 || j >= y_size)
        return 0.0;
    if (k < 0 || k >= z_size)
        return 0.0;

    xp = reg_min[0] + i * (reg_max[0] - reg_min[0]) / (x_size - 1);
    yp = reg_min[1] + j * (reg_max[1] - reg_min[1]) / (y_size - 1);
    zp = reg_min[2] + k * (reg_max[2] - reg_min[2]) / (z_size - 1);

    length = sqrt((x - xp) * (x - xp) + (y - yp) * (y - yp) + (z - zp) * (z - zp));
    if (length == 0.0)
    {
        return FLT_MAX;
    }
    else
    {
        return 1.0f / length;
    }
}

// slower than sample but faster than the precise method...
// char method == 0 -> feature expansion
// char method == 1 -> prevents feature expansion (recommended)
void unstruct_grid::sample_no_holes(const coDistributedObject **in_data,
                                    const char *grid_name, coDistributedObject **uni_grid_o,
                                    const char *data_name, coDistributedObject **out_data_o,
                                    int x_s, int y_s, int z_s, char method)
{
    x_size = x_s;
    y_size = y_s;
    z_size = z_s;
    float *scalars[3] = { NULL, NULL, NULL };

    // build structured grid according to specification and build dataset too
    coDoUniformGrid *uni_grid;
    coDoFloat *out_data = NULL;
    coDoVec3 *out_data_v = NULL;
    uni_grid = new coDoUniformGrid(grid_name, x_size, y_size, z_size, reg_min[0], reg_max[0], reg_min[1], reg_max[1],
                                   reg_min[2], reg_max[2]);
    *uni_grid_o = uni_grid;

    if (flagVector == VECTOR)
    {
        out_data_v = new coDoVec3(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        *out_data_o = out_data_v;
    }
    else
    {
        out_data = new coDoFloat(data_name, noDummy * x_size * noDummy * y_size * noDummy * z_size);
        *out_data_o = out_data;
    }

    if (!noDummy)
        return;

    // get address to be able to write into data
    if (flagVector == VECTOR)
        out_data_v->getAddresses(&scalars[0], &scalars[1], &scalars[2]);
    else
        out_data->getAddress(&scalars[0]);

    float value;
    int i;
    for (i = 0; i < x_size * y_size * z_size; i++)
    {
        if (!nan_flag)
            value = fill_value;
        else
            value = FLT_MAX;
        scalars[0][i] = value;
        if (flagVector == VECTOR)
        {
            scalars[1][i] = value;
            scalars[2][i] = value;
        }
    }

    int x_index, y_index, z_index;
    int x_index_m, y_index_m, z_index_m;
    int x_index_M, y_index_M, z_index_M;
    int uniIndex;
    int *countHits;
    float *weights;
    float *inSData[3];

    countHits = new int[x_size * y_size * z_size];
    memset(countHits, 0, x_size * y_size * z_size * sizeof(int));
    weights = new float[x_size * y_size * z_size];
    for (i = 0; i < x_size * y_size * z_size; ++i)
    {
        weights[i] = 0.0;
    }
    int block;
    for (block = 0; block < num_blocks; block++)
    {
        cur_block = block;

        if (flagVector != VECTOR)
            ((coDoFloat *)(in_data[cur_block]))->getAddress(&inSData[0]);
        else
            ((coDoVec3 *)(in_data[cur_block]))->getAddresses(&inSData[0], &inSData[1], &inSData[2]);

        float i_value[3];
        float i_length;
        for (i = 0; i < nelem[cur_block]; ++i)
        {
            findUniBox(i, &x_index_m, &x_index_M,
                       &y_index_m, &y_index_M,
                       &z_index_m, &z_index_M, method);
            if (x_index_m >= x_size || x_index_M < 0 || y_index_m >= y_size || y_index_M < 0 || z_index_m >= z_size || z_index_M < 0)
                continue;
            if (x_index_m < 0)
                x_index_m = 0;
            if (y_index_m < 0)
                y_index_m = 0;
            if (z_index_m < 0)
                z_index_m = 0;
            if (x_index_M >= x_size)
                x_index_M = x_size - 1;
            if (y_index_M >= y_size)
                y_index_M = y_size - 1;
            if (z_index_M >= z_size)
                z_index_M = z_size - 1;

            for (x_index = x_index_m; x_index <= x_index_M; ++x_index)
                for (y_index = y_index_m; y_index <= y_index_M; ++y_index)
                    for (z_index = z_index_m; z_index <= z_index_M; ++z_index)
                    {
                        if (flagVector != VECTOR)
                            i_value[0] = i_Value(i, inSData[0], x_index, y_index, z_index, &i_length);
                        else
                            i_ValueV(i, inSData, x_index, y_index, z_index, &i_length, i_value);
                        uniIndex = x_index * y_size * z_size + y_index * z_size + z_index;
                        if (countHits[uniIndex] != -1)
                        {
                            if (i_length == FLT_MAX) // geometric coincidence
                            {
                                // this is now the value
                                scalars[0][uniIndex] = i_value[0];
                                // without weight
                                if (flagVector == VECTOR)
                                {
                                    scalars[1][uniIndex] = i_value[1];
                                    scalars[2][uniIndex] = i_value[2];
                                }
                                countHits[uniIndex] = -1;
                            }
                            else
                            {
                                weights[uniIndex] += i_length;
                                if (countHits[uniIndex] == 0)
                                {
                                    scalars[0][uniIndex] = i_value[0];
                                    if (flagVector == VECTOR)
                                    {
                                        scalars[1][uniIndex] = i_value[1];
                                        scalars[2][uniIndex] = i_value[2];
                                    }
                                }
                                else
                                {
                                    scalars[0][uniIndex] += i_value[0];
                                    if (flagVector == VECTOR)
                                    {
                                        scalars[1][uniIndex] += i_value[1];
                                        scalars[2][uniIndex] += i_value[2];
                                    }
                                }
                                ++countHits[uniIndex];
                            }
                        } // else we have the exact value becaue of a geometric coincidence
                    }
        }
    }
    for (i = 0; i < x_size * y_size * z_size; ++i)
        if (countHits[i] > 0)
        {
            scalars[0][i] /= weights[i];
            if (flagVector == VECTOR)
            {
                scalars[1][i] /= weights[i];
                scalars[2][i] /= weights[i];
            }
        }

    delete[] weights;
    delete[] countHits;
}

// finds the smallest unibox that contains
// an element or
// the greatest unibox contained in the realbox of an element
void unstruct_grid::findUniBox(int ele, int *x_index_m, int *x_index_M,
                               int *y_index_m, int *y_index_M,
                               int *z_index_m, int *z_index_M, char method)
{
    *x_index_m = x_size;
    *y_index_m = y_size;
    *z_index_m = z_size;
    *x_index_M = -1;
    *y_index_M = -1;
    *z_index_M = -1;
    float *x_cl, *y_cl, *z_cl;
    int *cll, *ell;
    x_cl = x_c[cur_block];
    y_cl = y_c[cur_block];
    z_cl = z_c[cur_block];
    cll = cl[cur_block];
    ell = el[cur_block];

    int conn_M;
    if (ele == nelem[cur_block] - 1)
    {
        conn_M = nconn[cur_block];
    }
    else
    {
        conn_M = ell[ele + 1];
    }
    int conn_count;
    int i_up, j_up, k_up;
    int i_down, j_down, k_down;
    for (conn_count = ell[ele]; conn_count < conn_M; ++conn_count)
    {
        if (method)
            findIndexCeil(x_cl[cll[conn_count]],
                          y_cl[cll[conn_count]],
                          z_cl[cll[conn_count]],
                          &i_down, &j_down, &k_down);
        else
            findIndexFloor(x_cl[cll[conn_count]],
                           y_cl[cll[conn_count]],
                           z_cl[cll[conn_count]],
                           &i_down, &j_down, &k_down);

        if (*x_index_m > i_down)
            *x_index_m = i_down;
        if (*y_index_m > j_down)
            *y_index_m = j_down;
        if (*z_index_m > k_down)
            *z_index_m = k_down;

        if (method)
            findIndexFloor(x_cl[cll[conn_count]], y_cl[cll[conn_count]], z_cl[cll[conn_count]],
                           &i_up, &j_up, &k_up);
        else
            findIndexCeil(x_cl[cll[conn_count]], y_cl[cll[conn_count]], z_cl[cll[conn_count]],
                          &i_up, &j_up, &k_up);

        if (*x_index_M < i_up)
            *x_index_M = i_up;
        if (*y_index_M < j_up)
            *y_index_M = j_up;
        if (*z_index_M < k_up)
            *z_index_M = k_up;
    }
}

float unstruct_grid::i_Value(int ele, float *inSData,
                             int x_index, int y_index, int z_index,
                             float *i_length)
{
    float ivalue = 0.0;
    *i_length = 0.0;
    int conn_M;
    if (ele == nelem[cur_block] - 1)
    {
        conn_M = nconn[cur_block];
    }
    else
    {
        conn_M = el[cur_block][ele + 1];
    }
    int conn_count;
    float x, y, z, length;
    x = reg_min[0] + x_index * (reg_max[0] - reg_min[0]) / (x_size - 1);
    y = reg_min[1] + y_index * (reg_max[1] - reg_min[1]) / (y_size - 1);
    z = reg_min[2] + z_index * (reg_max[2] - reg_min[2]) / (z_size - 1);
    for (conn_count = el[cur_block][ele]; conn_count < conn_M; ++conn_count)
    {
        length = sqrt((x - x_c[cur_block][cl[cur_block][conn_count]]) * (x - x_c[cur_block][cl[cur_block][conn_count]]) + (y - y_c[cur_block][cl[cur_block][conn_count]]) * (y - y_c[cur_block][cl[cur_block][conn_count]]) + (z - z_c[cur_block][cl[cur_block][conn_count]]) * (z - z_c[cur_block][cl[cur_block][conn_count]]));
        if (length != 0.0)
        {
            *i_length += 1.0f / length;
            ivalue += inSData[cl[cur_block][conn_count]] / length;
        }
        else
        {
            *i_length = FLT_MAX;
            ivalue = inSData[cl[cur_block][conn_count]];
            return ivalue;
        }
    }
    return ivalue;
}

void unstruct_grid::i_ValueV(int ele, float **inSData,
                             int x_index, int y_index, int z_index,
                             float *i_length, float *ivalue)
{
    ivalue[0] = ivalue[1] = ivalue[2] = 0.0;
    *i_length = 0.0;
    int conn_M;
    if (ele == nelem[cur_block] - 1)
    {
        conn_M = nconn[cur_block];
    }
    else
    {
        conn_M = el[cur_block][ele + 1];
    }
    int conn_count;
    float x, y, z, length;
    x = reg_min[0] + x_index * (reg_max[0] - reg_min[0]) / (x_size - 1);
    y = reg_min[1] + y_index * (reg_max[1] - reg_min[1]) / (y_size - 1);
    z = reg_min[2] + z_index * (reg_max[2] - reg_min[2]) / (z_size - 1);
    for (conn_count = el[cur_block][ele]; conn_count < conn_M; ++conn_count)
    {
        length = sqrt((x - x_c[cur_block][cl[cur_block][conn_count]]) * (x - x_c[cur_block][cl[cur_block][conn_count]]) + (y - y_c[cur_block][cl[cur_block][conn_count]]) * (y - y_c[cur_block][cl[cur_block][conn_count]]) + (z - z_c[cur_block][cl[cur_block][conn_count]]) * (z - z_c[cur_block][cl[cur_block][conn_count]]));
        if (length != 0.0)
        {
            *i_length += 1.0f / length;
            ivalue[0] += inSData[0][cl[cur_block][conn_count]] / length;
            ivalue[1] += inSData[1][cl[cur_block][conn_count]] / length;
            ivalue[2] += inSData[2][cl[cur_block][conn_count]] / length;
        }
        else
        {
            *i_length = FLT_MAX;
            ivalue[0] = inSData[0][cl[cur_block][conn_count]];
            ivalue[1] = inSData[1][cl[cur_block][conn_count]];
            ivalue[2] = inSData[2][cl[cur_block][conn_count]];
        }
    }
}

void unstruct_grid::index(float x, float y, float z, int *i, int *j, int *k)
{
    float dummy;
    dummy = ((float)(x_size - 1) * (x - reg_min[0])) / (reg_max[0] - reg_min[0]);
    *i = (int)dummy;
    dummy = ((float)(y_size - 1) * (y - reg_min[1])) / (reg_max[1] - reg_min[1]);
    *j = (int)dummy;
    dummy = ((float)(z_size - 1) * (z - reg_min[2])) / (reg_max[2] - reg_min[2]);
    *k = (int)dummy;
}

void unstruct_grid::edge_coordinate(int i, int j, int k, float *x, float *y, float *z)
{
    *x = reg_min[0] + (float)i * (reg_max[0] - reg_min[0]) / (float)(x_size - 1);
    *y = reg_min[1] + (float)j * (reg_max[1] - reg_min[1]) / (float)(y_size - 1);
    *z = reg_min[2] + (float)k * (reg_max[2] - reg_min[2]) / (float)(z_size - 1);
}

unstruct_grid::unstruct_grid(std::vector<const coDistributedObject *> &grid,
                             vecFlag flag, bool isStrGrid)
    : flagVector(flag)
{
    transform_mat = NULL;
    transform_inv = NULL;
    str_grid = NULL;

    int i, j;
    int h_t[5][4] = {
        { 0, 1, 3, 4 },
        { 3, 4, 7, 6 },
        { 3, 1, 6, 2 },
        { 5, 4, 1, 6 },
        { 4, 3, 6, 1 }
    };
    x_c = y_c = z_c = NULL;
    cl = el = tl = NULL;
    ncoord = nconn = nelem = 0;
    for (i = 0; i < 5; i++)
    {
        h_tetras[i] = new int[4];
        for (j = 0; j < 4; j++)
            h_tetras[i][j] = h_t[i][j];
    }

    int p_t[3][4] = {
        { 0, 1, 2, 3 },
        { 0, 2, 3, 4 },
        { 2, 3, 4, 5 }
    };

    for (i = 0; i < 3; i++)
    {
        p_tetras[i] = new int[4];
        for (j = 0; j < 4; j++)
            p_tetras[i][j] = p_t[i][j];
    }

    int py_t[2][4] = {
        { 0, 1, 2, 4 },
        { 0, 2, 3, 4 }
    };
    py_tetras[0] = new int[4];
    py_tetras[1] = new int[4];
    for (j = 0; j < 4; j++)
    {
        py_tetras[0][j] = py_t[0][j];
        py_tetras[1][j] = py_t[1][j];
    }
    int t_t[4] = { 0, 1, 2, 3 };
    t_tetras = new int[4];

    for (j = 0; j < 4; j++)
        t_tetras[j] = t_t[j];

    // get addresses & size of grid
    size_t num_grid;
    num_blocks = grid.size();
    //fprintf(stderr, "numblocks=%lu\n", (unsigned long)grid.size());
    if (num_blocks == 0)
    {
        noDummy = 0;
    }
    else
    {
        noDummy = 1;
    }
    /* @@@
                   if(num==0 && numc > 0){
                      num_blocks=1;
                      noDummy=0;
                   } else {
                      num_blocks=num;
                      noDummy=1;
                   }
   */
    x_c = new float *[num_blocks];
    y_c = new float *[num_blocks];
    z_c = new float *[num_blocks];
    cl = new int *[num_blocks];
    el = new int *[num_blocks];
    tl = new int *[num_blocks];
    ncoord = new int[num_blocks];
    nconn = new int[num_blocks];
    nelem = new int[num_blocks];
    str_sz_x = str_sz_y = str_sz_z = NULL;
    str_grid = NULL;
    if (flagVector == POINT)
    {
        for (num_grid = 0; num_grid < num_blocks; num_grid++)
        {
            coDoPoints *the_points = (coDoPoints *)(grid[num_grid]);
            the_points->getAddresses(&x_c[num_grid], &y_c[num_grid], &z_c[num_grid]);
            el[num_grid] = cl[num_grid] = tl[num_grid] = NULL;
            ncoord[num_grid] = nelem[num_grid] = the_points->getNumPoints();
            nconn = NULL;
        }
    }
    else if (isStrGrid && (flagVector == SCALAR || flagVector == VECTOR))
    {
        transform_mat = new coMatrix *[num_blocks];
        transform_inv = new coMatrix *[num_blocks];
        str_grid = new coDoAbstractStructuredGrid *[num_blocks];
        str_sz_x = new int[num_blocks];
        str_sz_y = new int[num_blocks];
        str_sz_z = new int[num_blocks];
        for (num_grid = 0; num_grid < num_blocks; num_grid++)
        {
            transform_mat[num_grid] = transform_inv[num_grid] = NULL;

            coDoAbstractStructuredGrid *the_grid = (coDoAbstractStructuredGrid *)(grid[num_grid]);
            str_grid[num_grid] = the_grid;
            if (!the_grid)
                continue;
            the_grid->getGridSize(&str_sz_x[num_grid], &str_sz_y[num_grid], &str_sz_z[num_grid]);
            if (the_grid->isType("UNIGRD"))
            {
                coDoUniformGrid *uni = (coDoUniformGrid *)the_grid;
                ncoord[num_grid] = 8;
                x_c[num_grid] = new float[ncoord[num_grid]];
                y_c[num_grid] = new float[ncoord[num_grid]];
                z_c[num_grid] = new float[ncoord[num_grid]];
                float xmin, xmax, ymin, ymax, zmin, zmax;
                uni->getMinMax(&xmin, &xmax,
                               &ymin, &ymax,
                               &zmin, &zmax);

                x_c[num_grid][0] = x_c[num_grid][1] = x_c[num_grid][2] = x_c[num_grid][3] = xmin;
                x_c[num_grid][4] = x_c[num_grid][5] = x_c[num_grid][6] = x_c[num_grid][7] = xmax;

                y_c[num_grid][0] = y_c[num_grid][5] = y_c[num_grid][6] = y_c[num_grid][3] = ymin;
                y_c[num_grid][4] = y_c[num_grid][1] = y_c[num_grid][2] = y_c[num_grid][7] = ymax;

                z_c[num_grid][0] = z_c[num_grid][5] = z_c[num_grid][2] = z_c[num_grid][4] = zmin;
                z_c[num_grid][3] = z_c[num_grid][1] = z_c[num_grid][6] = z_c[num_grid][7] = zmax;
            }
            else if (the_grid->isType("RCTGRD"))
            {
                coDoRectilinearGrid *rct = (coDoRectilinearGrid *)the_grid;
                ncoord[num_grid] = str_sz_x[num_grid];
                if (ncoord[num_grid] < str_sz_y[num_grid])
                    ncoord[num_grid] = str_sz_y[num_grid];
                if (ncoord[num_grid] < str_sz_z[num_grid])
                    ncoord[num_grid] = str_sz_z[num_grid];
                x_c[num_grid] = new float[ncoord[num_grid]];
                y_c[num_grid] = new float[ncoord[num_grid]];
                z_c[num_grid] = new float[ncoord[num_grid]];

                float *x, *y, *z;
                rct->getAddresses(&x, &y, &z);

                for (int i = 0; i < ncoord[num_grid]; i++)
                {
                    if (str_sz_x[num_grid] > i)
                        x_c[num_grid][i] = x[i];
                    else
                        x_c[num_grid][i] = x_c[num_grid][0];

                    if (str_sz_y[num_grid] > i)
                        y_c[num_grid][i] = y[i];
                    else
                        y_c[num_grid][i] = y_c[num_grid][0];

                    if (str_sz_z[num_grid] > i)
                        z_c[num_grid][i] = z[i];
                    else
                        z_c[num_grid][i] = z_c[num_grid][0];
                }
            }
            else if (the_grid->isType("STRGRD"))
            {
                coDoStructuredGrid *str = (coDoStructuredGrid *)the_grid;
                ncoord[num_grid] = str_sz_x[num_grid] * str_sz_y[num_grid] * str_sz_z[num_grid];
                x_c[num_grid] = new float[ncoord[num_grid]];
                y_c[num_grid] = new float[ncoord[num_grid]];
                z_c[num_grid] = new float[ncoord[num_grid]];

                float *x, *y, *z;
                str->getAddresses(&x, &y, &z);
                memcpy(x_c[num_grid], x, sizeof(float) * ncoord[num_grid]);
                memcpy(y_c[num_grid], y, sizeof(float) * ncoord[num_grid]);
                memcpy(z_c[num_grid], z, sizeof(float) * ncoord[num_grid]);
            }

            el[num_grid] = cl[num_grid] = tl[num_grid] = NULL;
            nelem[num_grid] = 1;
            nconn = NULL;

            const char *attr = the_grid->getAttribute("Transformation");
            if (attr)
            {
                double mat[16];
                const char *p = attr;
                for (int i = 0; i < 16; i++)
                {
                    int n = 0;
                    if (sscanf(p, "%lf%n", &mat[i], &n) < 1)
                    {
                        cerr << "Sample:unstruct_grid: sscanf failed" << endl;
                    }
                    p += n;
                }
                transform_mat[num_grid] = new coMatrix(mat);

                transform_inv[num_grid] = new coMatrix(transform_mat[num_grid]->invers());
            }
            else
            {
                transform_mat[num_grid] = transform_inv[num_grid] = NULL;
            }
        }
    }
    else
    {
        for (num_grid = 0; num_grid < num_blocks; num_grid++)
        {
            coDoUnstructuredGrid *the_grid = (coDoUnstructuredGrid *)(grid[num_grid]);
            the_grid->getAddresses(&el[num_grid], &cl[num_grid], &x_c[num_grid], &y_c[num_grid], &z_c[num_grid]);
            the_grid->getTypeList(&tl[num_grid]);
            the_grid->getGridSize(&nelem[num_grid], &nconn[num_grid], &ncoord[num_grid]);
        }
    }
    for (i = 0; i < 4; i++)
        matrix[i] = new float[4];
    nan_flag = false;
    fill_value = 0;
    eps = 0.0;
}

void unstruct_grid::automaticBoundBox()
{
    int i, j;

    reg_max[0] = reg_max[1] = reg_max[2] = -FLT_MAX;
    reg_min[0] = reg_min[1] = reg_min[2] = FLT_MAX;

    for (j = 0; j < num_blocks; ++j)
    {
        if (transform_mat && transform_mat[j])
        {
            cerr << *(transform_mat[j]) << endl;
            for (i = 0; i < ncoord[j]; i++)
            {
                coVector v(x_c[j][i], y_c[j][i], z_c[j][i]);
                coVector u(*(transform_mat[j]) * v);
                for (int c = 0; c < 3; c++)
                {
                    if (u[c] < reg_min[c])
                        reg_min[c] = (float)u[c];
                    if (u[c] > reg_max[c])
                        reg_max[c] = (float)u[c];
                }
            }
        }
        else
        {
            for (i = 0; i < ncoord[j]; i++)
            {
                if (x_c[j][i] <= reg_min[0])
                    reg_min[0] = x_c[j][i];
                if (y_c[j][i] <= reg_min[1])
                    reg_min[1] = y_c[j][i];
                if (z_c[j][i] <= reg_min[2])
                    reg_min[2] = z_c[j][i];
                if (x_c[j][i] >= reg_max[0])
                    reg_max[0] = x_c[j][i];
                if (y_c[j][i] >= reg_max[1])
                    reg_max[1] = y_c[j][i];
                if (z_c[j][i] >= reg_max[2])
                    reg_max[2] = z_c[j][i];
            }
        }
    }
}

void unstruct_grid::manualBoundBox(float x1,
                                   float y1, float z1, float x2,
                                   float y2, float z2)
{
    reg_min[0] = (x1 < x2) ? x1 : x2;
    reg_min[1] = (y1 < y2) ? y1 : y2;
    reg_min[2] = (z1 < z2) ? z1 : z2;
    reg_max[0] = (x1 < x2) ? x2 : x1;
    reg_max[1] = (y1 < y2) ? y2 : y1;
    reg_max[2] = (z1 < z2) ? z2 : z1;
}

void unstruct_grid::mat_mult(float *x, float *y, float *z, const float *mat)
{
    float in[4] = { *x, *y, *z, 1. };
    float out[4] = { 0., 0., 0., 0. };

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            out[j] += mat[j * 4 + i] * in[i];
        }
    }
    if (fabs(out[3]) > eps)
    {
        for (int i = 0; i < 3; i++)
            out[i] /= out[3];
    }

    *x = out[0];
    *y = out[1];
    *z = out[2];
}

unstruct_grid::~unstruct_grid()
{
    delete[] x_c;
    delete[] y_c;
    delete[] z_c;
    delete[] cl;
    delete[] el;
    delete[] tl;
    delete[] ncoord;
    delete[] nconn;
    delete[] nelem;
}
