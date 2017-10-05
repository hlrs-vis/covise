/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _UNSTRUCT_H
#define _UNSTRUCT_H
#include <util/coMatrix.h>
#include <api/coModule.h>
using namespace covise;

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <float.h>
#include <util/coviseCompat.h>
#include <vector>

// The interfaces between unstruct_grid and Sample classes
// might be reduced if unstruct_grid were embedded in Sample...
class unstruct_grid
{
private:
    // barycentric coordinates of last search
    float lambda[4];

    // found Tetrahedra of the last search
    int *tetras;

    //  the current point
    float punkt[3];

    // the current element
    int element;

    // matrix for barycentric coordinates
    float *matrix[4];

    // calculates barycentric coordinates & decides if point is within
    // tetrahedron
    int tetra_search(int tetra[4], float *coeff);

    // Gauss Jordan Matrix inversion
    int gausj(float *a[], int n, float *b[], int m);

    // Tetrahedra of Hexaedron
    int *h_tetras[5];

    // Tetrahedra in prism
    int *p_tetras[3];

    //Tetrahedra in pyramid
    int *py_tetras[2];

    // Tetrahedron pur
    int *t_tetras;

    // make NAN
    bool nan_flag;

    // Fill Value
    float fill_value;

    // eps is added to baricentric coords to avoid numerical problems
    float eps;

    // Is the current point in element
    int is_in(int element);

    size_t num_blocks;
    int cur_block;
    int noDummy;
    float **x_c;
    float **y_c;
    float **z_c;
    int **cl;
    int **el;
    int **tl;
    int *ncoord, *nconn, *nelem;
    int *str_sz_x, *str_sz_y, *str_sz_z;
    coDoAbstractStructuredGrid **str_grid;
    coMatrix **transform_mat;
    coMatrix **transform_inv;

    //        int flagVector;

    // Data not in Covise-Unstructured Grid :-((
    float reg_min[3], reg_max[3];

    // Information about unstructured Grid
    int x_size, y_size, z_size;

    // calculates the index of the surrounding structured grid element
    // (x-xmin) : unigrid_cell_size
    void index(float x, float y, float z, int *i, int *j, int *k);

    // Find the nearest point of the uniform grid
    void findIndexFloor(float x, float y, float z, int *i, int *j, int *k);
    void findIndexCeil(float x, float y, float z, int *i, int *j, int *k);
    void findIndexRint(float x, float y, float z, int *i, int *j, int *k);

    // returns inverse of the length between point and node of uniform grid
    float i_Length(float x, float y, float z, int i, int j, int k);

    void findUniBox(int i, int *x_index_m, int *x_index_M,
                    int *y_index_m, int *y_index_M,
                    int *z_index_m, int *z_index_M, char method);

    float i_Value(int i, float *, int x_index, int y_index, int z_index,
                  float *i_length);

    void i_ValueV(int i, float **, int x_index, int y_index, int z_index,
                  float *i_length, float *outvect);

    // calculates coordinate of lower, front, left edge of structured grid;
    void edge_coordinate(int i, int j, int k, float *x, float *y, float *z);

    // mult 3-vector with 4x4-matrix
    void mat_mult(float *x, float *y, float *z, const float *mat);

public:
    //	unstruct_grid(std::vector<coDistributedObject *>& grid,int flag);
    enum vecFlag
    {
        DONTKNOW,
        SCALAR,
        VECTOR,
        POINT
    } flagVector;
    unstruct_grid(std::vector<const coDistributedObject *> &grid, vecFlag flag, bool isStrGrid);

    void automaticBoundBox();

    void manualBoundBox(float xmin, float xmax, float ymin, float ymax,
                        float zmin, float zmax);

    int interpolate(float x, float y, float z,
                    int element,
                    coDoFloat *data,
                    float *result);

    int interpolate(float x, float y, float z,
                    int element,
                    coDoVec3 *data,
                    float *result);

    void sample_accu(const coDistributedObject **in_data,
                     const char *grid_name, coDistributedObject **grid,
                     const char *data_name, coDistributedObject **out_data,
                     int x_size, int y_size, int z_size, float eps);

    void sample_holes(const coDistributedObject **in_data,
                      const char *grid_name, coDistributedObject **grid,
                      const char *data_name, coDistributedObject **out_data,
                      int x_size, int y_size, int z_size);

    void sample_no_holes(const coDistributedObject **in_data,
                         const char *grid_name, coDistributedObject **grid,
                         const char *data_name, coDistributedObject **out_data,
                         int x_size, int y_size, int z_size, char method);

    void sample_points(const coDistributedObject **in_data,
                       const char *grid_name, coDistributedObject **grid,
                       const char *data_name, coDistributedObject **out_data,
                       int x_size, int y_size, int z_size,
                       int algo);

    void sample_structured(const coDistributedObject **in_data,
                           const char *grid_name, coDistributedObject **grid,
                           const char *data_name, coDistributedObject **out_data,
                           int x_size, int y_size, int z_size);

    void getMinMax(const float *&min, const float *&max)
    {
        min = reg_min;
        max = reg_max;
    }

    void set_value(bool n, float v)
    {
        nan_flag = n;
        fill_value = v;
    }
    ~unstruct_grid();
};
#endif // _UNSTRUCT_H
