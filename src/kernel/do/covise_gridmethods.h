/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_GRIDMETHODS_H
#define COVISE_GRIDMETHODS_H

#include <util/coExport.h>
#include <ostream>
#include <vector>

#define PI 3.14159265358979323846

#ifdef _WIN32

#include <time.h>

#ifdef _MSC_VER
/// @note: rintf is not available in VS;
///    the POSIX definition of rintf is emulated thus
///    (http://www.opengroup.org/onlinepubs/009695399/functions/rintf.html)
/// @param dval expected to be a floating point value
#define rintf(dval) ((_isnan(dval)) ? (dval) : (((((dval)-floor(dval)) >= 0.5f) ? (ceil(dval)) : (floor(dval)))))
#endif // _MSC_VER
#endif // _WIN32

namespace covise
{

class DOEXPORT grid_methods
{
public:
    static int isin_triangle(const float *point,
                             const float *point0,
                             const float *point1,
                             const float *point2,
                             float tolerance);
    static void interpolateInTriangle(float *v_interp, const float *point,
                                      int no_arrays, int array_dim,
                                      const float *const *velo, int c0, int c1, int c2,
                                      const float *p0, const float *p1, const float *p2);
    static void ExtractNormal(float *normal, int base, int second, int third,
                              const float *x_in, const float *y_in, const float *z_in);
    static void ProjectPoint(float *proj_point, const float *point,
                             const int *conn, int elem_cell, int num_of_vert,
                             const float *x_in, const float *y_in, const float *z_in);
    static float tri_surf(float *surf, const float *p0,
                          const float *p1, const float *p2);

    // Given the fem natural coordinates (fem_c) of a point ,
    // in a range from -1 to 1 in an
    // 8-noded cell, and a field at these 8 points,
    // interpolate the field at the point at issue.
    // velos has 8*array_len floats.
    // output: interp points to array_len floats allocated by the caller.
    static void interpElem(float fem_c[3], float *interp,
                           int array_len, const float *velos);

    // used for rectangular grids in binary searches for cell searching
    static int asc_compar_fp(const void *key, const void *fp);
    static int desc_compar_fp(const void *key, const void *fp);

    struct DOEXPORT BoundBox
    {
        float x_min_;
        float y_min_;
        float z_min_;
        float x_max_;
        float y_max_;
        float z_max_;
        float length();
    };

    static void getBoundBox(BoundBox &bbox, int no_v, const int *v_l,
                            const float *x_in, const float *y_in, const float *z_in);

    static float getMaxVel(int no_v, const int *v_l,
                           const float *u, const float *v, const float *w);
    /////////////////////////////////////////////////////////
    // Tetrahedronisation for elements in unstructured grids
    /////////////////////////////////////////////////////////
    // ind>0: positive decomposition
    // ind<0: negative decomposition
    // el, cl: element and connectivity lists of the unstr. grid
    // i: number of the element we want to decompose
    // tel: element list of the "minigrid" tetrahedronised output grid
    // tcl: connectivity list of the "minigrid" tetrahedronised output grid
    //      the entries of the tcl list point to the same coordinate
    //      lists (x_c, y_c, z_c) as are pointed to by the cl input list.
    static void hex2tet(int ind, const int *el, const int *cl, int i, int *tel, int *tcl);
    static void prism2tet(int ind, const int *el, const int *cl, int i, int *tel, int *tcl);
    static void pyra2tet(int ind, const int *el, const int *cl, int i, int *tel, int *tcl);

    //returns the volume of a tetrahedra cell
    static float tetra_vol(const float p0[3], const float p1[3],
                           const float p2[3], const float p3[3]);
    // test if px is in the tetraheder formed by the other 4 points
    static int isin_tetra(const float px[3], const float p0[3], const float p1[3],
                          const float p2[3], const float p3[3], float rel_tol);
    // interpolate in a tetrahedron at "point"
    // c? are the indices in the x_in,y_in and z_in arrays
    // where the grid point coordinates are found
    static void interpolateInTetra(float *v_interp, const float *point,
                                   int no_arrays, int array_dim, const float *const *velo,
                                   int c0, int c1, int c2, int c3,
                                   const float *p0, const float *p1, const float *p2, const float *p3);
    // interpolate in a tetrahedron at "point"
    // c? are the indices in the x_in,y_in and z_in arrays
    // where the grid point coordinates are found
    static int interpolateVInHexa(float *v_interp, const float *point,
                                  const float *const *velo, const int *connl,
                                  const float *x_in, const float *y_in, const float *z_in);
    // interpolate in an hexaeder
    static int interpolateInHexa(float *v_interp, const float *point,
                                 int no_arrays, int array_dim, const float *const *velo,
                                 const int *connl,
                                 const float *x_in, const float *y_in, const float *z_in);

    /******************************/
    /* Support for polyhedral cells */
    /******************************/

    typedef struct
    {
        double x;
        double y;
        double z;
    } POINT3D;

    typedef struct
    {
        int vertex1;
        int vertex2;
        int vertex3;
    } TRIANGLE;

    typedef std::vector<TRIANGLE> TESSELATION;
    typedef std::vector<int> POLYGON;
    typedef std::vector<int>::iterator POLYGON_ITERATOR;

    static double dot_product(POINT3D vector1, POINT3D vector2);
    static POINT3D cross_product(POINT3D vector1, POINT3D vector2);

    static void TesselatePolyhedron(TESSELATION &triangulated_polyhedron, int num_elem_in, int *elem_in, int num_conn_in, int *conn_in, float *xcoord_in, float *ycoord_in, float *zcoord_in);
    static void ComputeBoundingBox(int num_coord_in, float *x_coord_in, float *y_coord_in, float *z_coord_in, POINT3D &box_min, POINT3D &box_max, int &radius /*, vector<POINT3D> &box_vertices*/);

    /* Group of functions required for performing an in-polyhedron test, based on the algorithms of O'Rourke */
    /* (Computational Geometry in C) */
    static bool InBox(POINT3D box_min, POINT3D box_max, POINT3D query_point);
    static void RandomRay(POINT3D &end_point, int radius);
    static char RayBoxTest(POINT3D end_point, POINT3D query_point, POINT3D triangle_box_min, POINT3D triangle_box_max);
    static int PlaneCoeff(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D &normal, double &distance);
    static char RayPlaneIntersection(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D query_point, POINT3D end_point, POINT3D &int_point, int &component_index);
    static int AreaSign(POINT3D new_vertex_1, POINT3D new_vertex_2, POINT3D new_vertex_3);
    static char InTri2D(POINT3D new_vertex_1, POINT3D new_vertex_2, POINT3D new_vertex_3, POINT3D projected_int_point);
    static char InTri3D(float *triangle_x, float *triangle_y, float *triangle_z, int component_index, POINT3D int_point);
    static char InPlane(/*float *triangle_x, float *triangle_y, float *triangle_z, int component_index, POINT3D query_point, POINT3D end_point, POINT3D int_point*/);
    static int VolumeSign(POINT3D a, POINT3D b, POINT3D c, POINT3D d);
    static char RayTriangleCrossing(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D query_point, POINT3D end_point);
    static char RayTriangleIntersection(float *triangle_x, float *triangle_y, float *triangle_z, POINT3D query_point, POINT3D end_point, POINT3D &int_point);
    static char InPolyhedron(float *x_coord_in, float *y_coord_in, float *z_coord_in, POINT3D box_min, POINT3D box_max, POINT3D query_point, POINT3D &end_point, int radius, TESSELATION triangulated_polyhedron);

    /* Field interpolation using mean value barycentric coordinates, based on the algorithm of Ju, Schaefer and Warren */
    /* (Mean Value Coordinates for Closed Triangular Meshes) */
    //  static double InterpolateCellData(int num_coord_in, float *x_coord_in, float *y_coord_in, float *z_coord_in, float *data_in, TESSELATION triangulated_polyhedron, POINT3D query_point);
    static double InterpolateCellData(int num_coord_in, float *x_coord_in, float *y_coord_in, float *z_coord_in, float *data_in, POINT3D query_point);

    /***********************************************************/

    // derivative operators
    static int derivativesAtCenter(float **v_interp[3],
                                   int no_points, int no_arrays, const float *const *velo,
                                   int no_el, int no_vert,
                                   const int *tl, const int *el, const int *connl,
                                   const float *x_in, const float *y_in, const float *z_in);

    /////////////////////////////////////////////////////////
    // OCT_TREE stuff
    /////////////////////////////////////////////////////////
    // probably an int is more efficient than a bitset.
    // 10 bits are used per coordinate.
    typedef int oct_tree_key;

    struct keyBoundBox
    {
        oct_tree_key min_;
        oct_tree_key max_;
    };

    struct constants
    {
        static const int NO_OF_BITS;
        static const int out_of_domain;
    };

    // given a grid bounding box, and a point, calculate oct-tree key
    static void get_oct_tree_key(oct_tree_key &key, const BoundBox &bbox, float point[3], int exc);

    // return 1 if there is intersection of a macrocell and a keyBoundBox
    static int key_bbox_intersection(oct_tree_key macroEl, const keyBoundBox *bbox2, int level);

    class lists; // hide here STL stuff (see cpp file)

    // The oct-tree class
    class octTree
    {
        enum //!!!!!!!!!!!!!!!!!!
        {
            CRIT_LEVEL = 6,
            SMALL_ENOUGH = 20
        };
        lists *lists_;
        int num_grid_cells_;
        const int *keyBBoxes_;
        int fill_son_share(oct_tree_key MacroCell, int son, int elem, int level,
                           unsigned char *son_share);
        int maxOfCountSons(int *);

    public:
        octTree(int num_grid_cells, const int *keyBBoxes);
        int *SonList(oct_tree_key son_key, int *list_cells, int num,
                     unsigned char *son_share, int *count_sons);
        void ModifyLists(int num, int *elements, int offset);
        void DivideOctTree(oct_tree_key MacroCell, int *list_cells,
                           int num, int level, int offset);
        void treePrint(std::ostream &, int, oct_tree_key, int);
        ~octTree();
    };
    /////////////////////////////////////////////////////////
    // OCT_TREE stuff ends
    /////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////
    // See cpp file for info about the following functions
    ///////////////////////////////////////////////////////////////
    static void cell3(int idim, int jdim, int kdim,
                      float *x_in, float *y_in, float *z_in,
                      int *i, int *j, int *k,
                      float *a, float *b, float *g,
                      float x[3], float amat[3][3], float bmat[3][3],
                      int *status);

    static void intp3(int idim, int jdim, int kdim,
                      float *u_in, float *v_in, float *w_in,
                      int i, int j, int k,
                      float a, float b, float g,
                      float *fi);

    static void metr3(int idim, int jdim, int kdim,
                      float *x_in, float *y_in, float *z_in,
                      int i, int j, int k,
                      float a, float b, float g,
                      float amat[3][3], float bmat[3][3],
                      int *idegen, int *status);

    static void padv3(int *first, float cellfr, int direction,
                      int idim, int jdim, int kdim,
                      float *x_in, float *y_in, float *z_in,
                      float *u_in, float *v_in, float *w_in,
                      int *i, int *j, int *k,
                      float *a, float *b, float *g, float x[4],
                      float min_velo, int *status, float *ovel, float *nvel);

    static void ssvdc(float *x, int n, int p, float *s, float *e,
                      float *u, float *v, float *work,
                      int job, int *info);

    static void srot(int n, float *sx, int incx, float *sy,
                     int incy, float c, float s);
    static void srotg(float sa, float sb, float c, float s);
    static void sscal(int n, float sa, float *sx, int incx);
    static void sswap(int n, float *sx, int incx, float *sy, int incy);
    static void saxpy(int n, float sa, float *sx, int incx, float *sy, int incy);
    static float sdot(int n, float *sx, int incx, float *sy, int incy);
    static float snrm2(int n, float *sx, int incx);
    static void ptran3(float amat[3][3], float v[3], float vv[3]);
    static void inv3x3(float a[3][3], float ainv[3][3], int *status);
};

std::ostream &operator<<(std::ostream &, grid_methods::octTree &);
}
#endif
