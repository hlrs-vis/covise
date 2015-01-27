/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CUTTINGSURFACE_H
#define CO_CUTTINGSURFACE_H

#include <map>

#include "CuttingSurfaceGPMUtil.h"

#include <do/coDoData.h>

#define UP 0
#define DOWN 1
#define FINDX 1
#define FINDY 2
#define FINDZ 3
#define TOO_SMALL 4 // For rectilinear grids: if dimension is lower than TWO_SMALL we just cut every cube
#define CHECK_Z (fabs(planek) > 0.15)
#define CREFN 20 // the degree of refinement for cylinder surfaces

namespace covise
{
class coDistributedObject;
class coDoPolygons;
class coDoTriangleStrips;
class coDoUnstructuredGrid;
class coDoUniformGrid;
class coDoRectilinearGrid;
class coDoStructuredGrid;
class CuttingSurface;

typedef struct NodeInfo_s
{
    float dist;
    int side;
    int targets[12];
    int vertice_list[12];
} NodeInfo;

typedef struct cutting_info_s
{
    int node_pairs[12];
    int nvert;
} cutting_info;

typedef struct border_s
{
    int upper;
    int lower;
} border;

class ALGEXPORT AttributeContainer
{
private:
    const coDistributedObject *p_obj_;
    int no_attrs_;
    const char **attr_;
    const char **setting_;

public:
    AttributeContainer(const coDistributedObject *p_obj);
    const coDistributedObject *getPObj();
    void addAttributes(coDistributedObject *p_obj);
    void addAttributes(coDistributedObject *p_obj, const char *probeAttr);
};

class ALGEXPORT Plane
{
    friend class Isoline;
    friend class STR_Plane;
    friend class UNI_Plane;
    friend class RECT_Plane;
    friend class CELL_Plane;
    friend class POLYHEDRON_Plane;

protected:
    bool unstr_;

private:
    void initialize();

    const coDoUnstructuredGrid *grid_in;
    const coDoStructuredGrid *sgrid_in;
    const coDoUniformGrid *ugrid_in;
    const coDoRectilinearGrid *rgrid_in;

    coDoPolygons *polygons_out;
    coDoTriangleStrips *strips_out;
    coDoVec3 *normals_out;
    coDoFloat *s_data_out;
    coDoVec3 *v_data_out;

    int *el, *cl, *tl;
    float *x_in;
    float *y_in;
    float *z_in;
    float *s_in;
    unsigned char *bs_in;
    float *i_in;
    float *u_in;
    float *v_in;
    float *w_in;
    int num_nodes;
    int num_elem;
    int num_faces;
    int num_cells;
    int num_edges;
    int num_triangles;
    int num_strips;
    int num_vertices;
    int *vertice_list;
    int *ts_vertice_list;
    int *ts_line_list;
    int *vertex;

    int num_coords;
    int max_coords; // max. number of allocated coords
    int maxPolyPerVertex; //maximal number of polygons dor one vertex

    float *coords_x;
    float *coords_y;
    float *coords_z;
    float *coord_x;
    float *coord_y;
    float *coord_z;
    float *V_Data_U;
    float *V_Data_V;
    float *V_Data_W;
    float *S_Data;
    float *I_Data;
    float *V_Data_U_p;
    float *V_Data_V_p;
    float *V_Data_W_p;
    float *S_Data_p;
    float *I_Data_p;
    float x_min, x_max, y_min, y_max, z_min, z_max;
    int x_size, y_size, z_size;
    NodeInfo *node_table;
    int Datatype;

    float planei, planej, planek;
    float startx, starty, startz;
    float myDistance;
    float radius;
    int gennormals, option, genstrips;
    char *iblank;

    float x_minb, y_minb, z_minb, x_maxb, y_maxb, z_maxb;

    static float gsin(float angle);
    static float gcos(float angle);
    static int trs2pol(int nb_con, int nb_tr, int *trv, int *tr_list, int *plv, int *pol_list);
    int DummyData(int dtype, float **data1, float **data2, float **data3);
    static int DummyNormals(int);
    static void fill_normals(float *u_out, float *v_out, float *w_out,
                             const float *coords_x, const float *coords_y, const float *coords_z,
                             int nb_coords, int param_option,
                             float pla[3], float rad, float start[3]);
    static int check_orientation(float *x4, float *y4, float *z4, float planei,
                                 float planej, float planek);
    static void preserve_inertia(const float *x8,
                                 const float *y8, const float *z8,
                                 float *x4, float *y4, float *z4,
                                 float pli, float plj, float plk);
    static void buildSphere(float *xSphere, float *ySphere, float *zSphere,
                            float xPoint, float yPoint, float zPoint, float rad);
    static void build_SphereStrips(int *tsl, int *vl);
    static void border_proj(float *px, float *py, float *pz,
                            float xminb,
                            float xmaxb,
                            float yminb,
                            float ymaxb,
                            float zminb,
                            float zmaxb,
                            float pli, float plj, float plk, float distance);

public:
    static coDoTriangleStrips *dummy_tr_strips(const char *name,
                                               float x_minb,
                                               float x_maxb,
                                               float y_minb,
                                               float y_maxb,
                                               float z_minb,
                                               float z_maxb, int param_option,
                                               float pli, float plj, float plk, float distance,
                                               float strx, float stry, float strz);
    static coDoPolygons *dummy_polygons(const char *name,
                                        float x_minb,
                                        float x_maxb,
                                        float y_minb,
                                        float y_maxb,
                                        float z_minb,
                                        float z_maxb, int param_option,
                                        float pli, float plj, float plk, float distance,
                                        float strx, float stry, float strz);

    static coDoVec3 *dummy_normals(const char *nname,
                                   float *coords_x, float *coords_y, float *coords_z, int,
                                   float pla[3], float rad, float start[3]);

    Plane();

    Plane(int n_elem, int n_nodes, int Type, int *p_el, int *p_cl, int *p_tl,
          float *p_x_in, float *p_y_in, float *p_z_in,
          float *p_s_in, unsigned char *p_bs_in, float *p_i_in,
          float *p_u_in, float *p_v_in, float *p_w_in,
          const coDoStructuredGrid *p_sgrid_in,
          const coDoUnstructuredGrid *p_grid_in,
          float vertexRatio, int maxPoly,
          float planei_, float planej_, float planek_, float startx_,
          float starty_, float startz_, float myDistance_, float radius_,
          int gennormals_, int option_, int genstrips_, char *ib);
    virtual ~Plane();

    int cur_line_elem; // counter for line elements

    bool add_vertex(int n1, int n2);
    void add_vertex(int n1, int n2, int x, int y, int z, int u, int v, int w);
    virtual bool createPlane();
    virtual void createStrips();

    virtual void createcoDistributedObjects(const char *Data_name_scal, const char *Data_name_vect,
                                            const char *Normal_name, const char *Triangle_name,
                                            AttributeContainer &gridAttrs, AttributeContainer &dataAttrs);

    void get_vector_data(int *numc, float **x, float **y, float **z,
                         float **u, float **v, float **w)
    {
        *numc = num_coords;
        *x = coords_x;
        *y = coords_y;
        *z = coords_z;
        *u = V_Data_U;
        *v = V_Data_V;
        *w = V_Data_W;
    }

    void get_scalar_data(int *numc, float **x, float **y, float **z, float **s)
    {
        *numc = num_coords;
        *s = S_Data;
        *x = coords_x;
        *y = coords_y;
        *z = coords_z;
    }

    coDoPolygons *get_obj_pol()
    {
        return polygons_out;
    }
    coDoTriangleStrips *get_obj_strips()
    {
        return strips_out;
    }
    coDoVec3 *get_obj_normal()
    {
        return normals_out;
    }
    coDoFloat *get_obj_scalar()
    {
        return s_data_out;
    }
    coDoVec3 *get_obj_vector()
    {
        return v_data_out;
    }
};

class ALGEXPORT POLYHEDRON_Plane : public Plane
{
private:
    /***************/
    /* Output Data */
    /***************/

    int num_elem_out;
    int num_conn_out;
    int num_coord_out;
    int *elem_out;
    int *conn_out;
    float *x_coord_out;
    float *y_coord_out;
    float *z_coord_out;
    float *data_out_s;
    float *data_out_u;
    float *data_out_v;
    float *data_out_w;

public:
    POLYHEDRON_Plane(int n_elem, int n_nodes, int Type,
                     int *p_el, int *p_cl, int *p_tl,
                     float *p_x_in, float *p_y_in, float *p_z_in,
                     float *p_s_in, unsigned char *p_bs_in, float *p_i_in,
                     float *p_u_in, float *p_v_in, float *p_w_in,
                     const coDoStructuredGrid *p_sgrid_in,
                     const coDoUnstructuredGrid *p_grid_in,
                     int maxPoly,
                     float planei_, float planej_, float planek_, float startx_,
                     float starty_, float startz_, float myDistance_,
                     float radius_, int gennormals_, int option_,
                     int genstrips_, char *ib);

    ~POLYHEDRON_Plane();

    virtual bool createPlane();

    virtual coDistributedObject *create_data_output(vector<float> &data_vector, const char *data_obj_name);

    virtual void createcoDistributedObjects(const char *Data_name_scal, const char *Data_name_vect,
                                            const char *Normal_name, const char *Triangle_name,
                                            AttributeContainer &gridAttrs, AttributeContainer &dataAttrs);
    virtual void create_contour(float *x_coord_in, float *y_coord_in, float *z_coord_in, int *elem_in, int *conn_in, int num_coord_in, int num_elem_in, int num_conn_in,
                                float *sdata_in, float *udata_in, float *vdata_in, float *wdata_in, CONTOUR &capping_contour, PLANE_EDGE_INTERSECTION_VECTOR &intsec_vector, vector<float> &s_data, vector<float> &u_data_vector, vector<float> &v_data_vector, vector<float> &w_data_vector, EDGE_VECTOR &unit_normal_vector, EDGE_VECTOR &plane_base_x, EDGE_VECTOR &plane_base_y, float p);
};

class ALGEXPORT STR_Plane : public Plane
{
    friend class Isoline;

public:
    STR_Plane(int n_elem, int n_nodes, int Type,
              int *p_el, int *p_cl, int *p_tl,
              float *p_x_in, float *p_y_in, float *p_z_in,
              float *p_s_in, unsigned char *p_bs_in, float *p_i_in,
              float *p_u_in, float *p_v_in, float *p_w_in,
              const coDoStructuredGrid *p_sgrid_in,
              const coDoUnstructuredGrid *p_grid_in,
              int p_x_size, int p_y_size, int p_z_size, int maxPoly,
              float planei_, float planej_, float planek_, float startx_,
              float starty_, float startz_, float myDistance_,
              float radius_, int gennormals_, int option_,
              int genstrips_, char *ib)
        : Plane(n_elem, n_nodes, Type, p_el, p_cl, p_tl,
                p_x_in, p_y_in, p_z_in, p_s_in, p_bs_in, p_i_in,
                p_u_in, p_v_in, p_w_in, p_sgrid_in, p_grid_in,
                -1.0, maxPoly,
                planei_, planej_, planek_, startx_, starty_, startz_, myDistance_, radius_, gennormals_, option_, genstrips_, ib)
    {
        unstr_ = false;
        x_size = p_x_size;
        y_size = p_y_size;
        z_size = p_z_size;
    }
    bool createPlane();
};

class ALGEXPORT UNI_Plane : public Plane
{
    friend class Isoline;

public:
    UNI_Plane(int n_elem, int n_nodes, int Type, int *el, int *cl,
              int *tl, float *x_in, float *y_in, float *z_in, float *s_in, unsigned char *p_bs_in, float *i_in, float *u_in, float *v_in, float *w_in, const coDoUniformGrid *p_ugrid_in, float p_x_min, float p_x_max, float p_y_min, float p_y_max, float p_z_min, float p_z_max, int p_x_size, int p_y_size, int p_z_size, int maxPoly, char *ib);
    //virtual bool createPlane();
};
}
#if 0
#if (!defined(CO_ia64icc) || (__GNUC__ < 4))
template class ALGEXPORT std::map< covise::border **,int>;
#endif
#endif

namespace covise
{
class ALGEXPORT RECT_Plane : public Plane
{
    friend class Isoline;

private: // additions to optimize speed
    float sradius; // square of radius

    int xsize, ysize, zsize;
    int xdim, ydim, zdim;
    float xmin, xmax, ymin, ymax, zmin, zmax;
    float *xunit, *yunit, *zunit;
    int xind_min, xind_max, xind_start, yind_min, yind_max, zind_min, zind_max, zind_start;

    std::map<border **, int> _SpaghettiAntidot;
    border **sym_cutting_cubes; // addition for all symmetrical objects
    border **cutting_cubes; // the cubes we really have to work on

    void necessary_cubes(int option);
    inline bool is_between(float a, float b, float c);
    // find index of a in array
    int find(float a, char find_what, char mode, bool checkborders);

    float min_of_four(float a, float b, float c, float d, float cmin, float cmax);
    float max_of_four(float a, float b, float c, float d, float cmin, float cmax);
    float min_of_four(float a, float b, float c, float d);
    float max_of_four(float a, float b, float c, float d);
    float x_line_with_plane(float y, float z);
    float y_line_with_plane(float x, float z);
    float z_line_with_plane(float x, float y);
    float xy_line_with_cylinder(float y);
    float zx_line_with_cylinder(float x);
    float zy_line_with_cylinder(float y);
    float z_line_with_sphere(float x, float y);
    inline void add_to_corner(int *, int *, int *, int *, int *, int *, int *, int *, int);
    int Point_is_relevant(int i, int j, int k);

public:
    RECT_Plane(int n_elem, int n_nodes, int Type, int *el, int *cl, int *tl, float *x_in,
               float *y_in, float *z_in, float *s_in, unsigned char *p_bs_in, float *i_in, float *u_in, float *v_in,
               float *w_in, const coDistributedObject *p_rgrid_in, int p_x_size, int p_y_size, int p_z_size, int maxPoly,
               float planei_, float planej_, float planek_, float startx_,
               float starty_, float startz_, float myDistance_,
               float radius_, int gennormals_, int option_,
               int genstrips_, char *ib);
    bool createPlane();
};

#if 0
class ALGEXPORT CELL_Plane:public Plane
{
   friend class Isoline;

   public:

      CELL_Plane(int n_elem, int n_nodes, int Type,int *p_el,int *p_cl,int *p_tl,float *p_x_in,float *p_y_in,float *p_z_in,float *p_s_in,unsigned char *p_bs_in,float *p_i_in,float *p_u_in,float *p_v_in,float *p_w_in,coDoCellGrid*    p_cgrid_in,coDoUnstructuredGrid*  p_grid_in,int maxPoly,
         float planei_,float planej_,float planek_,float startx_,
         float starty_,float startz_,float myDistance_,
         float radius_,int gennormals_,int option_,
         int genstrips_, char *ib);
      bool createPlane();
      void add_vertex(int n1,int n2);
      void add_vertex(int n1,int n2, int x,int y,int z,int u,int v,int w);
};
#endif

class ALGEXPORT Isoline
{
private:
    float planei, planej, planek;
    int option;
    int num_iso;
    float offset;
    int num_nodes;
    int num_elem;
    int num_lines;
    int num_isolines;
    int num_vertice;
    int *vertice_list;
    int *vertex;
    int *vl;
    int *ll;
    float *I_Data;
    float *I_Data_p;
    int num_coords;
    int *neighborlist;
    float *coords_x;
    float *coords_y;
    float *coords_z;
    float *coord_x;
    float *coord_y;
    float *coord_z;
    float **iso_coords_x;
    float **iso_coords_y;
    float **iso_coords_z;
    int **iso_ll;
    int **iso_vl;
    int *iso_numvert;
    int *iso_numlines;
    int *iso_numcoords;
    NodeInfo *node;
    Plane *plane;

public:
    Isoline(Plane *pl, int numiso, float offset_, int option_, float planei_, float planej_, float planek_)
        : planei(planei_)
        , planej(planej_)
        , planek(planek_)
        , option(option_)
        , num_iso(numiso)
        , offset(offset_)
        , plane(pl)
    {
        neighborlist = new int[plane->num_triangles];
        vertice_list = new int[plane->num_triangles * 2];
        iso_coords_x = new float *[num_iso];
        iso_coords_y = new float *[num_iso];
        iso_coords_z = new float *[num_iso];
        iso_ll = new int *[num_iso];
        iso_vl = new int *[num_iso];
        iso_numvert = new int[num_iso];
        iso_numlines = new int[num_iso];
        iso_numcoords = new int[num_iso];
        num_isolines = 0;
    }
    ~Isoline()
    {
        delete[] neighborlist;
        delete[] vertice_list;
        delete[] iso_coords_x;
        delete[] iso_coords_y;
        delete[] iso_coords_z;
        delete[] iso_ll;
        delete[] iso_vl;
        delete[] iso_numvert;
        delete[] iso_numlines;
        delete[] iso_numcoords;
    }
    void createcoDistributedObjects(const char *Line_name, coDistributedObject **Line_set,
                                    int currNumber, AttributeContainer &idataattr);
    void add_vertex(int n1, int n2);
    void createIsoline(float Isovalue);
    void sortIsoline();
};
}
#endif
