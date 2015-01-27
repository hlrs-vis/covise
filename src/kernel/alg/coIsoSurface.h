/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLANE_KRAMM_H
#define _PLANE_KRAMM_H

#include <util/coTypes.h>
#include <cstdlib>
#include <alg/IsoSurfaceGPMUtil.h>

namespace covise
{

class coOutputPort;

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

class ALGEXPORT IsoPlane
{
    friend class STR_IsoPlane;
    friend class UNI_IsoPlane;
    friend class RECT_IsoPlane;
    friend class POLYHEDRON_IsoPlane;

protected:
    const int *el, *cl, *tl;
    const float *x_in;
    const float *y_in;
    const float *z_in;
    const float *s_in; // mapped scalar data
    const float *i_in; // isodata
    const float *u_in; // mapped vector data
    const float *v_in; // mapped vector data
    const float *w_in; // mapped vector data

private:
    int num_nodes;
    int num_elem;
    int num_triangles;
    int num_vertices;
    int num_strips;
    int *vertice_list;
    int *vertex;
    int num_coords;
    int max_coords;
    int *ts_vertice_list;
    int *ts_line_list;
    int *neighbors;
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
    float *V_Data_U_p;
    float *V_Data_V_p;
    float *V_Data_W_p;
    float *S_Data_p;
    float *Normals_U;
    float *Normals_V;
    float *Normals_W;
    NodeInfo *node_table;
    int Datatype;
    float _isovalue;
    bool _isConnected;
    char *iblank;

    // Maximal number of triangles attached to one Vertex.
    // configure at IsoSurface.MAX_TRI_PER_VERT
    // starting value for
    static int maxTriPerVertex;

    // This variable represents the current state. Set when
    // a neighbour list is built. Might be increased if the
    // list was not built successfully with the given default
    int triPerVertex;

protected:
    bool add_vertex(int n1, int n2);
    void add_vertex(int n1, int n2, int x, int y, int z, int u, int v, int w);

public:
    bool polyhedral_cells_found;

    IsoPlane();
    IsoPlane(int n_elem, int n_nodes, int Type, float cutVertexRatio,
             const int *el, const int *cl, const int *tl,
             const float *x_in, const float *y_in, const float *z_in,
             const float *s_in, const float *i_in,
             const float *u_in, const float *v_in, const float *w_in, float isovalue,
             bool isConnected, char *ib);
    IsoPlane(int n_elem, int n_nodes, int Type, /*float cutVertexRatio,*/
             const int *el, const int *cl, const int *tl,
             const float *x_in, const float *y_in, const float *z_in,
             const float *s_in, const float *i_in,
             const float *u_in, const float *v_in, const float *w_in, float isovalue,
             bool isConnected, char *ib);
    virtual ~IsoPlane();
    void createNormals(int genstrips);
    void createStrips(int gennormals);
    void createcoDistributedObjects(coOutputPort *, coOutputPort *, coOutputPort *,
                                    int gennormals, int genstrips,
                                    const char *colorn);
    bool createIsoPlane();
    void createNeighbourList();

    // access to output fields
    int getNumCoords()
    {
        return num_coords;
    };
    int getNumVertices()
    {
        return (int)(vertex - vertice_list);
    };
    int getNumTriangles()
    {
        return num_triangles;
    }
    float *getXout()
    {
        return coords_x;
    };
    float *getYout()
    {
        return coords_y;
    };
    float *getZout()
    {
        return coords_z;
    };
    int *getVerticeList()
    {
        return vertice_list;
    };
};

class ALGEXPORT STR_IsoPlane : public IsoPlane
{

public:
    STR_IsoPlane(int n_elem, int n_nodes, int Type,
                 int xsiz, int ysiz, int zsiz,
                 const float *x_in, const float *y_in, const float *z_in,
                 const float *s_in, const float *i_in,
                 const float *u_in, const float *v_in, const float *w_in, float isovalue,
                 bool isConnected, char *ib)
        : IsoPlane(n_elem, n_nodes, Type, -1, NULL, NULL, NULL, x_in, y_in, z_in,
                   s_in, i_in, u_in, v_in, w_in, isovalue, isConnected, ib)
        , x_size(xsiz)
        , y_size(ysiz)
        , z_size(zsiz)
    {
    }
    bool createIsoPlane();

private:
    int x_size;
    int y_size;
    int z_size;
};

class ALGEXPORT UNI_IsoPlane : public IsoPlane
{

public:
    UNI_IsoPlane(int n_elem, int n_nodes, int Type,
                 float x_min, float x_max, float y_min,
                 float y_max, float z_min, float z_max,
                 int xsiz, int ysiz, int zsiz,
                 const float *sin, const float *iin,
                 const float *uin, const float *vin, const float *win, float isovalue,
                 bool isConnected, char *ib);
    virtual ~UNI_IsoPlane();
    void createIsoPlane();

private:
    int x_size;
    int y_size;
    int z_size;
};

class ALGEXPORT RECT_IsoPlane : public IsoPlane
{

public:
    RECT_IsoPlane(int n_elem, int n_nodes, int Type,
                  int xsiz, int ysiz, int zsiz,
                  const float *xin, const float *yin, const float *zin,
                  const float *sin, const float *iin,
                  const float *uin, const float *vin, const float *win, float isovalue,
                  bool isConnected, char *ib);
    void createIsoPlane();

private:
    int x_size;
    int y_size;
    int z_size;
};

class ALGEXPORT POLYHEDRON_IsoPlane : public IsoPlane
{
private:
    int num_conn;

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
    float *sdata_out;
    float *udata_out;
    float *vdata_out;
    float *wdata_out;

public:
    POLYHEDRON_IsoPlane(int n_elem, int n_conn, int n_nodes, int Type,
                        const int *el, const int *cl, const int *tl,
                        const float *x_in, const float *y_in, const float *z_in,
                        const float *s_in, const float *i_in,
                        const float *u_in, const float *v_in, const float *w_in, float isovalue,
                        bool isConnected, char *ib);

    ~POLYHEDRON_IsoPlane();

    void create_isocontour(int num_elem_in, int *elem_in, int num_conn_in, int *conn_in,
                           int num_coord_in, float *x_coord_in, float *y_coord_in, float *z_coord_in,
                           int *polygon_list, int *index_list, float *isodata_in, float *sdata_in, float *udata_in, float *vdata_in, float *wdata_in,
                           float isovalue, ISOSURFACE_EDGE_INTERSECTION_VECTOR &intsec_vector, CONTOUR &contour, TESSELATION &triangulation);

    bool createIsoPlane();

    void createcoDistributedObjects(coOutputPort *p_GridOut, coOutputPort *p_DataOut);
};
}
#endif
