/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SDOMAINSURFACE_H
#define _SDOMAINSURFACE_H

#include <api/coSimpleModule.h>
using namespace covise;
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <do/coDoLines.h>
#include <do/coDoPolygons.h>

class SDomainsurface : public coSimpleModule
{

private:
    float *x_coords_in, *y_coords_in, *z_coords_in;

    // compute callback
    virtual int compute(const char *port);
    void copyAttributesToOutObj(coInputPort **input_ports,
                                coOutputPort **output_ports, int n);
    double get_angle(int v1, int v2, int v3, int v21, int v22, int v23);

    coOutputPort *outPort_polySet, *outPort_dataP, *outPort_dataL, *outPort_bound;
    coInputPort *p_grid1, *p_data1;
    coFloatParam *param_angle;
    coFloatVectorParam *param_vertex;
    coFloatParam *param_scalar;
    coBooleanParam *param_double;
    //       coChoiceParam *param_optimize;

    //       enum OptimizeParam
    //       {
    //          Speed = 0x00,
    //          Memory = 0x01
    //       };

    //coStringParam *p_string;

    //
    // stuff for unstructured grids
    //

    enum
    {
        DATA_NONE = 0,
        DATA_S,
        DATA_V,
        DATA_S_E,
        DATA_V_E
    };

    //       int MEMORY_OPTIMIZED;
    float x_center, y_center, z_center;

    //  adjacency vars
    float angle, n2x, n2y, n2z, scalar;

    int numelem, numelem_o, numconn, numcoord;
    int doDoubleCheck;
    int *el, *cl, *tl;

    float *x_in, *y_in, *z_in;
    float *u_in, *v_in, *w_in;
    int num_vert, DataType;
    int num_bar; // number of bar elements
    float *x_out, *y_out, *z_out;
    float *u_out, *v_out, *w_out;
    int num_conn, *conn_list, *conn_tag, num_elem, *elem_list, i_i;
    int *elemMap; // attach elements in line list to
    // grid element list
    int lnum_vert;
    float *lx_out, *ly_out, *lz_out, tresh;
    int lnum_conn, *lconn_list, lnum_elem, *lelem_list, i_l;
    float *lu_out, *lv_out, *lw_out;

    /////////////////////////////////////////////////////////
    vector<int> temp_conn_list;
    vector<int> temp_elem_list;

    vector<float> temp_x_out;
    vector<float> temp_y_out;
    vector<float> temp_z_out;
    vector<float> temp_u_out;
    vector<float> temp_v_out;
    vector<float> temp_w_out;
    vector<float> temp_lu_out;
    vector<float> temp_lv_out;
    vector<float> temp_lw_out;
    //////////////////////////////////////////////////////////

    coDoUnstructuredGrid *tmp_grid;
    coDoLines *Lines;
    coDoPolygons *Polygons;
    coDoFloat *SOut;
    coDoVec3 *VOut;

    coDoFloat *SlinesOut;
    coDoVec3 *VlinesOut;

    coDoFloat *USIn;
    coDoVec3 *UVIn;

    void doModule(const coDistributedObject *meshIn,
                  const coDistributedObject *dataIn,
                  const char *meshOutName,
                  const char *dataOutName,
                  const char *linesOutName,
                  const char *ldataOutName,
                  coDistributedObject **meshOut,
                  coDistributedObject **dataOut,
                  coDistributedObject **linesOut,
                  coDistributedObject **ldataOut);

    void surface();
    void lines();
    int test(int, int, int);
    int add_vertex(int v);
    int ladd_vertex(int v);
    int norm_check(int v1, int v2, int v3, int v4 = -1);

public:
    SDomainsurface(int argc, char *argv[]);
};
#endif
