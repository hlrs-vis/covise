/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUT_GEOMETRY_H
#define _CUT_GEOMETRY_H

#include <util/coviseCompat.h>
#include <api/coSimpleModule.h>
using namespace covise;
#include <vector>

////// class to help on interpolating new vertices
class Interpol
{
private:
    // pointers to the 'real' data
    // int *num_vert_out;
    std::vector<int> *vl_out_;
    // int *num_coord_out;
    std::vector<float> *x_coord_out_;
    std::vector<float> *y_coord_out_;
    std::vector<float> *z_coord_out_;

    // internal data
    struct record_struct
    {
        int to_vertex;
        int interpol_id;
        record_struct *next;
    } *record_data, **vertexID;

    int cur_heap_pos;
    int heap_blk_size;

    // internal functions
    void add_record(int v1, int v2, int id);

public:
    // reset everything
    // void reset(int, int *, int **, int *, float **, float **, float **, int);
    void reset(int num_coord, std::vector<int> *vl_out, std::vector<float> *x_out, std::vector<float> *y_out, std::vector<float> *z_out,
               int weissDerGeierWasDasIst);

    // check if the connection from v1 to v2 has allready been
    // interpolated
    int interpolated(int v1, int v2);

    // add a new vertex to the output data
    int add_vertex(int, int, float, float, float);

    // clean up
    void finished(void);
};

#include <api/coHideParam.h>

////// our class
class CutGeometry : public coSimpleModule, public Interpol
{
private:
    // some standard vector-computation
    float scalar_product(float *, float *);
    float line_with_plane(float *p1, float *p2, float **r);

    // a function to find vertices that have to be removed
    int to_remove_vertex(float, float, float);

    // utilities
    int find_vertex(int, float **, float **, float **, float, float, float);

    // base vector
    float base_A[3];
    float normal_A[3];
    float distance_A;
    float cylBottom_A[3];
    int algo_option, smooth_option, dir_option;

    /// Input Parameters

    // float base_B[3];
    // float normal_B[3];

    coChoiceParam *p_method;
    coChoiceParam *p_geoMethod;
    coFloatParam *distance_of_plane;
    coFloatVectorParam *normal_of_plane;
    coFloatVectorParam *cylinder_bottom;
    coFloatParam *p_data_min;
    coFloatParam *p_data_max;
    coBooleanParam *p_invert_cut;
    coBooleanParam *p_strictSelection;

    std::vector<coHideParam *> hparams_;
    coHideParam *h_distance_of_plane_;
    coHideParam *h_normal_of_plane_;
    bool preOK_;
    virtual void preHandleObjects(coInputPort **);

    /////////ports

    coInputPort *p_geo_in;
    coOutputPort *p_geo_out;
    coInputPort *p_data_in[NUM_DATA_IN_PORTS];
    coInputPort *p_adjustParams_;
    coOutputPort *p_data_out[NUM_DATA_IN_PORTS];

    /// Methods
    // coDistributedObject **ComputeObject(coDistributedObject **, char **, int);

    virtual int compute(const char *port);
    virtual void postInst();
    virtual void param(const char *paramname, bool in_map_loading);

    void copyAttributesToOutObj(coInputPort **input_ports, coOutputPort **output_ports, int n);

public:
    CutGeometry(int argc, char *argv[]);
};
#endif
