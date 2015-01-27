/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READCGNS_H
#define _READCGNS_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                       			  ++
// ++ COVISE CGNS Reader						                          ++
// ++ Author: Vlad                                                        ++
// ++                            				                          ++
// ++               St. Petersburg Polytechnical University               ++
// ++**********************************************************************/
#include "zone.h"
#include <api/coModule.h>
#include <do/coDoSet.h> //for multizone bases and "TIMESTEP" time-dependent data
#include <vector>
//#include <string>
using namespace covise;

struct out_objs
{
    coDistributedObject *gridobj;
    coDistributedObject *floatobj[4];
    coDistributedObject *velobj;
};

class base
{
    int index_file;
    int ibase;

    //base info
    int cell_dim, phys_dim;
    char basename[100];

    params p; //UI params

    int error;
    enum
    {
        FAIL = -1,
        SUCCESS = 0 //return values
    };

    vector<zone> zones;

public:
    base(int i_file, int i_base, params p);
    int read();

    bool is_single_zone();

    coDistributedObject *create_do(const char *name, int type, int scal_no = 0);
    coDoSet *create_do_set(const char *name, int type, int scal_no = 0);
};

class ReadCGNS : public coModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    coFileBrowserParam *param_file;

    //zone reading params
    coBooleanParam *param_load_2d;
    coBooleanParam *param_use_string;
    coStringParam *param_sections_string;

    //Select solution fields to display
    coChoiceParam *param_vx;
    coChoiceParam *param_vy;
    coChoiceParam *param_vz;

    coChoiceParam *param_f[4];

    //file timesteps
    coBooleanParam *param_use_file_timesteps;
    coIntScalarParam *param_first_file_idx;
    coIntScalarParam *param_last_file_idx;

    // Base loading params
    coBooleanParam *param_read_single_zone;
    coIntScalarParam *param_zone;

    coOutputPort *out_mesh;
    //coOutputPort	*points_out;

    coOutputPort *out_float[4];

    coOutputPort *out_velocity;

    void params_out();
    bool indexed_file_name(char *s, int n);

    // ReadCGNS functions
    int read_params(params &p);

    // COVISE objects creation
    int create_base_objs(base &b, out_objs &objs, int number);

    int set_output_objs(out_objs &objs);

    //coModule functions
    virtual void param(const char *paramName, bool inMapLoading);
    virtual void postInst();

public:
    ReadCGNS(int argc, char *argv[]);
};

#endif
