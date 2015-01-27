/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ASSEMBLE_USG_H
#define __ASSEMBLE_USG_H

#include <util/coviseCompat.h>
#include <api/coModule.h>
#include <do/coDoSet.h>

using namespace covise;

class AssembleUsg : public coModule
{
private:
    enum GRID_TYPE
    {
        GT_NONE,
        GT_UNSGRD,
        GT_POLYGN
    };
    GRID_TYPE gridType;
    const coDistributedObject *lastGridElement;
    enum DATA_TYPE
    {
        DT_NONE,
        DT_FLOAT,
        DT_VEC3
    };
    DATA_TYPE dataType;
    const coDistributedObject *lastDataElement;

    std::vector<float *> x_coord_in;
    std::vector<float *> y_coord_in;
    std::vector<float *> z_coord_in;
    std::vector<int *> elem_in;
    std::vector<int *> conn_in;
    std::vector<int *> tl_in;
    std::vector<int> num_elem_in;
    std::vector<int> num_conn_in;
    std::vector<int> num_coord_in;

    std::vector<float *> x_data_in;
    std::vector<float *> y_data_in;
    std::vector<float *> z_data_in;
    std::vector<int> num_points_in;

    coInputPort *p_gridIn, *p_dataIn;
    coOutputPort *p_gridOut, *p_dataOut;
    coBooleanParam *p_removeUnnescessaryTimesteps;

    int compute(const char *port);

    coDistributedObject *unpack_grid(const coDistributedObject *obj_in, const char *obj_name);
    coDistributedObject *unpack_data(const coDistributedObject *obj_in, const char *obj_name);
    void copyAttributes(coDistributedObject *tgt, const coDistributedObject *src);

    bool checkGrid(const coDistributedObject *obj_in);
    bool checkData(const coDistributedObject *obj_in);
    void findGrid(const coDoSet *set_in);
    void findData(const coDoSet *set_in);

public:
    AssembleUsg(int argc, char *argv[]);
};
#endif
