/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FIX_PART_H
#define __FIX_PART_H

#include <api/coModule.h>
using namespace covise;

class FixPart : public coModule
{
private:
    coOutputPort *p_grid_out;
    coInputPort *p_grid_in;
    coIntScalarParam *part_param;

    float **displace_vector;
    int timestep;
    int PartID;

    coDistributedObject *handle_objects(const coDistributedObject *obj_in);
    int find_reference(const coDistributedObject *obj_in);
    void copyAttributes(coDistributedObject *obj_out, const coDistributedObject *obj_in);

public:
    int compute(const char *port);
    FixPart(int argc, char *argv[]);
};
#endif
