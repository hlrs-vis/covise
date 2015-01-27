/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2000 RUS   **
 **                                                                          **
 ** Description: Fix one part of an Dyna3D animation                         **
 **                                                                          **
 ** Name:        FixPart                                                     **
 ** Category:    Tools                                                       **
 **                                                                          **
 ** Author: Sven Kufer		                                            **
 **                                                                          **
 ** History:  								    **
 ** January-01     					       		    **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "FixPart.h"
#include <util/coviseCompat.h>
#include <float.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>

FixPart::FixPart(int argc, char *argv[])
    : coModule(argc, argv, "Fix one part")
{
    p_grid_out = addOutputPort("GridOut0", "UnstructuredGrid", "set of modified grids");
    p_grid_in = addInputPort("GridIn0", "UnstructuredGrid", "set of grids to modify");
    part_param = addInt32Param("PartID", "Part ID to be fixed");
    part_param->setValue(0);
}

coDistributedObject *FixPart::handle_objects(const coDistributedObject *obj_in)
{
    coDoSet *out_set;
    const coDistributedObject *const *objs;
    coDistributedObject **objs_out;
    int numsets, i, num_coords, num_elem, num_conn, *elem, *conn, *type;
    coDoUnstructuredGrid *grid_out;
    float *x, *y, *z;
    char obj_name[100];

    if (const coDoSet *in_set = dynamic_cast<const coDoSet *>(obj_in))
    {
        objs = in_set->getAllElements(&numsets);

        objs_out = new coDistributedObject *[numsets + 1];

        for (i = 0; i < numsets; i++)
        {
            objs_out[i] = handle_objects(objs[i]);
            copyAttributes(objs_out[i], objs[i]);
            if (in_set->getAttribute("TIMESTEP"))
                timestep++;
        }

        objs_out[numsets] = NULL;
        if (in_set->getAttribute("TIMESTEP"))
        {
            out_set = new coDoSet(p_grid_out->getObjName(), objs_out);
            delete[] objs_out;
            for (i = 0; i < numsets; i++)
            {
                delete[] displace_vector[i];
            }
            delete[] displace_vector;
            return out_set;
        }
        else
        {
            sprintf(obj_name, "%s_fix", in_set->getName());
            out_set = new coDoSet(obj_name, objs_out);
            delete[] objs_out;
            return out_set;
        }
    }

    if (const coDoUnstructuredGrid *grid_in = dynamic_cast<const coDoUnstructuredGrid *>(obj_in))
    {
        grid_in->getGridSize(&num_elem, &num_conn, &num_coords);
        grid_in->getAddresses(&elem, &conn, &x, &y, &z);
        grid_in->getTypeList(&type);
        float *x_out = new float[num_coords];
        float *y_out = new float[num_coords];
        float *z_out = new float[num_coords];
        for (i = 0; i < num_coords; i++)
        {
            x_out[i] = x[i] - displace_vector[timestep][0];
            y_out[i] = y[i] - displace_vector[timestep][1];
            z_out[i] = z[i] - displace_vector[timestep][2];
        }
        sprintf(obj_name, "%s_fix", grid_in->getName());
        grid_out = new coDoUnstructuredGrid(obj_name, num_elem, num_conn, num_coords,
                                            elem, conn, x_out, y_out, z_out, type);
        delete[] x_out;
        delete[] y_out;
        delete[] z_out;
        return grid_out;
    }
    return NULL;
}

int FixPart::find_reference(const coDistributedObject *obj_in)
{
    const coDoSet *in_set2;
    const coDistributedObject *const *objs;
    const coDistributedObject *const *objs2;
    int numsets, numsets2, i, j, *fdummy;
    int dummy;
    float *x, *y, *z;
    bool found;
    const char *part_number;
    int part_nr;

    if (const coDoSet *in_set = dynamic_cast<const coDoSet *>(obj_in))
    {
        objs = in_set->getAllElements(&numsets);

        if (in_set->getAttribute("TIMESTEP"))
        {
            displace_vector = new float *[numsets];
            if (displace_vector == NULL)
                return 0;
            for (i = 0; i < numsets; i++)
            {
                coDoUnstructuredGrid *grid_in = NULL;
                in_set2 = dynamic_cast<const coDoSet *>(objs[i]);
                if (in_set2)
                {
                    displace_vector[i] = new float[3];
                    objs2 = in_set2->getAllElements(&numsets2);
                    found = false;
                    for (j = 0; j < numsets2 && !found; j++)
                    {
                        grid_in = (coDoUnstructuredGrid *)objs2[j];
                        part_number = grid_in->getAttribute("PART");
                        if (part_number == NULL)
                            return 0;
                        if (sscanf(part_number, "%d", &part_nr) != 1)
                        {
                            fprintf(stderr, "FixPart: sscanf failed\n");
                        }
                        if (part_nr == PartID)
                            found = true;
                    }
                    if (!found)
                    {
                        sendError("Your chosen partID is not in this data set!");
                        return 0;
                    }
                    //grid_in = (coDoUnstructuredGrid *) objs2[PartID];
                    grid_in->getGridSize(&dummy, &dummy, &dummy);
                    grid_in->getAddresses(&fdummy, &fdummy, &x, &y, &z);
                    displace_vector[i][0] = x[0];
                    displace_vector[i][1] = y[0];
                    displace_vector[i][2] = z[0];
                }
                else
                {
                    sendError("Expect set of parts per timestep");
                    return 0;
                }
            }
            return 1;
        }
        else
        {
            sendError("Expect set of timesteps");
            return 0;
        }
    }
    return 0;
}

int FixPart::compute(const char *)
{

    PartID = part_param->getValue();

    const coDistributedObject *obj_in = p_grid_in->getCurrentObject();
    coDistributedObject *obj_out;

    if (!dynamic_cast<const coDoSet *>(obj_in))
    {
        sendError("Expect set of grids");
        return STOP_PIPELINE;
    }

    if (find_reference(obj_in))
    {
        timestep = 0;
        obj_out = handle_objects(obj_in);
        copyAttributes(obj_out, obj_in);
        p_grid_out->setCurrentObject(obj_out);
        return CONTINUE_PIPELINE;
    }
    else
        return STOP_PIPELINE;
}

void FixPart::copyAttributes(coDistributedObject *obj_out, const coDistributedObject *obj_in)
{
    int num_attrib;
    const char **attr_names, **attr_values;

    num_attrib = obj_in->getAllAttributes(&attr_names, &attr_values);
    obj_out->addAttributes(num_attrib, attr_names, attr_values);
}

MODULE_MAIN(Tools, FixPart)
