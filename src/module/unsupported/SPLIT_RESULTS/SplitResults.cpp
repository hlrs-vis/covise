/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SplitResults.h"

int
main(int argc, char *argv[])
{
    SplitResults *application = new SplitResults;

    application->start(argc, argv);

    return 0;
}

SplitResults::SplitResults()
{
    p_indices_ = addInputPort("Indices", "coDoIntArr", "Grid splitting indices");
    p_in_data_ = addInputPort("InData",
                              "coDoFloat|coDoVec3",
                              "Input data");
    p_out_data_ = addOutputPort("OutData",
                                "coDoFloat|coDoVec3",
                                "Output data");
}

SplitResults::~SplitResults()
{
}

int
SplitResults::compute()
{
    coDistributedObject *indices = p_indices_->getCurrentObject();
    if (indices == NULL
        || !indices->objectOk())
    {
        sendError("Index object is null or not OK");
        return FAIL;
    }
    if (!indices->isType("INTARR"))
    {
        sendError("Index object is not an integer array");
        return FAIL;
    }
    coDoIntArr *intArr = dynamic_cast<coDoIntArr *>(indices);
    if (intArr->getNumDimensions() != 1)
    {
        sendError("Only 1D INTARR are considered");
        return FAIL;
    }
    int *int_arr = intArr->getAddress();
    int no_ints = intArr->get_dim(0);
    int maxInt = INT_MIN;
    {
        int indInt;
        for (indInt = 0; indInt < no_ints; ++indInt)
        {
            if (maxInt < int_arr[indInt])
                maxInt = int_arr[indInt];
            if (int_arr[indInt] < 0)
            {
                sendError("An index is negative");
                return FAIL;
            }
        }
    }

    coDistributedObject *data = p_in_data_->getCurrentObject();
    if (data == NULL
        || !data->objectOk())
    {
        sendError("Data object is null or not OK");
        return FAIL;
    }

    int no_data_points;
    if (data->isType("USTSDT"))
    {
        coDoFloat *scalar = dynamic_cast<coDoFloat *>(data);
        no_data_points = scalar->getNumPoints();
        float *u_in;
        scalar->getAddress(&u_in);
        if (no_data_points < maxInt)
        {
            sendError("An index spans a range out of bounds");
        }
        coDoFloat *scalar_out = new coDoFloat(p_out_data_->getObjName(), no_ints);
        float *u;
        scalar_out->getAddress(&u);
        int point;
        for (point = 0; point < no_ints; ++point)
        {
            u[point] = u_in[int_arr[point]];
        }
        p_out_data_->setCurrentObject(scalar_out);
    }
    else if (data->isType("USTVDT"))
    {
        coDoVec3 *vector = dynamic_cast<coDoVec3 *>(data);
        no_data_points = vector->getNumPoints();
        float *u_in, *v_in, *w_in;
        vector->getAddresses(&u_in, &v_in, &w_in);
        if (no_data_points < maxInt)
        {
            sendError("An index spans a range out of bounds");
        }
        coDoVec3 *vector_out = new coDoVec3(p_out_data_->getObjName(), no_ints);
        float *u, *v, *w;
        vector_out->getAddresses(&u, &v, &w);
        int point;
        for (point = 0; point < no_ints; ++point)
        {
            u[point] = u_in[int_arr[point]];
            v[point] = v_in[int_arr[point]];
            w[point] = w_in[int_arr[point]];
        }
        p_out_data_->setCurrentObject(vector_out);
    }
    else
    {
        sendError("Index object is not an integer array");
        return FAIL;
    }
    return SUCCESS;
}

void
SplitResults::copyAttributesToOutObj(coInputPort **input_ports,
                                     coOutputPort **output_ports, int i)
{
    if (i == 0)
        copyAttributes(output_ports[i]->getCurrentObject(), input_ports[1]->getCurrentObject());
}
