/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*************************************************************************
 **                                                        (C)1997 RUS  **
 **                                                                     **
 ** Description: Module to interpolate between unstructured data types  **
 **                                                                     **
 **                                                                     **
 **                                                                     **
 **                                                                     **
 **                                                                     **
 ** Author:                                                             **
 **                                                                     **
 **                           Reiner Beller                             **
 **              Computer Center University of Stuttgart                **
 **                           Allmandring 30                            **
 **                           70550 Stuttgart                           **
 **                                                                     **
 ** Date:  10.10.97  V0.1                                               **
 ************************************************************************/

#include <appl/ApplInterface.h>
#include "Interpolate.h"
#include <util/coviseCompat.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>

Interpolate::Interpolate(int argc, char *argv[])
    : coModule(argc, argv, "Interpolation module")
{
    // Input
    p_dataIn_3 = addInputPort("DataIn3", "Float|Vec3", "Data Input to interpolate from");
    p_dataIn_3->setRequired(0);
    p_dataIn_1 = addInputPort("DataIn0", "Float|Vec3", "Data Input ");
    p_dataIn_2 = addInputPort("DataIn1", "Float|Vec3", "Data Input");
    p_dataIn_2->setRequired(0);
    p_indexIn = addInputPort("DataIn2", "IntArr", "Index Input");
    p_indexIn->setRequired(0);

    // Output
    p_dataOut_1 = addOutputPort("DataOut0", "Float|Vec3", "Output data");
    p_dataOut_2 = addOutputPort("DataOut1", "Float|Vec3", "Output data");
    p_dataOut_2->setDependencyPort(p_dataIn_2);
    p_indexOut = addOutputPort("DataOut2", "IntArr", "Output index");
    p_indexOut->setDependencyPort(p_indexIn);

    // Parameters
    p_motion = addChoiceParam("motion", "Motion characteristic");
    const char *defMotion[] = { "linear", "sinusoidal" };
    p_motion->setValue(2, defMotion, 0);

    p_type = addChoiceParam("type", "Type of animation");
    const char *defType[] = { "linear", "cyclic" };
    p_type->setValue(2, defType, 0);

    p_steps = addIntSliderParam("steps", "Number of interpolation steps");
    p_steps->setValue(1, 50, 10);

    p_abs = addBooleanParam("abs", "Absolute value for scalar data? y|n");
    p_abs->setValue(true);

    p_osci = addBooleanParam("oscillate", "full oscillation? y|n");
    p_osci->setValue(false);
}

static int
NumElemsInSetList(const coDistributedObject *const *setList)
{
    int count_steps = 0;
    while (setList[count_steps])
    {
        ++count_steps;
    }
    return count_steps;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////full_osci////////////////

int Interpolate::compute(const char *)
{
    ////////// Get motion characteristic

    int motion = p_motion->getValue();

    ////////// Get type of animation

    int type = p_type->getValue();

    ////////// Get number of animation steps

    int steps = p_steps->getValue();

    ////////// Get key for scalar data

    int abs = p_abs->getValue();

    ////////// Get key for oscillation mode
    int osci = p_osci->getValue();
    if (osci != 0)
    {
        full_osci = 0.5;
        f_start = -(steps) / 2;
        f_end = (steps) / 2;
    }
    else
    {
        full_osci = 1;
        f_start = 0;
        f_end = steps;
    }
    ///////// Check the step intervall
    if (steps < 1)
    {
        sendError("step number smaller than 1");
        return STOP_PIPELINE;
    }

    ////////// Get data 1

    const coDistributedObject *data1_obj = p_dataIn_1->getCurrentObject();
    if (data1_obj == NULL)
    {
        Covise::sendError("Error receiving data of input port #1");
        return STOP_PIPELINE;
    }

    ////////// Get data 2

    const coDistributedObject *data2_obj = p_dataIn_2->getCurrentObject();

    ////////// Get data 3

    const coDistributedObject *data3_obj = p_dataIn_3->getCurrentObject();

    ////////// Get indexfull_osci

    const coDistributedObject *index_obj = p_indexIn->getCurrentObject();

    ///////// Is the input port for an index field used?
    if (index_obj)
    {

        int i, j;

        const char *indexType;
        indexType = index_obj->getType();

        if (strcmp(indexType, "INTARR") != 0)
        {
            sendError("Incorrect input type for index input");
            delete index_obj;
            return STOP_PIPELINE;
        }
        coDoIntArr *index = (coDoIntArr *)index_obj;
        coDoIntArr **index_out;
        coDistributedObject **newIndex;

        //// create output object name
        const char *o_index_name = p_indexOut->getObjName();

        //// create set of identical index fields
        int num_dim;
        int num_indices;
        int num_types;
        int *size;
        int *in_index;

        num_dim = index->getNumDimensions();
        num_indices = index->getDimension(0);
        size = new int[num_dim];
        switch (num_dim)
        {

        case 1:
            num_types = 1;
            size[0] = num_indices;
            break;

        case 2:
            num_types = index->getDimension(1);
            size[0] = num_indices;
            size[1] = num_types;
            break;

        default:
            sendError("Wrong num_dim");
            return STOP_PIPELINE;
            break;
        }
        index->getAddress(&in_index);
        char int_index_name[512];
        int *out_index;
        switch (type)
        {

        case 0: // linear
            index_out = new coDoIntArr *[steps + 1];
            index_out[steps] = NULL;

            for (i = 0; i < steps; i++)
            {
                sprintf(int_index_name, "%s_%d", o_index_name, i);
                index_out[i] = new coDoIntArr(int_index_name, num_dim, size);
                if (index_out[i] == NULL)
                {
                    sendError("could not create index output object \"%s\"", int_index_name);
                    return STOP_PIPELINE;
                }
                index_out[i]->getAddress(&out_index);
                for (j = 0; j < num_types * num_indices; j++)
                    out_index[j] = in_index[j];
            }
            break;

        case 1: // cyclic
            index_out = new coDoIntArr *[4 * steps + 1];
            index_out[4 * steps] = NULL;
            for (i = 0; i < 4 * steps; i++)
            {
                sprintf(int_index_name, "%s_%d", o_index_name, i);
                index_out[i] = new coDoIntArr(int_index_name, num_dim, size);
                if (index_out[i] == NULL)
                {
                    sendError("could not create index output object \"%s\"", int_index_name);
                    return STOP_PIPELINE;
                }
                index_out[i]->getAddress(&out_index);
                for (j = 0; j < num_types * num_indices; j++)
                    out_index[j] = in_index[j];
            }
            break;
        default:
            sendError("Wrong type");
            return STOP_PIPELINE;
            break;
        }

        // create set
        char attr[256];
        newIndex = (coDistributedObject **)index_out;
        sprintf(attr, "%d %d", 1, NumElemsInSetList(newIndex));
        coDoSet *index_set = new coDoSet(o_index_name, newIndex);
        index_set->addAttribute("TIMESTEP", attr);
        p_indexOut->setCurrentObject(index_set);
    }

    //////////  Distinguish if second input port is used or whether to start from
    //////////  third input port (the left one)
    //
    if ((data2_obj == NULL) && (data3_obj == NULL)) // only one data object
    {

        coDistributedObject **newData;

        //// create output object name
        const char *o_data_name = p_dataOut_1->getObjName();

#ifdef _AIRBUS
        // for the NS3 application only copy the data if a set comes in

        if (data1_obj->isType("SETELE"))
        {
            coDoSet *setIn = (coDoSet *)data1_obj;
            int numElem;
            coDistributedObject *const *inObj = setIn->getAllElements(&numElem);
            coDistributedObject **outObj = new coDistributedObject *[numElem + 1];
            outObj[numElem] = NULL;
            int i;
            for (i = 0; i < numElem; i++)
            {
                outObj[i] = inObj[i];
            }
            coDoSet *setOut = new coDoSet(o_data_name, outObj);
            setOut->copyAllAttributes(setIn);
            p_dataOut_1->setCurrentObject(setOut);
            return CONTINUE_PIPELINE;
        }
#endif
        //// interpolate and create output objects list
        switch (type)
        {

        case 0: // linear
            newData = new coDistributedObject *[steps + 1];
            newData[steps] = NULL;
            break;

        case 1: // cyclic
            newData = new coDistributedObject *[4 * steps + 1];
            newData[4 * steps] = NULL;
            break;
        default:
            sendError("Wrong type");
            return STOP_PIPELINE;
            break;
        }

        int i;
        char int_data_name[512];
        int k = 0;
        for (i = f_start; i < f_end; i++) //gott
        {
            sprintf(int_data_name, "%s_%d", o_data_name, i);
            interpolate(data1_obj, motion, steps, int_data_name, i, &newData[k]);
            if (newData[k] == NULL)
            {
                sendError("Error in interpolation");
                return STOP_PIPELINE;
            }
            k++;
        }

        if (type == 1) // cycle
        {

            setupCycle(newData, steps, abs, o_data_name,
                       &newData[steps], &newData[2 * steps], &newData[3 * steps]);
        }

        // create set
        coDoSet *data_set = new coDoSet(o_data_name, newData);
        char attr[256];
        sprintf(attr, "%d %d", 1, NumElemsInSetList(newData));
        data_set->addAttribute("TIMESTEP", attr);

        return CONTINUE_PIPELINE;
    }

    // interpolate two data objects starting from 0
    else if (((data3_obj == NULL) && (data1_obj != NULL) && (data2_obj != NULL)) || ((data3_obj != NULL) && (data1_obj != NULL) && (data2_obj != NULL)))
    {

        if ((data3_obj != NULL) && (data1_obj != NULL) && (data2_obj != NULL))
        {
            sendWarning("All three data object input ports are connected !");
            sendWarning("Object 3 (left input port) will be ignored");
        }

        coDistributedObject **newData1;
        coDistributedObject **newData2;

        //// create output object name
        const char *o_data1_name = p_dataOut_1->getObjName();
        const char *o_data2_name = p_dataOut_2->getObjName();

        //// interpolate and create output objects list
        switch (type)
        {

        case 0: // linear
            newData1 = new coDistributedObject *[steps + 1];
            newData1[steps] = NULL;
            newData2 = new coDistributedObject *[steps + 1];
            newData2[steps] = NULL;
            break;

        case 1: // cyclic
            newData1 = new coDistributedObject *[4 * steps + 1];
            newData1[4 * steps] = NULL;
            newData2 = new coDistributedObject *[4 * steps + 1];
            newData2[4 * steps] = NULL;
            break;
        default:
            sendError("Wrong type");
            return STOP_PIPELINE;
        }

        int i;
        char int_data1_name[512], int_data2_name[512];
        int k = 0;
        for (i = 0; i < steps; i++)
        {

            sprintf(int_data1_name, "%s_%d", o_data1_name, k);
            sprintf(int_data2_name, "%s_%d", o_data2_name, k);
            interpolate(data1_obj, motion, steps, int_data1_name, i, &newData1[k]);
            interpolate(data2_obj, motion, steps, int_data2_name, i, &newData2[k]);
            k++;
        }

        if (type == 1) // cycle
        {

            setupCycle(newData1, steps, abs, o_data1_name,
                       &newData1[steps], &newData1[2 * steps], &newData1[3 * steps]);

            setupCycle(newData2, steps, abs, o_data2_name,
                       &newData2[steps], &newData2[2 * steps], &newData2[3 * steps]);
        }

        // create sets
        coDoSet *data1_set = new coDoSet(o_data1_name, newData1);
        coDoSet *data2_set = new coDoSet(o_data2_name, newData2);
        char attr[256];
        sprintf(attr, "%d %d", 1, NumElemsInSetList(newData1));
        data1_set->addAttribute("TIMESTEP", attr);

        sprintf(attr, "%d %d", 1, NumElemsInSetList(newData2));
        data2_set->addAttribute("TIMESTEP", attr);

        p_dataOut_1->setCurrentObject(data1_set);
        p_dataOut_2->setCurrentObject(data2_set);

        return CONTINUE_PIPELINE;
    }

    //## interpolate starting from data3_obj ending at data1_obj ##########################
    else if ((data2_obj == NULL) && (data1_obj != NULL) && (data3_obj != NULL))
    {

        coDistributedObject **newData;

        //// create output object name
        const char *o_data_name = p_dataOut_1->getObjName();

#ifdef _AIRBUS
        // for the NS3 application only copy the data if a set comes in
        if (data1_obj->isType("SETELE"))
        {
            sendWarning("ifdef _AIRBUS with data object three connected !");
            sendWarning("Data object three will be ignored !");
            coDoSet *setIn = (coDoSet *)data1_obj;
            int numElem;
            coDistributedObject *const *inObj = setIn->getAllElements(&numElem);
            coDistributedObject **outObj = new coDistributedObject *[numElem + 1];
            outObj[numElem] = NULL;
            int i;
            for (i = 0; i < numElem; i++)
            {
                outObj[i] = inObj[i];
            }
            coDoSet *setOut = new coDoSet(o_data_name, outObj);
            setOut->copyAllAttributes(setIn);
            p_dataOut_1->setCurrentObject(setOut);
            return CONTINUE_PIPELINE;
        }
#endif
        //// interpolate and create output objects list
        switch (type)
        {

        case 0: // linear
            newData = new coDistributedObject *[steps + 1];
            newData[steps] = NULL;
            break;

        case 1: // cyclic
            newData = new coDistributedObject *[4 * steps + 1];
            newData[4 * steps] = NULL;
            break;
        default:
            sendError("Wrong type");
            return STOP_PIPELINE;
            break;
        }

        int i;
        char int_data_name[512];
        int k = 0;
        for (i = f_start; i < f_end; i++) //gott
        {
            sprintf(int_data_name, "%s_%d", o_data_name, i);
            interpolate_two_fields(data1_obj, data3_obj, motion, steps, int_data_name, i, &newData[k]);
            if (newData[k] == NULL)
            {
                sendError("Error in interpolation");
                return STOP_PIPELINE;
            }
            k++;
        }

        if (type == 1) // cycle
        {

            setupCycle(newData, steps, abs, o_data_name,
                       &newData[steps], &newData[2 * steps], &newData[3 * steps]);
        }

        // create set
        coDoSet *data_set = new coDoSet(o_data_name, newData);
        char attr[256];
        sprintf(attr, "%d %d", 1, NumElemsInSetList(newData));
        data_set->addAttribute("TIMESTEP", attr);

        return CONTINUE_PIPELINE;
    }

    return STOP_PIPELINE;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void Interpolate::interpolate(const coDistributedObject *d, const int m,
                              const int s, char *dataName,
                              int no, coDistributedObject **outDataPtr)
{
    int i;

    //////// check data

    int numComp = 0; // how many data components
    if (d)
    {
        const char *data_type = d->getType();
        if (strcmp(data_type, "USTSDT") == 0)
            numComp = 1;
        else if (strcmp(data_type, "USTVDT") == 0)
            numComp = 3;
        else if (strcmp(data_type, "SETELE") == 0)
        {
            numComp = 0;
        }
        else
        {
            sendError("Incorrect data type in input data");
            *outDataPtr = NULL;
            return;
        }
    }
    else
        *outDataPtr = NULL;

    ////////////////////////////////////////////////////////////////
    //////// create data output

    if (d)
    {
        coDoVec3 *v_in_data = NULL;
        coDoFloat *s_in_data = NULL;
        coDoVec3 *v_out_data = NULL;
        coDoFloat *s_out_data = NULL;

        float *in_data[3]; // 1-3 data components
        float *out_data[3]; // for in- and output data
        int num_data;

        if (numComp == 1) ///// scalar data
        {

            s_in_data = (coDoFloat *)d;
            s_in_data->getAddress(&in_data[0]);
            num_data = s_in_data->getNumPoints();

            in_data[1] = NULL;
            in_data[2] = NULL;

            s_out_data = new coDoFloat(dataName, num_data);
            if (s_out_data == NULL)
            {
                sendError("could not create data output object %s", dataName);
                *outDataPtr = NULL;
                return;
            }

            s_out_data->getAddress(&out_data[0]);

            switch (m)
            {

            case 0: // linear
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = in_data[0][i];
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = no * in_data[0][i] / (s - 1);
                }
                break;

            case 1: // sinusoidal
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = in_data[0][i];
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = (float)sin(no * M_PI / (2 * (s - 1))) * in_data[0][i];
                }
                break;
            }

            out_data[1] = NULL;
            out_data[2] = NULL;
        }
        else if (numComp == 3) ///// vector data
        {

            v_in_data = (coDoVec3 *)d;

            v_in_data->getAddresses(&in_data[0], &in_data[1], &in_data[2]);
            num_data = v_in_data->getNumPoints();

            v_out_data = new coDoVec3(dataName, num_data);
            if (v_out_data == NULL)
            {
                sendError("could not create data output object '%s'", dataName);
                *outDataPtr = NULL;
                return;
            }

            v_out_data->getAddresses(&out_data[0], &out_data[1], &out_data[2]);

            switch (m)
            {

            case 0: // linear
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = in_data[0][i];
                        out_data[1][i] = in_data[1][i];
                        out_data[2][i] = in_data[2][i];
                    }
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = no * in_data[0][i] / ((s * full_osci) - 1);
                        out_data[1][i] = no * in_data[1][i] / ((s * full_osci) - 1);
                        out_data[2][i] = no * in_data[2][i] / ((s * full_osci) - 1);
                    }
                }
                break;

            case 1: // sinusoidal
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = in_data[0][i];
                        out_data[1][i] = in_data[1][i];
                        out_data[2][i] = in_data[2][i];
                    }
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = (float)sin(no * M_PI / (2 * ((s * full_osci) - 1))) * in_data[0][i];
                        out_data[1][i] = (float)sin(no * M_PI / (2 * ((s * full_osci) - 1))) * in_data[1][i];
                        out_data[2][i] = (float)sin(no * M_PI / (2 * ((s * full_osci) - 1))) * in_data[2][i];
                    }
                }
                break;
            }
        }
        else if (numComp == 0) /////
        {
            coDoSet *setIn = (coDoSet *)d;
            int numElem;
            const coDistributedObject *const *inObj = setIn->getAllElements(&numElem);
            coDistributedObject **outObj = new coDistributedObject *[numElem + 1];
            outObj[numElem] = NULL;
            int i;
            for (i = 0; i < numElem; i++)
            {
                char name[512];
                sprintf(name, "%s_%d", dataName, i);
                interpolate(inObj[i], m, s, name, no, &outObj[i]);
            }
            coDoSet *setOut = new coDoSet(dataName, outObj);
            setOut->copyAllAttributes(setIn);
            *outDataPtr = setOut;
        }
        else
        {
            sendError("strange type in interpolation");
            *outDataPtr = NULL;
            return;
        }

        ////////////////////////////////////////////////////////////////
        /////// Back to caller with correct field

        if (numComp == 1)
            *outDataPtr = s_out_data;
        else if (numComp == 3)
            *outDataPtr = v_out_data;
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void Interpolate::interpolate_two_fields(
    const coDistributedObject *d,
    const coDistributedObject *f,
    const int m,
    const int s, char *dataName,
    int no, coDistributedObject **outDataPtr)
{
    int i;

    //////// check data

    int numComp = 0; // how many data components

    if (d && f)
    {
        const char *data_type_d = d->getType();
        const char *data_type_f = f->getType();

        // Same data types ?
        if (strcmp(data_type_d, data_type_f) == 0)
        {

            if (strcmp(data_type_d, "USTSDT") == 0)
                numComp = 1;
            else if (strcmp(data_type_d, "USTVDT") == 0)
                numComp = 3;
            else if (strcmp(data_type_d, "SETELE") == 0)
            {
                numComp = 0;
            }
            else
            {
                sendError("Incorrect data type in input data");
                *outDataPtr = NULL;
                return;
            }
        }
        else
        {
            sendError("Different data types to interpolate in between");
            *outDataPtr = NULL;
            return;
        }
    }
    else
        *outDataPtr = NULL;

    ////////////////////////////////////////////////////////////////
    //////// create data output

    if (d && f)
    {
        coDoVec3 *v_in_data_d = NULL;
        coDoFloat *s_in_data_d = NULL;
        coDoVec3 *v_in_data_f = NULL;
        coDoFloat *s_in_data_f = NULL;

        coDoVec3 *v_out_data = NULL;
        coDoFloat *s_out_data = NULL;

        float *in_data_d[3]; // 1-3 data components
        float *in_data_f[3]; // 1-3 data components
        float *out_data[3]; // for in- and output data
        int num_data, num_data_f, num_data_d;

        if (numComp == 1) ///// scalar data
        {

            s_in_data_d = (coDoFloat *)d;
            s_in_data_f = (coDoFloat *)f;

            s_in_data_d->getAddress(&in_data_d[0]);
            s_in_data_f->getAddress(&in_data_f[0]);

            num_data_d = s_in_data_d->getNumPoints();
            num_data_f = s_in_data_f->getNumPoints();

            if (num_data_f == num_data_d)
            {
                num_data = num_data_d;
            }
            else
            {
                sendError("Data objects to interpolate inbetween are not of equal length");
                *outDataPtr = NULL;
                return;
            }

            in_data_d[1] = NULL;
            in_data_d[2] = NULL;
            in_data_f[1] = NULL;
            in_data_f[2] = NULL;

            s_out_data = new coDoFloat(dataName, num_data);
            if (s_out_data == NULL)
            {
                sendError("could not create data output object %s", dataName);
                *outDataPtr = NULL;
                return;
            }

            s_out_data->getAddress(&out_data[0]);

            switch (m)
            {

            case 0: // linear
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = in_data_d[0][i];
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = in_data_f[0][i] + no * (in_data_d[0][i] - in_data_f[0][i]) / (s * full_osci - 1);
                }
                break;

            case 1: // sinusoidal
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = in_data_d[0][i];
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                        out_data[0][i] = in_data_f[0][i] + (float)sin(no * M_PI / (2 * (s * full_osci - 1))) * (in_data_d[0][i] - in_data_f[0][i]);
                }
                break;
            }

            out_data[1] = NULL;
            out_data[2] = NULL;
        }
        else if (numComp == 3) ///// vector data
        {

            v_in_data_d = (coDoVec3 *)d;
            v_in_data_f = (coDoVec3 *)f;

            v_in_data_d->getAddresses(&in_data_d[0], &in_data_d[1], &in_data_d[2]);
            v_in_data_f->getAddresses(&in_data_f[0], &in_data_f[1], &in_data_f[2]);

            num_data_d = v_in_data_d->getNumPoints();
            num_data_f = v_in_data_f->getNumPoints();

            if (num_data_f == num_data_d)
            {
                num_data = num_data_d;
            }
            else
            {
                sendError("Data objects to interpolate inbetween are not of equal length");
                *outDataPtr = NULL;
                return;
            }

            v_out_data = new coDoVec3(dataName, num_data);
            if (v_out_data == NULL)
            {
                sendError("could not create data output object '%s'", dataName);
                *outDataPtr = NULL;
                return;
            }

            v_out_data->getAddresses(&out_data[0], &out_data[1], &out_data[2]);

            switch (m)
            {

            case 0: // linear
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = in_data_d[0][i];
                        out_data[1][i] = in_data_d[1][i];
                        out_data[2][i] = in_data_d[2][i];
                    }
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = in_data_f[0][i] + no * (in_data_d[0][i] - in_data_f[0][i]) / ((s * full_osci) - 1);
                        out_data[1][i] = in_data_f[1][i] + no * (in_data_d[1][i] - in_data_f[1][i]) / ((s * full_osci) - 1);
                        out_data[2][i] = in_data_f[2][i] + no * (in_data_d[2][i] - in_data_f[2][i]) / ((s * full_osci) - 1);
                    }
                }
                break;

            case 1: // sinusoidal
                if (s == 1)
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = in_data_d[0][i];
                        out_data[1][i] = in_data_d[1][i];
                        out_data[2][i] = in_data_d[2][i];
                    }
                }
                else
                {
                    for (i = 0; i < num_data; i++)
                    {
                        out_data[0][i] = in_data_f[0][i] + (float)sin(no * M_PI / (2 * ((s * full_osci) - 1))) * (in_data_d[0][i] - in_data_f[0][i]);
                        out_data[1][i] = in_data_f[1][i] + (float)sin(no * M_PI / (2 * ((s * full_osci) - 1))) * (in_data_d[1][i] - in_data_f[1][i]);
                        out_data[2][i] = in_data_f[2][i] + (float)sin(no * M_PI / (2 * ((s * full_osci) - 1))) * (in_data_d[2][i] - in_data_f[2][i]);
                    }
                }
                break;
            }
        }
        // else if (numComp==0)
        // {
        //    coDoSet *setIn = (coDoSet *) d;
        //    int numElem;
        //    const coDistributedObject *const* inObj  = setIn->getAllElements(&numElem);
        //    coDistributedObject **outObj = new coDistributedObject *[numElem+1];
        //    outObj[numElem] = NULL;
        //    int i;
        //    for (i=0;i<numElem;i++)
        //    {
        //       char name[512];
        //       sprintf(name,"%s_%d",dataName,i);
        //       interpolate(inObj[i],m,s,name,no,&outObj[i]);
        //    }
        //    coDoSet *setOut = new coDoSet(dataName,outObj);
        //    setOut->copyAllAttributes(setIn);
        //    *outDataPtr = setOut;
        // }
        else
        {
            sendError("strange type in interpolation");
            *outDataPtr = NULL;
            return;
        }

        ////////////////////////////////////////////////////////////////
        /////// Back to caller with correct field

        if (numComp == 1)
            *outDataPtr = s_out_data;
        else if (numComp == 3)
            *outDataPtr = v_out_data;
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void Interpolate::setupCycle(coDistributedObject **d, const int s, const int key,
                             const char *namebase,
                             coDistributedObject **outDataPtr1,
                             coDistributedObject **outDataPtr2,
                             coDistributedObject **outDataPtr3)
{
    // prerequisite: s>=1
    int i, j, k;
    int inv;
    int numComp = 0; // how many data components

    char cycle_data_name[3][50];

    coDoVec3 *v_in_data = NULL;
    coDoFloat *s_in_data = NULL;
    coDoVec3 **v_out_data[3];
    coDoFloat **s_out_data[3];

    for (j = 0; j < 3; j++)
    {
        v_out_data[j] = new coDoVec3 *[s];
        s_out_data[j] = new coDoFloat *[s];
    }

    // complete animation cycle
    for (i = 0; i < s; i++)
    {

        //////// check interpolated data
        if (d[i] == NULL)
        {
            sendError("Retrieval of interpolated data not succeeded");
            *outDataPtr1 = NULL;
            *outDataPtr2 = NULL;
            *outDataPtr3 = NULL;
            break;
        }
        const char *data_type = d[i]->getType();
        if (strcmp(data_type, "USTSDT") == 0)
            numComp = 1;
        else if (strcmp(data_type, "USTVDT") == 0)
            numComp = 3;
        else
        {
            sendError("Incorrect data type in interpolated data");
            *outDataPtr1 = NULL;
            *outDataPtr2 = NULL;
            *outDataPtr3 = NULL;
            break;
        }

        ////////////////////////////////////////////////////////////////
        //////// create data output for cycle

        float *in_data[3]; // 1-3 data components
        float *out_data[3]; // for in- and output data
        int num_data;

        if (numComp == 1) ///// scalar data
        {

            s_in_data = (coDoFloat *)d[i];
            s_in_data->getAddress(&in_data[0]);
            num_data = s_in_data->getNumPoints();

            in_data[1] = NULL;
            in_data[2] = NULL;

            sprintf(cycle_data_name[0], "%s_%d", namebase, 2 * s - 1 - i);
            sprintf(cycle_data_name[1], "%s_%d", namebase, 2 * s + i);
            sprintf(cycle_data_name[2], "%s_%d", namebase, 4 * s - 1 - i);

            inv = s - 1 - i;
            s_out_data[0][inv] = new coDoFloat(cycle_data_name[0], num_data);
            s_out_data[1][i] = new coDoFloat(cycle_data_name[1], num_data);
            s_out_data[2][inv] = new coDoFloat(cycle_data_name[2], num_data);

            if (s_out_data[0][inv] == NULL
                || s_out_data[1][i] == NULL
                || s_out_data[2][inv] == NULL)
            {
                *outDataPtr1 = NULL;
                *outDataPtr2 = NULL;
                *outDataPtr3 = NULL;
                break;
            }

            for (j = 0; j < 3; j++)
            {
                if (j == 0 || j == 2)
                {
                    s_out_data[j][inv]->getAddress(&out_data[0]);
                    if (!key)
                    {
                        switch (j)
                        {

                        case 0:
                            for (k = 0; k < num_data; k++)
                                out_data[0][k] = in_data[0][k];
                            break;

                        case 2:
                            for (k = 0; k < num_data; k++)
                                out_data[0][k] = (-1) * in_data[0][k];
                            break;
                        };
                    }
                    else
                        for (k = 0; k < num_data; k++)
                            out_data[0][k] = in_data[0][k];
                }
                else // j==1
                {
                    s_out_data[j][i]->getAddress(&out_data[0]);
                    if (!key)
                        for (k = 0; k < num_data; k++)
                            out_data[0][k] = (-1) * in_data[0][k];
                    else
                        for (k = 0; k < num_data; k++)
                            out_data[0][k] = in_data[0][k];
                }
            }
            out_data[1] = NULL;
            out_data[2] = NULL;
        }
        else ///// vector data
        {

            v_in_data = (coDoVec3 *)d[i];
            v_in_data->getAddresses(&in_data[0], &in_data[1], &in_data[2]);
            num_data = v_in_data->getNumPoints();

            sprintf(cycle_data_name[0], "%s_%d", namebase, 2 * s - 1 - i);
            sprintf(cycle_data_name[1], "%s_%d", namebase, 2 * s + i);
            sprintf(cycle_data_name[2], "%s_%d", namebase, 4 * s - 1 - i);

            inv = s - 1 - i;
            v_out_data[0][inv] = new coDoVec3(cycle_data_name[0], num_data);
            v_out_data[1][i] = new coDoVec3(cycle_data_name[1], num_data);
            v_out_data[2][inv] = new coDoVec3(cycle_data_name[2], num_data);

            if (v_out_data[0][inv] == NULL
                || v_out_data[1][i] == NULL
                || v_out_data[2][inv] == NULL)
            {
                *outDataPtr1 = NULL;
                *outDataPtr2 = NULL;
                *outDataPtr3 = NULL;
                break;
            }
            for (j = 0; j < 3; j++)
            {
                if (j == 0 || j == 2)
                {
                    v_out_data[j][inv]->getAddresses(&out_data[0], &out_data[1], &out_data[2]);
                    switch (j)
                    {

                    case 0:
                        for (k = 0; k < num_data; k++)
                        {
                            out_data[0][k] = in_data[0][k];
                            out_data[1][k] = in_data[1][k];
                            out_data[2][k] = in_data[2][k];
                        }
                        break;

                    case 2:
                        for (k = 0; k < num_data; k++)
                        {
                            out_data[0][k] = (-1) * in_data[0][k];
                            out_data[1][k] = (-1) * in_data[1][k];
                            out_data[2][k] = (-1) * in_data[2][k];
                        }
                        break;
                    }
                }
                else // j==1
                {
                    v_out_data[j][i]->getAddresses(&out_data[0], &out_data[1], &out_data[2]);
                    for (k = 0; k < num_data; k++)
                    {
                        out_data[0][k] = (-1) * in_data[0][k];
                        out_data[1][k] = (-1) * in_data[1][k];
                        out_data[2][k] = (-1) * in_data[2][k];
                    }
                }
            }
        }
    }
    ////////////////////////////////////////////////////////////////
    /////// Back to caller with correct fields

    if (numComp == 1)
    {
        for (i = 0; i < s; i++)
        {
            outDataPtr1[i] = s_out_data[0][i];
            outDataPtr2[i] = s_out_data[1][i];
            outDataPtr3[i] = s_out_data[2][i];
        }
    }
    else
    {
        for (i = 0; i < s; i++)
        {
            outDataPtr1[i] = v_out_data[0][i];
            outDataPtr2[i] = v_out_data[1][i];
            outDataPtr3[i] = v_out_data[2][i];
        }
    }

    for (j = 0; j < 3; j++)
    {
        delete v_out_data[j];
        delete s_out_data[j];
    }
}

MODULE_MAIN(Interpolator, Interpolate)
