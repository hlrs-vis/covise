/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                     (C)1999 Vircinity  **
 **   Clip_Interval module for Covise API 2.0                              **
 **                                                                        **
 ** Author:                                                                **
 **                           Ralph Bruckschen                             **
 **                            Vircinity GmbH                              **
 **                             Nobelstr. 15                               **
 **                            70550 Stuttgart                             **
 ** Date:  30.03.00  V0.1                                                  **
\**************************************************************************/

#include "ClipInterval.h"
#include "clip_interval.h"
#include <util/coviseCompat.h>
#include <alg/coColors.h>

#include <float.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Clip_Interval::Clip_Interval(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Clips Geometry according to a Scalar Interval")
{
    int i;
    char buf[32], buf1[32];
    // declare the name of our module

    Min_Slider = addFloatSliderParam("min_Slider", "Min_Value");
    Min_Slider->setValue(-FLT_MAX, FLT_MAX, -FLT_MAX);
    Max_Slider = addFloatSliderParam("max_Slider", "Max_Value");
    Max_Slider->setValue(-FLT_MAX, FLT_MAX, FLT_MAX);
    p_dummy = addBooleanParam("dummy", "check to clip objects without data");
    p_dummy->setValue(1); // 1 -> make dummy geometry on dummy clipping data
    // 0 -> replivate geometry on dummy clipping data
    p_autominmax = addBooleanParam("auto_minmax", "Automatic minmax alignment to incoming data");
    p_autominmax->setValue(false); // false, so previous behavior (when param didnt exist) doesnt change

    // add input and output ports for polygon and data objects
    Geo_In_Port = addInputPort("GridIn0", "Polygons|Points", "Geo input");
    Data_In_Port = addInputPort("DataIn0", "Float", "Scalar input");
    Geo_Out_Port = addOutputPort("GridOut0", "Polygons|Points", "Geo output");
    Data_Out_Port = addOutputPort("DataOut0", "Float", "Scalar output");
    for (i = 0; i < NUM_ADDITIONAL_PORTS; ++i)
    {
        sprintf(buf, "DataOut%d", i + 1);
        sprintf(buf1, "Field output %d", i);
        Data_Map_Out_Port[i] = addOutputPort(buf, "Float|Vec3|Tensor", buf1);
        sprintf(buf, "DataIn%d", i + 1);
        sprintf(buf1, "Field input %d", i);
        Data_Map_In_Port[i] = addInputPort(buf, "Float|Vec3|Tensor", buf1);
        Data_Map_In_Port[i]->setRequired(0);
        Data_Map_Out_Port[i]->setDependencyPort(Data_Map_In_Port[i]);
    }
    p_minmax = addInputPort("DataIn6", "MinMax_Data", "The mininum and the maximum values");
    p_minmax->setRequired(0);
    // and that's all ... no init() or anything else ... that's done in the lib
}

void Clip_Interval::preHandleObjects(coInputPort **InPorts)
{
    (void)InPorts;

    if (Data_In_Port->getCurrentObject() != NULL)
    {
        if (p_autominmax->getValue())
        {
            float min = FLT_MAX;
            float max = -FLT_MAX;
            ScalarContainer scalarField;
            scalarField.Initialise(Data_In_Port->getCurrentObject());
            scalarField.MinMax(min, max);
            if (min <= max)
            {
                float old_val_min = Min_Slider->getValue();
                float old_val_max = Max_Slider->getValue();

                if (max < old_val_min)
                {
                    Min_Slider->setValue(min, max, max);
                }
                else if (min > old_val_min)
                {
                    Min_Slider->setValue(min, max, min);
                }
                else
                {
                    Min_Slider->setValue(min, max, old_val_min);
                }

                if (max < old_val_max)
                {
                    Max_Slider->setValue(min, max, max);
                }
                else if (min > old_val_max)
                {
                    Max_Slider->setValue(min, max, min);
                }
                else
                {
                    Max_Slider->setValue(min, max, old_val_max);
                }
            }
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Compute callback: Called when the module is executed
// ++++
// ++++  NEVER use input/output ports or distributed objects anywhere
// ++++        else than inside this function
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Clip_Interval::compute(const char *)

{
    //cerr << "\n ------- COMPUTE" << endl;

    // here we try to retrieve the data object from the required port

    float min, max, min_value, max_value;

    const coDistributedObject *GeoObj = Geo_In_Port->getCurrentObject();
    if (!GeoObj)
    {
        sendError("Did not receive object at port '%s'", Geo_In_Port->getName());
        return FAIL;
    }
    const coDistributedObject *DataObj = Data_In_Port->getCurrentObject();
    if (!DataObj)
    {
        sendError("Did not receive object at port '%s'", Data_In_Port->getName());
        return FAIL;
    }
    const coDistributedObject *DataInMapObj[NUM_ADDITIONAL_PORTS];
    coDistributedObject *DataOutMapObj[NUM_ADDITIONAL_PORTS];
    int i;
    for (i = 0; i < NUM_ADDITIONAL_PORTS; ++i)
    {
        DataInMapObj[i] = Data_Map_In_Port[i]->getCurrentObject();
        DataOutMapObj[i] = 0;
    }
    const coDistributedObject *MinMaxObj = p_minmax->getCurrentObject();
    Min_Slider->getValue(min, max, min_value);
    Max_Slider->getValue(min, max, max_value);

    if (MinMaxObj)
    {
        if (const coDoFloat *minmaxIn = dynamic_cast<const coDoFloat *>(MinMaxObj))
        {
            int numVal = minmaxIn->getNumPoints();
            if (numVal != 2)
            {
                sendError("Illegal input at minmax port");
                return FAIL;
            }
            float *mmdata;
            minmaxIn->getAddress(&mmdata);
            min = mmdata[0];
            max = mmdata[1];
            if (min_value > max_value)
            {
                float tmp = min_value;
                min_value = max_value;
                max_value = tmp;
            }
            if (min_value < min)
                min_value = min;
            if (max_value > max)
                max_value = max;
            Min_Slider->setValue(min, max, min_value);
            Max_Slider->setValue(min, max, max_value);
        }
        else
        {
            sendError("Received illegal type at port %s", p_minmax->getName());
            return FAIL;
        }
    }
    else
    {
        // only make a change, if auto_minmax==false, since perHandleObjects() didnt adjust the sliders
        if (!p_autominmax->getValue())
        {
            if (min_value > max_value)
            {
                float tmp = min_value;
                min_value = max_value;
                max_value = tmp;
            }
            Min_Slider->setValue(-FLT_MAX, FLT_MAX, min_value);
            Max_Slider->setValue(-FLT_MAX, FLT_MAX, max_value);
        }
    }

    // now we create an object for the output port: get the name and make the Obj
    const char *Geo_outObjName = Geo_Out_Port->getObjName();
    const char *Data_outObjName = Data_Out_Port->getObjName();
    const char *Data_outMapObjName[NUM_ADDITIONAL_PORTS];
    for (i = 0; i < NUM_ADDITIONAL_PORTS; ++i)
    {
        Data_outMapObjName[i] = Data_Map_Out_Port[i]->getObjName();
    }
    coDoPolygons *in_poly = NULL, *out_poly;
    coDoPoints *in_points = NULL, *out_points;
    coDoFloat *in_data, *out_data;
    if (!strcmp(GeoObj->getType(), "POLYGN"))
    {
        in_poly = (coDoPolygons *)GeoObj;
    }
    else if (!strcmp(GeoObj->getType(), "POINTS"))
    {
        in_points = (coDoPoints *)GeoObj;
    }
    else
    {
        sendWarning("Need coDoPolygons or coDoPoints as input");
        return FAIL;
    }

    if (!strcmp(DataObj->getType(), "USTSDT"))
        in_data = (coDoFloat *)DataObj;
    else
    {
        sendWarning("Need coDoFloat as clipping data");
        return FAIL;
    }

    if (!strcmp(GeoObj->getType(), "POINTS"))
    {
        clip_interval *clip = new clip_interval(in_points, in_data,
                                                DataInMapObj, NUM_ADDITIONAL_PORTS,
                                                p_dummy->getValue(), min_value, max_value);
        if (clip->do_clip(&out_points, Geo_outObjName, &out_data, Data_outObjName,
                          DataOutMapObj, Data_outMapObjName))
        {
            sendError("Error while clipping");
            delete clip;
            return FAIL;
        }
        else
        {
            delete clip;
            Geo_Out_Port->setCurrentObject(out_points);
            Data_Out_Port->setCurrentObject(out_data);
            for (i = 0; i < NUM_ADDITIONAL_PORTS; ++i)
            {
                if (DataOutMapObj[i])
                {
                    Data_Map_Out_Port[i]->setCurrentObject(DataOutMapObj[i]);
                }
            }
            return SUCCESS;
        }
        return SUCCESS;
    } // polygons
    else
    {
        clip_interval *clip = new clip_interval(in_poly, in_data,
                                                DataInMapObj, NUM_ADDITIONAL_PORTS,
                                                p_dummy->getValue(), min_value, max_value);
        if (clip->do_clip(&out_poly, Geo_outObjName, &out_data, Data_outObjName,
                          DataOutMapObj, Data_outMapObjName))
        {
            sendError("Error while clipping");
            delete clip;
            return FAIL;
        }
        else
        {
            delete clip;
            Geo_Out_Port->setCurrentObject(out_poly);
            Data_Out_Port->setCurrentObject(out_data);
            for (i = 0; i < NUM_ADDITIONAL_PORTS; ++i)
            {
                if (DataOutMapObj[i])
                {
                    Data_Map_Out_Port[i]->setCurrentObject(DataOutMapObj[i]);
                }
            }
            return SUCCESS;
            // ... do whatever you like with in- or output objects,
            // BUT: do NOT delete them !!!!

            // tell the output port that this is his object
        }
    }
    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Quit callback: as the name tells...
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Clip_Interval::quit()
{
    //cerr << "Ende" << endl;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  postInst() is called once after we contacted Covise, but before
// ++++             getting into the main loop
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Clip_Interval::postInst()
{
    // cerr << "after Contruction" << endl;
}

MODULE_MAIN(Filter, Clip_Interval)
