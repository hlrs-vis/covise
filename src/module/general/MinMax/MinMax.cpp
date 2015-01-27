/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE MinMax      application module                    **
 **                                                                        **
 **                                                                        **
 **                       (C) Vircinity 2000, 2001                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner, Sasha Cioringa                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 ** Date:  05.09.98                                                        **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

#include "MinMax.h"
#include <util/coviseCompat.h>
#include <float.h>
#include <do/coDoData.h>

MinMax::MinMax(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Find Minimum and Maximum values")
{

    //parameters

    p_buck = addIntSliderParam("NumBuck", "Number of Buckets");
    p_buck->setValue(5, 100, 20);

    //ports
    p_inPort1 = addInputPort("Data", "Float|Vec3", "scalar data");
    p_outPort1 = addOutputPort("plot2d", "Vec2", "plotdata");
    p_outPort2 = addOutputPort("DataOut1", "Float", "histogram data");
    p_outPort3 = addOutputPort("minmax", "MinMax_Data", "minmax");
    yplot = new float[MAX_BUCKETS];
}

MinMax::~MinMax()
{
    delete[] yplot;
}

int MinMax::compute(const char *)
{
    float step;

    // get parameter
    long minbuck, maxbuck;
    p_buck->getValue(minbuck, maxbuck, buckets);

    if (maxbuck > MAX_BUCKETS)
        maxbuck = MAX_BUCKETS;

    if (buckets <= 1)
        buckets = 2;

    //get input object
    const coDistributedObject *data_obj = p_inPort1->getCurrentObject();

    if (!data_obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort1->getName());
        return FAIL;
    }

    float *scalar;
    int npoint;
    if (const coDoFloat *fdata = dynamic_cast<const coDoFloat *>(data_obj))
    {
        npoint = fdata->getNumPoints();
        fdata->getAddress(&scalar);
    }

    else if (const coDoVec3 *vdata = dynamic_cast<const coDoVec3 *>(data_obj))
    {
        npoint = vdata->getNumPoints();
        float *u, *v, *w;
        vdata->getAddresses(&u, &v, &w);
        scalar = new float[npoint];
        for (int i = 0; i < npoint; i++)
            scalar[i] = sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort1->getName());
        return FAIL;
    }

    for (int i = 0; i < npoint; i++)
    {
        if (scalar[i] < min)
            min = scalar[i];
        if (scalar[i] > max)
            max = scalar[i];
    }

    step = ((max - min) / (buckets - 1));

    if (step > 0)
    {
        for (int i = 0; i < npoint; i++)
            yplot[(int)((scalar[i] - min) / step)]++;
    }
    return SUCCESS;
}

void MinMax::preHandleObjects(coInputPort **InPorts)
{
    (void)InPorts;

    min = FLT_MAX;
    max = -FLT_MAX;
    for (int n = 0; n < MAX_BUCKETS; n++)
        yplot[n] = 0;
}

void MinMax::postHandleObjects(coOutputPort **OutPorts)
{
    sendInfo("min=%g, max=%g (\"%s\")", min, max, p_inPort1->getCurrentObject()->getName());

    // output 0
    coDoFloat *out_data = new coDoFloat(OutPorts[2]->getObjName(), 2);
    if (!out_data->objectOk())
    {
        sendError("Failed to create the object '%s' for the port '%s'", OutPorts[1]->getObjName(), OutPorts[1]->getName());
        return;
    }
    float *minmax;
    out_data->getAddress(&minmax);
    minmax[0] = min;
    minmax[1] = max;

    // output 1
    coDoFloat *histo_data = new coDoFloat(OutPorts[1]->getObjName(), 2 + buckets);
    if (!out_data->objectOk())
    {
        sendError("Failed to create the object '%s' for the port '%s'", OutPorts[1]->getObjName(), OutPorts[1]->getName());
        return;
    }
    float *hdata;
    histo_data->getAddress(&hdata);
    hdata[0] = min;
    hdata[1] = max;
    for (int n = 0; n < buckets; n++)
        hdata[2 + n] = yplot[n];

    // output 2
    coDoVec2 *plot = new coDoVec2(OutPorts[0]->getObjName(), buckets);
    if (!plot->objectOk())
    {
        sendError("Failed to create the object '%s' for the port '%s'", OutPorts[0]->getObjName(), OutPorts[0]->getName());
        return;
    }
    float *xpl, *ypl, step;
    plot->getAddresses(&xpl, &ypl);
    plot->addAttribute("COMMANDS", "AUTOSCALE\nSETS SYMBOL 16\nSETS LINESTYLE 0\n");

    step = ((max - min) / (buckets - 1));

    for (int n = 0; n < buckets; n++)
        xpl[n] = min + n * step;

    for (int n = 0; n < buckets; n++)
        ypl[n] = yplot[n];

    OutPorts[0]->setCurrentObject(plot);
    OutPorts[1]->setCurrentObject(histo_data);
    OutPorts[2]->setCurrentObject(out_data);
}

MODULE_MAIN(Tools, MinMax)
