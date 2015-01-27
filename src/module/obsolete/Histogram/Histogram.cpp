/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE Histogram application module                   **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 **       13.09.95  Added Sets					          **
 **       13.02.01 Sven Kufer (C) 2000 VirCinity                           **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "Histogram.h"
#include <util/coviseCompat.h>
//
// static stub callback functions calling the real class
// member functions
//

int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();

    return 0;
}

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//
//
//..........................................................................
//
//

//
//
//..........................................................................
//
//

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
void Application::compute(void *)
{
    coDistributedObject *data_obj, *tmp_obj;
    coDistributedObject *const *data_objs = NULL;
    long minbuck, maxbuck, buckets;
    int num_set_elem = 0;
    float min = 0.0, max = 0.0; //will be overwritten later
    float *xpl = NULL, *ypl = NULL, step = 0.0, *scalar = NULL;
    char *Data, *PLOT_Name, *dataType, *MinMax_Name;
    int i, n, npoint = 0;
    coDoFloat *u_data;
    coDoVec2 *plot;
    coDoSet *data_set;

    Data = Covise::get_object_name("Data");
    if (Data == 0L)
    {
        Covise::sendError("ERROR: Object name not correct for 'Data'");
        return;
    }

    PLOT_Name = Covise::get_object_name("2dplot");
    if (PLOT_Name == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 2dplot");
        return;
    }
    MinMax_Name = Covise::get_object_name("minmax");
    if (MinMax_Name == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for minmax");
        return;
    }

    //	get parameter
    Covise::get_slider_param("NumBuck", &minbuck, &maxbuck, &buckets);
    if (buckets <= 0)
    {
        Covise::sendError("Parameter NumBuck must be positive");
        return;
    }
    buckets++; // plot ignores the last one,
    // which is only used to fix the right limit
    // of the last bar
    //	retrieve object from shared memeory
    tmp_obj = new coDistributedObject(Data);
    data_obj = tmp_obj->createUnknown();
    if (data_obj)
    {
        dataType = data_obj->getType();
        if (strcmp(dataType, "USTSDT") == 0)
        {
            u_data = (coDoFloat *)data_obj;
            npoint = u_data->getNumPoints();
            u_data->getAddress(&scalar);
            if (npoint == 0)
                Covise::sendWarning("WARNING: Data object 'Data' is empty");
            min = scalar[0];
            max = scalar[0];
        }
        else if (strcmp(dataType, "SETELE") == 0)
        {
            data_set = (coDoSet *)data_obj;
            data_objs = data_set->getAllElements(&num_set_elem);
            if (num_set_elem == 0)
            {
                Covise::sendError("ERROR: Set is empty");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'Data' has wrong data type");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'Data' can't be accessed in shared memory");
        return;
    }

    if (num_set_elem > 0)
    {
        for (n = 0; n < num_set_elem; n++)
        {

            dataType = data_objs[n]->getType();
            if (strcmp(dataType, "USTSDT") == 0)
            {
                u_data = (coDoFloat *)data_objs[n];
                npoint = u_data->getNumPoints();
                u_data->getAddress(&scalar);
                if (npoint == 0)
                    Covise::sendWarning("WARNING: Set_elem 'Data' is empty");
            }
            if (n == 0)
            {
                min = scalar[0];
                max = scalar[0];
            }
            for (i = 0; i < npoint; i++)
            {
                if (scalar[i] < min)
                    min = scalar[i];
                if (scalar[i] > max)
                    max = scalar[i];
            }
        }
    }
    else
    {
        for (i = 0; i < npoint; i++)
        {
            if (scalar[i] < min)
                min = scalar[i];
            if (scalar[i] > max)
                max = scalar[i];
        }
    }
    plot = new coDoVec2(PLOT_Name, buckets);
    plot->getAddresses(&xpl, &ypl);
    plot->addAttribute("COMMANDS", "AUTOSCALE\nSETS SYMBOL 16\nSETS LINESTYLE 0\n");
    step = ((max - min) / (buckets - 1));

    for (n = 0; n < buckets; n++)
    {
        xpl[n] = min + /*(step/2.0)*/ +(n * step);
        ypl[n] = 0;
    }
    if (num_set_elem > 0)
    {
        for (n = 0; n < num_set_elem; n++)
        {

            dataType = data_objs[n]->getType();
            if (strcmp(dataType, "USTSDT") == 0)
            {
                u_data = (coDoFloat *)data_objs[n];
                npoint = u_data->getNumPoints();
                u_data->getAddress(&scalar);
                if (npoint == 0)
                    Covise::sendWarning("WARNING: Set_elem 'Data' is empty");
            }
            if (step == 0.0)
            {
                Covise::sendWarning("All values are equal.");
                xpl[0] = min - npoint * 0.15;
                xpl[1] = min + npoint * 0.15;
                xpl[2] = min - npoint * 0.5;
                xpl[3] = min + npoint * 0.5;
                ypl[0] = ypl[1] = npoint;
            }
            else
            {
                for (i = 0; i < npoint; i++)
                {
                    // ypl[(int)((scalar[i]-min)/step)]++;
                    ypl[int(floor((scalar[i] - min) / step))]++;
                }
                // The maximum value has fallen in the last bucket,
                // which is only used to fix the right limit of the
                // last visible bucket (the previous one)
                // we transfer it to the previous one
                if (buckets >= 2)
                {
                    ypl[buckets - 2] += ypl[buckets - 1];
                    ypl[buckets - 1] = 0;
                }
            }
        }
    }
    else
    {
        if (step == 0.0)
        {
            Covise::sendWarning("All values are equal.");
            xpl[0] = min - npoint * 0.005;
            xpl[1] = min + npoint * 0.005;
            xpl[2] = min - npoint * 0.5;
            xpl[3] = min + npoint * 0.5;
            ypl[0] = npoint;
        }
        else
        {
            for (i = 0; i < npoint; i++)
            {
                // ypl[(int)((scalar[i]-min)/step)]++;
                ypl[int(floor((scalar[i] - min) / step))]++;
            }
            // The maximum value has fallen in the last bucket,
            // which is only used to fix the right limit of the
            // last visible bucket (the previous one)
            // we transfer it to the previous one
            if (buckets >= 2)
            {
                ypl[buckets - 2] += ypl[buckets - 1];
                ypl[buckets - 1] = 0;
            }
        }
    }
    // Covise::set_scalar_param("Min",min);
    // Covise::set_scalar_param("Max",max);
    u_data = new coDoFloat(MinMax_Name, 2);
    u_data->getAddress(&scalar);
    scalar[0] = min;
    scalar[1] = max;
    delete u_data;
    delete tmp_obj;
    delete data_obj;
}
