/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                    (C) 2000 VirCinity  **
 ** Description: axe-murder geometry                                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Lars Frenzel                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  05.01.97  V0.1                                                  **
 **        18.10.2000 V1.0 new API, several data ports, triangle strips    **
 **                                               ( converted to polygons )**
 **			Sven Kufer 					  **
\**************************************************************************/

#define NUM_DATA_IN_PORTS 4 // number of data ports

#include <iostream.h>
#include "MagmaTrace.h"

Magma_Trace::Magma_Trace()
    : coModule("cut something out of an object")
{
    int i;
    char portname[32];

    // ports
    p_geo_in = addInputPort("geo_in", "coDoPoints", "geometry");
    p_data_in = addInputPort("data_in", "coDoFloat", "data");

    p_geo_out = addOutputPort("geo_out", "coDoLines", "geometry");
    p_data_out = addOutputPort("data_out", "coDoFloat", "data");

    p_len = addInt32Param("len", "trace len");
    p_len->setValue(13);
    p_skip = addInt32Param("skip", "skip steps");
    p_skip->setValue(20);
}

int main(int argc, char *argv[])
{
    Magma_Trace *application = new Magma_Trace;
    application->start(argc, argv);
    return 0;
}

////// this is called whenever we have to do something

int Magma_Trace::compute()
{
    coDistributedObject *data_out;
    p_geo_out->setCurrentObject(
        traceLines(p_geo_out->getObjName(), (coDoSet *)p_geo_in->getCurrentObject(),
                   p_data_out->getObjName(), (coDoSet *)p_data_in->getCurrentObject(), &data_out));
    p_data_out->setCurrentObject(data_out);
}

coDistributedObject *Magma_Trace::traceLines(const char *name, coDoSet *all_points, const char *data_name, coDoSet *all_data, coDistributedObject **data_out)
{
    int num_points, num_steps, steps_out = 0;
    int i, j, t;
    coDistributedObject *const *elems = all_points->getAllElements(&num_steps);
    coDistributedObject *const *s_elems = all_data->getAllElements(&num_steps);

    coDistributedObject **output_lines = new coDistributedObject *[num_steps + 1];
    output_lines[num_steps] = NULL;

    coDistributedObject **output_data = new coDistributedObject *[num_steps + 1];
    output_data[num_steps] = NULL;

    float **x = new float *[num_steps];
    float **y = new float *[num_steps];
    float **z = new float *[num_steps];
    float **s = new float *[num_steps];

    coDoPoints *last_set = (coDoPoints *)elems[num_steps - 1];
    int max_points = last_set->getNumPoints();

    cout << "mp: " << max_points << endl;

    for (i = 0; i < num_steps; i++)
    {
        x[i] = new float[max_points];
        y[i] = new float[max_points];
        z[i] = new float[max_points];
        s[i] = new float[max_points];

        for (j = 0; j < max_points; j++)
        {
            x[i][j] = FLT_MAX;
            y[i][j] = FLT_MAX;
            z[i][j] = FLT_MAX;
            s[i][j] = FLT_MAX;
        }
    }

    for (i = 0; i < num_steps; i++)
    {
        coDoPoints *points_t = (coDoPoints *)elems[i];
        int num_t = points_t->getNumPoints();
        float *x_t, *y_t, *z_t;
        points_t->getAddresses(&x_t, &y_t, &z_t);

        coDoFloat *s_data = (coDoFloat *)s_elems[i];
        float *s_t;
        s_data->getAddress(&s_t);

        for (j = 0; j < num_t; j++)
        {
            x[i][j] = x_t[j];
            y[i][j] = y_t[j];
            z[i][j] = z_t[j];
            s[i][j] = s_t[j];
        }
    }

    float l_x[28], l_y[28], l_z[28];
    float l_d[28];

    char objname[256];
    int len = p_len->getValue();
    int first_step = len;
    int skip = p_skip->getValue();
    int *dummy;

    for (i = first_step; i < num_steps; i++)
    {
        Lines *out_lines = new Lines;
        for (j = 0; j < max_points; j++)
        {
            if (x[i - len][j] != FLT_MAX)
            {
                for (t = 0; t <= len; t++)
                {
                    l_x[t] = x[i - t][j];
                    l_y[t] = y[i - t][j];
                    l_z[t] = z[i - t][j];
                    l_d[t] = s[i - t][j];
                }
                out_lines->addLine(len + 1, l_x, l_y, l_z, l_d);
            }
        }

        sprintf(objname, "%s_%d", name, i);
        output_lines[steps_out] = out_lines->getDOLines(objname);

        sprintf(objname, "%s_%d", data_name, i);
        output_data[steps_out++] = out_lines->getDOData(objname);
        delete out_lines;

        i += skip;
    }
    /*
   for( i=0; i<first_step; i++ )
   {
      output_lines[i] = output_lines[first_step];
      output_data[i] = output_data[first_step];
   }*/

    output_lines[steps_out] = NULL;
    coDoSet *ret = new coDoSet(name, output_lines);

    char time_string[256];
    sprintf(time_string, "1 %d", steps_out);
    ret->addAttribute("TIMESTEP", time_string);

    output_data[steps_out] = NULL;
    *data_out = new coDoSet(data_name, output_data);

    return ret;
}

Lines_::Lines_()
{
    ll_ = new int[NUMT];
    cl_ = new int[NUMT * PARTS];

    x_ = new float[NUMT * PARTS];
    y_ = new float[NUMT * PARTS];
    z_ = new float[NUMT * PARTS];

    data_ = new float[NUMT * PARTS];

    num_ll_ = 0;
    num_cl_ = 0;
    num_points_ = 0;
}

Lines_::~Lines_()
{
    delete[] ll_;
    delete[] cl_;
    delete[] x_;
    delete[] y_;
    delete[] z_;
    delete[] data_;
}

void Lines_::addLine(int num, float *x, float *y, float *z, float *val)
{
    int i;
    if (num_ll_ > NUMT - 5)
    {
        return;
    }
    ll_[num_ll_++] = num_cl_;

    for (i = 0; i < num; i++)
    {
        cl_[num_cl_++] = num_points_;
        x_[num_points_] = x[i];
        y_[num_points_] = y[i];
        z_[num_points_] = z[i];
        data_[num_points_] = val[i];

        num_points_++;
    }
}

coDoLines *Lines_::getDOLines(const char *objname)
{
    coDoLines *li;
    li = new coDoLines(objname, num_points_, x_, y_, z_,
                       num_cl_, cl_, num_ll_, ll_);
    return li;
}

coDoFloat *Lines_::getDOData(const char *objname)
{
    return new coDoFloat(objname, num_points_, data_);
}
