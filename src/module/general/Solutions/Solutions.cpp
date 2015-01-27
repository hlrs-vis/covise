/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                     (C) 2001 VirCinity **
 **                                                                        **
 ** Description:   COVISE ReadPlot3D Solutions module                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **                                      **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 ** Date: 26.09.97                                                         **
 **                                                                        **
 ** changed to newAPI + further solutions:                                 **
 ** 30.10.2001        Sven Kufer                                           **
 **                   VirCinity IT-Consulting GmbH                         **
 **                   Nobelstrasse 15                                      **
 **                   70569 Stuttgart                                      **
 **                                                                        **
\**************************************************************************/

#include "Solutions.h"
#include <do/coDoData.h>

Solutions::Solutions(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Calculate Plot3D-Solution data")
{
    p_density = addInputPort("density", "Float", "density");
    p_momentum = addInputPort("MomentumDirection", "Vec3", "x,-,y-, z-momentum");
    p_momentum->setRequired(0);
    p_energy = addInputPort("energy", "Float", "energy per volume unit");
    p_rhou = addInputPort("rhou", "Float", "rhou");
    p_rhou->setRequired(0);
    p_rhov = addInputPort("rhov", "Float", "rhov");
    p_rhov->setRequired(0);
    p_rhow = addInputPort("rhow", "Float", "rhow");
    p_rhow->setRequired(0);
    p_solution = addOutputPort("solution", "Float|Vec3",
                               "calculated data");

    const char *types[] = { "VELOCITY", "P_STATIC", "P_TOTAL", "M", "CP", "T_STATIC", "T_TOTAL", "MACH" };
    p_calctype = addChoiceParam("calctype", "what should be calulcated");
    p_calctype->setValue(8, types, 1);

    p_gamma = addFloatParam("gamma", "gamma");
    p_gamma->setValue(1.4f);

    p_cp = addFloatParam("cp", "cp");
    p_cp->setValue(1004.3f);
    p_T_or_c = paraSwitch("base", "Please choose a reference base");
    paraCase("Tref");
    p_Tref = addFloatParam("Tref", "Tref");
    p_Tref->setValue(293.0f);
    paraEndCase();
    paraCase("cref");
    p_cref = addFloatParam("cref", "cref");
    p_cref->setValue((float)sqrt(283 * 1.4 * 293));
    paraEndCase();
    paraEndSwitch();
    p_T_or_c->setValue(2);
}

int Solutions::compute(const char *)
{
    calctype = p_calctype->getValue();
    coDistributedObject *p_output = Calculate(p_density->getCurrentObject(), p_momentum->getCurrentObject(), p_energy->getCurrentObject(),
                                              p_rhou->getCurrentObject(), p_rhov->getCurrentObject(), p_rhow->getCurrentObject(),
                                              p_solution->getObjName());
    if (p_output)
    {
        p_solution->setCurrentObject(p_output);
    }
    else
    {
        return FAIL;
    }

    // done
    return CONTINUE_PIPELINE;
}

//////
////// the final Calculate-function
//////

coDistributedObject *Solutions::Calculate(const coDistributedObject *dens_in,
                                          const coDistributedObject *mom_in,
                                          const coDistributedObject *nrg_in, const coDistributedObject *rhou_in, const coDistributedObject *rhov_in, const coDistributedObject *rhow_in, const char *obj_name)
{

    // output stuff
    coDistributedObject *return_object = NULL;
    float *data_out[3] = { NULL, NULL, NULL };

    // input stuff
    //const char *dataType;
    //float mach, re, alpha, time_var;
    float *density, *energy;
    float *momentum[3] = { NULL, NULL, NULL };
    int npoints;
    //char *attrib;

    // temp stuff
    coDoVec3 *unstr_v3d;
    coDoFloat *unstr_s3d;
    //coDistributedObject *tmp_obj;

    // counters
    int i;

    // check for errors
    if (dens_in == NULL || nrg_in == NULL)
    {
        Covise::sendError("ERROR: input is garbled");
        return (NULL);
    }
    //if( dens_in->getAttribute("MACH")==NULL || dens_in->getAttribute("ALPHA")==NULL ||
    //		dens_in->getAttribute("RE")==NULL || dens_in->getAttribute("TIME")==NULL )
    //{
    //   Covise::sendError("ERROR: MACH, ALPHA, RE and TIME attributes not set !");
    //    return( NULL);
    //}

    //////
    ////// get input
    //////

    // density
    //dataType = dens_in->getType();
    unstr_s3d = (coDoFloat *)dens_in;
    unstr_s3d->getAddress(&density);
    //tmp_obj = unstr_s3d;
    npoints = unstr_s3d->getNumPoints();

    // attributes
    /* attrib = tmp_obj->getAttribute("MACH");
    sscanf(attrib, "%f", &mach);
    attrib = tmp_obj->getAttribute("RE");
    sscanf(attrib, "%f", &re);
    attrib = tmp_obj->getAttribute("ALPHA");
    sscanf(attrib, "%f", &alpha);
    attrib = tmp_obj->getAttribute("TIME");
    sscanf(attrib, "%f", &time_var);*/

    if (mom_in)
    {
        // momentum
        //dataType = mom_in->getType();
        unstr_v3d = (coDoVec3 *)mom_in;
        unstr_v3d->getAddresses(&(momentum[0]), &(momentum[1]), &(momentum[2]));
    }

    // energy
    //dataType = nrg_in->getType();
    unstr_s3d = (coDoFloat *)nrg_in;
    unstr_s3d->getAddress(&energy);
    if (rhou_in)
    {
        // rhou_in
        //dataType = rhou_in->getType();
        unstr_s3d = (coDoFloat *)rhou_in;
        unstr_s3d->getAddress(&(momentum[0]));
    }
    if (rhov_in)
    {
        // energy
        //dataType = rhov_in->getType();
        unstr_s3d = (coDoFloat *)rhov_in;
        unstr_s3d->getAddress(&(momentum[1]));
    }
    if (rhow_in)
    {
        // energy
        //dataType = rhow_in->getType();
        unstr_s3d = (coDoFloat *)rhow_in;
        unstr_s3d->getAddress(&(momentum[2]));
    }

    if (Get_Output_Type(calctype) == _VECTOR_OUTPUT
        && !mom_in
        && (!rhou_in || !rhov_in || !rhow_in))
    {
        sendError("Cannot make a vector out of scalars");
        return NULL;
    }

    if (calctype != _CALC_CP && (!momentum[0] || !momentum[1] || !momentum[2]))
    {
        sendError("Either momentum or all of rhou, rhov and rhow have to be connected");
        return NULL;
    }

    //////
    ////// create output object and get pointer(s)
    //////

    if (Get_Output_Type(calctype) == _VECTOR_OUTPUT)
    {
        unstr_v3d = new coDoVec3(obj_name, npoints);
        unstr_v3d->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
        return_object = unstr_v3d;
    }
    else
    {
        unstr_s3d = new coDoFloat(obj_name, npoints);
        unstr_s3d->getAddress(&(data_out[0]));
        return_object = unstr_s3d;
    }

    //////
    ////// compute the output
    //////
    float GAMMA, cp, Tref, cv, Rgas, cref, ggm1;
    // constants (maybe have to be given through module parameters !!!
    GAMMA = p_gamma->getValue();
    cp = p_cp->getValue();

    cv = cp / GAMMA;
    Rgas = (GAMMA - 1.0f) * cv;
    ggm1 = GAMMA / (GAMMA - 1.0f);

    if (p_T_or_c->getValue() == 2)
    {
        Tref = p_Tref->getValue();
        cref = sqrt(Tref * GAMMA * Rgas);
    }
    else
    {
        cref = p_cref->getValue();
        Tref = cref * cref / (GAMMA * Rgas);
    }

    // go
    float u, v, w, P;

    switch (calctype)
    {
    case _CALC_VELOCITY:
        for (i = 0; i < npoints; i++)
        {
            if (density[i] != 0.0)
            {
                data_out[0][i] = momentum[0][i] / density[i];
                data_out[1][i] = momentum[1][i] / density[i];
                data_out[2][i] = momentum[2][i] / density[i];
            }
            else
            {
                data_out[0][i] = 0;
                data_out[1][i] = 0;
                data_out[2][i] = 0;
            }
        }
        return_object->addAttribute("SPECIES", "vel");
        break;

    case _CALC_CP:
        for (i = 0; i < npoints; i++)
        {
            data_out[0][i] = (density[i] - 1.0f) / (0.567f);
        }
        return_object->addAttribute("SPECIES", "Cp");
        break;

    case _CALC_M:
        for (i = 0; i < npoints; i++)
        {
            prepare(density[i], momentum[0][i], momentum[1][i], momentum[2][i],
                    &u, &v, &w);

            P = (0.4f * (energy[i] - 0.5f * density[i] * (u + v + w)));

            data_out[0][i] = sqrt((density[i] * (u + v + w)) / 1.4f * P);
        }
        break;
        return_object->addAttribute("SPECIES", "M");

    case _CALC_P_STATIC:
        for (i = 0; i < npoints; i++)
        {
            prepare(density[i], momentum[0][i], momentum[1][i], momentum[2][i],
                    &u, &v, &w);

            P = (energy[i] - ((u + v + w) / (2.0f * density[i]))) / (density[i] * cv) + Tref;

            data_out[0][i] = P * density[i] * Rgas;
        }
        return_object->addAttribute("SPECIES", "P_STATIC");
        break;

    case _CALC_P_TOTAL:
        if (GAMMA == 1.)
            break;

        for (i = 0; i < npoints; i++)
        {
            prepare(density[i], momentum[0][i], momentum[1][i], momentum[2][i],
                    &u, &v, &w);

            P = (energy[i] - ((u + v + w) / (2.0f * density[i]))) / (density[i] * cv) + Tref;

            data_out[0][i] = (float)(P * density[i] * Rgas * pow((double)((P + 0.5f * sqrt(u + v + w) / density[i]) / P), (double)ggm1));
        }
        return_object->addAttribute("SPECIES", "P_TOTAL");
        break;

    case _CALC_T_STATIC:
        for (i = 0; i < npoints; i++)
        {
            prepare(density[i], momentum[0][i], momentum[1][i], momentum[2][i],
                    &u, &v, &w);

            P = (energy[i] - ((u + v + w) / (2.0f * density[i]))) / (density[i] * cv) + Tref;

            data_out[0][i] = P;
        }
        return_object->addAttribute("SPECIES", "T_STATIC");
        break;

    case _CALC_T_TOTAL:
        for (i = 0; i < npoints; i++)
        {
            if (density[i])
            {
                prepare(density[i], momentum[0][i], momentum[1][i], momentum[2][i],
                        &u, &v, &w);

                P = (energy[i] - ((u + v + w) / (2.0f * density[i]))) / (density[i] * cv) + Tref;

                data_out[0][i] = P + 0.5f * sqrt(u + v + w) / density[i];
            }
            else
            {
                Covise::sendWarning("Density zero detected");
                data_out[0][i] = 0;
            }
        }
        return_object->addAttribute("SPECIES", "T_TOTAL");
        break;

    case _CALC_MACH:
        for (i = 0; i < npoints; i++)
        {
            if (density[i])
            {
                prepare(density[i], momentum[0][i], momentum[1][i], momentum[2][i],
                        &u, &v, &w);

                P = (energy[i] - ((u + v + w) / (2.0f * density[i]))) / (density[i] * cv) + Tref;

                data_out[0][i] = sqrt(u + v + w) / (density[i] * sqrt(GAMMA * Rgas * P));
            }
            else
            {
                Covise::sendWarning("Density zero detected");
                data_out[0][i] = 0;
            }
        }
        return_object->addAttribute("SPECIES", "MACH");
        break;
    }

    // done
    return (return_object);
}

//////
////// tools
//////

void Solutions::prepare(float density, float momentum_x, float momentum_y, float momentum_z,
                        float *u, float *v, float *w)
{
    if (density)
    {
        *u = momentum_x / density;
        *v = momentum_y / density;
        *w = momentum_z / density;
    }
    else
    {
        *u = 0;
        *v = 0;
        *w = 0;
    }
    *u = (*u) * (*u);
    *v = (*v) * (*v);
    *w = (*w) * (*w);
    return;
}

int Solutions::Get_Output_Type(int n)
{
    int r;

    switch (n)
    {
    case _CALC_VELOCITY:
        r = _VECTOR_OUTPUT;
        break;

    default:
        r = _SCALAR_OUTPUT;
        break;
    }

    return (r);
}

MODULE_MAIN(Tools, Solutions)
