/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                     (C) 2001 VirCinity **
 **                                                                        **
 ** Description:   COVISE  MeanValues module                               **
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
 **                                                                        **
\**************************************************************************/

#include "MeanValues.h"
#include <do/coDoData.h>

MeanValues::MeanValues(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Calculate Mean Values of scalor or vector data data")
{
    p_data = addInputPort("data", "Float|Vec3", "dataValues");
    p_mesh = addInputPort("mesh", "Polygons", "surface mesh");
    p_mesh->setRequired(0);
    p_solution = addOutputPort("meanValues", "Float|Vec3", "meanValues");

    const char *types[] = { "Simple", "Weighted" };
    p_calctype = addChoiceParam("calctype", "how should the mean values be computed");
    p_calctype->setValue(2, types, 0);
}

int MeanValues::compute(const char *)
{
    calctype = p_calctype->getValue();
    coDistributedObject *p_output = Calculate(p_data->getCurrentObject(), p_mesh->getCurrentObject(), p_solution->getObjName());
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

coDistributedObject *MeanValues::Calculate(const coDistributedObject *data_in, const coDistributedObject *mesh_in, const char *obj_name)
{

    // output stuff
    coDistributedObject *return_object = NULL;
    int npoints;
    // temp stuff
    const coDoVec3 *vectorData;
    const coDoFloat *scalarData;
    //coDistributedObject *tmp_obj;

    // check for errors
    /* if( dens_in==NULL  || nrg_in==NULL )
   {
      Covise::sendError("ERROR: input is garbled");
      return( NULL );
   }*/
    //dataType = dens_in->getType();
    float *sData = NULL;
    float *vx = NULL, *vy = NULL, *vz = NULL;
    double mean = 0;
    scalarData = dynamic_cast<const coDoFloat *>(data_in);
    if (scalarData != NULL)
    {
        scalarData->getAddress(&sData);
        npoints = scalarData->getNumPoints();
        for (int i = 0; i < npoints; i++)
        {
            mean += sData[i];
        }
        mean /= npoints;
        sendInfo("scalar mean = %f", (float)mean);
    }

    double meanx = 0;
    double meany = 0;
    double meanz = 0;
    vectorData = dynamic_cast<const coDoVec3 *>(data_in);
    if (vectorData != NULL)
    {
        vectorData->getAddresses(&vx, &vy, &vz);
        npoints = vectorData->getNumPoints();
        for (int i = 0; i < npoints; i++)
        {
            meanx += vx[i];
            meany += vy[i];
            meanz += vz[i];
        }
        meanx /= npoints;
        meany /= npoints;
        meanz /= npoints;
        sendInfo("vector mean = %f %f %f, length %f", (float)meanx, (float)meany, (float)meanz, (float)sqrt(meanx * meanx + meany * meany + meanz * meanz));
    }

    /*  if(calctype != _MEAN_SIMPLE && (!momentum[0] || !momentum[1] || !momentum[2]))
   {
      sendError("Either momentum or all of rhou, rhov and rhow have to be connected");
      return NULL;
   }*/

    //////
    ////// create output object and get pointer(s)
    //////

    coDoVec3 *vDataOut;
    coDoFloat *sDataOut;
    float *data_out[3];
    if (sData == NULL)
    {
        vDataOut = new coDoVec3(obj_name, npoints);
        vDataOut->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
        float mx = (float)meanx;
        float my = (float)meany;
        float mz = (float)meanz;
        for (int i = 0; i < npoints; i++)
        {
            data_out[0][i] = mx;
            data_out[1][i] = my;
            data_out[2][i] = mz;
        }
        return_object = vDataOut;
    }
    else
    {
        sDataOut = new coDoFloat(obj_name, npoints);
        sDataOut->getAddress(&(data_out[0]));
        float m = (float)mean;
        for (int i = 0; i < npoints; i++)
        {
            data_out[0][i] = m;
        }
        return_object = sDataOut;
    }

    // done
    return (return_object);
}

MODULE_MAIN(Tools, MeanValues)
