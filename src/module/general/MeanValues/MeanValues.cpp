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
    p_mesh = addInputPort("indices", "Int", "indices");
    p_mesh->setRequired(0);
    p_solution = addOutputPort("meanValues", "Float|Vec3", "meanValues");
    p_numContrib = addOutputPort("numContributors", "Int", "number of contributors");

    const char *types[] = { "Average", "Accumulate" };
    p_calctype = addChoiceParam("calctype", "how should the mean values be computed");
    p_calctype->setValue(2, types, 0);
}

int MeanValues::compute(const char *)
{
    calctype = p_calctype->getValue();
    coDistributedObject *p_output = Calculate(p_data->getCurrentObject(), p_mesh->getCurrentObject(), p_solution->getObjName(), p_numContrib->getObjName());
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

coDistributedObject *MeanValues::Calculate(const coDistributedObject *data_in, const coDistributedObject *index_in, const char *obj_name, const char *nameNumContrib)
{

    // output stuff
    coDistributedObject *return_object = NULL;
    int npoints = 0, nindex=0, numOut=0;
    // temp stuff
    const coDoInt *indexData = dynamic_cast<const coDoInt *>(index_in);
    if (index_in && !indexData)
    {
        sendError("indices format wrong");
        return NULL;
    }
    coDoInt *numOccOut=NULL;
    int *numOccurances=NULL;
    if (indexData)
    {
        nindex = indexData->getNumPoints();
        const int *ind = indexData->getAddress();
        int minIndex = INT_MAX, maxIndex = INT_MIN;
        for (int i=0; i<nindex; ++i)
        {
            if (ind[i] < minIndex) minIndex = ind[i];
            if (ind[i] > maxIndex) maxIndex = ind[i];
        }
        if (minIndex < 0)
        {
            sendError("indices array contains negative values");
            return NULL;
        }

        numOut = maxIndex+1;
        numOccOut = new coDoInt(nameNumContrib, numOut);
        numOccurances = numOccOut->getAddress();
        memset(numOccurances, 0, sizeof(*numOccurances)*numOut);
    }
    //coDistributedObject *tmp_obj;

    // check for errors
    /* if( dens_in==NULL  || nrg_in==NULL )
   {
      Covise::sendError("ERROR: input is garbled");
      return( NULL );
   }*/
    //dataType = dens_in->getType();
    double mean = 0;
    const coDoFloat *scalarData = dynamic_cast<const coDoFloat *>(data_in);
    if (scalarData != NULL)
    {
        float *sData = scalarData->getAddress();
        npoints = scalarData->getNumPoints();
        for (int i = 0; i < npoints; i++)
        {
            mean += sData[i];
        }
        mean /= npoints;
        sendInfo("scalar mean = %f", (float)mean);
    }

    if (!indexData)
        numOut = npoints;

    double meanx = 0;
    double meany = 0;
    double meanz = 0;
    const coDoVec3 *vectorData = dynamic_cast<const coDoVec3 *>(data_in);
    if (vectorData != NULL)
    {
        float *vx=NULL, *vy=NULL, *vz=NULL;
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

    if (vectorData)
    {
        coDoVec3 *vDataOut = new coDoVec3(obj_name, numOut);
        float *data_out[3]={NULL, NULL, NULL};
        vDataOut->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
        return_object = vDataOut;
        if (indexData)
        {
            for (int c=0; c<3; ++c)
                memset(data_out[c], 0, sizeof(*data_out[c])*numOut);
            float *vx=NULL, *vy=NULL, *vz=NULL;
            vectorData->getAddresses(&vx, &vy, &vz);
            const int *ind = indexData->getAddress();
            for (int i = 0; i < npoints; i++)
            {
                int idx = ind[i];
                data_out[0][idx] += vx[i];
                data_out[1][idx] += vy[i];
                data_out[2][idx] += vz[i];
                ++numOccurances[idx];
            }
            if (calctype == MEAN_AVG)
            {
                for (int i=0; i<numOut; ++i)
                {
                    if (numOccurances[i] > 0)
                    {
                        data_out[0][i] /= numOccurances[i];
                        data_out[1][i] /= numOccurances[i];
                        data_out[2][i] /= numOccurances[i];
                    }

                }
            }
        }
        else
        {
            float mx = (float)meanx;
            float my = (float)meany;
            float mz = (float)meanz;
            for (int i = 0; i < npoints; i++)
            {
                data_out[0][i] = mx;
                data_out[1][i] = my;
                data_out[2][i] = mz;
            }
        }
    }
    else
    {
        coDoFloat *sDataOut = new coDoFloat(obj_name, numOut);
        float *data_out = sDataOut->getAddress();
        return_object = sDataOut;
        if (indexData)
        {
            memset(data_out, 0, sizeof(*data_out)*numOut);
            const float *sData = scalarData->getAddress();
            const int *ind = indexData->getAddress();
            for (int i = 0; i < npoints; i++)
            {
                int idx = ind[i];
                data_out[idx] += sData[i];
                ++numOccurances[idx];
            }
            if (calctype == MEAN_AVG)
            {
                for (int i=0; i<numOut; ++i)
                {
                    if (numOccurances[i] > 0)
                    {
                        data_out[i] /= numOccurances[i];
                    }

                }
            }
        }
        else
        {
            float m = (float)mean;
            for (int i = 0; i < npoints; i++)
            {
                data_out[i] = m;
            }
        }
    }

    if (indexData)
    {
        p_numContrib->setCurrentObject(numOccOut);
    }

    // done
    return (return_object);
}

MODULE_MAIN(Tools, MeanValues)
