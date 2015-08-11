/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2015 HLRS       ++
// ++ Description: connect particle positions through lines               ++
// ++                                                                     ++
// ++ Author:  Uwe Woessner                                               ++
// ++                                                                     ++
// ++               HLRS                                                  ++
// ++               Nobelstrasse 19                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 09.08.2015                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <do/coDoGeometry.h>
#include <do/coDoSet.h>
#include <do/coDoIntArr.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoTriangleStrips.h>
#include "ParticleTrails.h"
#include <string>

//#define VERBOSE

//
// Constructor
//
ParticleTrails::ParticleTrails(int argc, char *argv[])
    : coModule(argc, argv, "Create trails for particles")
{
    
    pTrailLength = addInt32Param("maxTrailLength", "Maximum length of trails");
    pTrailLength->setValue(100);

    pPoints = addInputPort("points", "Points", "set of Points");
    pLines = addOutputPort("lines", "Lines", "set of Lines");
    pPoints->setRequired(true);

    pLines->setDependencyPort(pPoints);
}

//
// compute method
//
int
ParticleTrails::compute(const char *)
{
    
    const coDoSet *PointSetIn = NULL;
    const coDistributedObject *objIn = pPoints->getCurrentObject();
    PointSetIn = dynamic_cast<const coDoSet *>(objIn);
    if(PointSetIn !=NULL)
    {
        int numTimesteps = PointSetIn->getNumElements();
        std::string LinesName = pLines->getObjName();
        LinesName.append("_");
        
        const coDistributedObject **lines = new const coDistributedObject *[numTimesteps + 1];
        lines[numTimesteps]=NULL;
        int numPoints=0;
        std::vector<float> xc;
        std::vector<float> yc;
        std::vector<float> zc;
        int maxLength = pTrailLength->getValue();
        if(maxLength < 1)
            maxLength = 1;
        for(int i=0;i<numTimesteps;i++)
        {
            int numToRemove = i - maxLength;
            if(numToRemove < 0)
                numToRemove = 0;
            int traceLen = i;
            if(traceLen > maxLength)
                traceLen = maxLength;
            const coDistributedObject *di = PointSetIn->getElement(i);
            const coDoPoints *points = dynamic_cast<const coDoPoints *>(di);
            if(points !=NULL)
            {
                int numPts = points->getNumPoints();
                numPoints += numPts;
                float *x_c, *y_c, *z_c;
                points->getAddresses(&x_c,&y_c,&z_c);
                if(numToRemove > 0)
                {
                    xc.erase(xc.begin(),xc.begin() + numPts);
                    yc.erase(yc.begin(),yc.begin() + numPts);
                    zc.erase(zc.begin(),zc.begin() + numPts);
                }
                for(int c=0;c<numPts;c++)
                {
                    xc.push_back(x_c[c]);
                    yc.push_back(y_c[c]);
                    zc.push_back(z_c[c]);
                }
                int numVert;
                if(traceLen==0)
                {
                    numVert = (2)*numPts;
                }
                else
                {
                    numVert = (traceLen+1)*numPts;
                }
                int numLines = numPts;
                
                int *v_l = new int[numVert];
                int *l_l = new int[numLines];

                int nv=0;
                for(int l=0;l<numLines;l++)
                {
                    l_l[l]=nv;
                    if(traceLen==0)
                    {
                        v_l[nv] = l;
                        nv++;
                    }
                    for(int p = 0; p <= traceLen;p++)
                    {
                        v_l[nv] = l+p*numPts;
                        nv++;
                    }
                }
                std::string lineName = LinesName+std::to_string(i);
                coDoLines *line = new coDoLines(lineName,xc.size(),&xc[0],&yc[0],&zc[0],numVert,v_l,numLines,l_l);
                lines[i]=line;
            }
            else
            {
                sendError("Did not get points in the set");
                return STOP_PIPELINE;
            }
        }
        coDoSet *linesSet = new coDoSet(coObjInfo(pLines->getObjName()),lines);
        linesSet->addAttribute("TIMESTEP", "1 x");
        pLines->setCurrentObject(linesSet);

    }
    else
    {
        sendError("Did not receive a SET of points");
        return STOP_PIPELINE;
    }

return SUCCESS;
}

//
// Destructor
//
ParticleTrails::~ParticleTrails()
{
}

MODULE_MAIN(Tools, ParticleTrails)
