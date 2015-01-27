/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	                  **
 **                                                                        **
 ** Description: Displacement                                              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Benjamin Lechner (MPA)                                         **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <map>
#include <iostream>
#include <iomanip>

#include <do/coDoSet.h>
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include "Displacement.h"

Displacement::Displacement(int argc, char *argv[])
    : coModule(argc, argv, "Calculate transient Displacement to a reference timestep")
{

    m_pTimestep = addInt32Param("Timestep", "number of timestep from which to compare");
    m_pTimestep->setValue(0);

    m_portPoints = addInputPort("points", "Points", "points Input");
    m_portPoints->setInfo("points Input");

    m_portID = addInputPort("ID", "Int", "e.g. Atom Type");
    m_portID->setInfo("e.g. Atom Type");

    m_portDisplacement = addOutputPort("Displacement", "Vec3", "Displacement Output");
    m_portDisplacement->setInfo("Displacement Output");
}

Displacement::~Displacement()
{
}

// =======================================================

int Displacement::compute(const char *)
{

    coDistributedObject **pdisplacement;

    sendInfo("Covise was here ;)");
    // Get Points at Port
    coDistributedObject *obj = m_portPoints->getCurrentObject();

    if (!obj)
    {
        sendError("Did not receive an Object at Port %s", m_portPoints->getName());
        return STOP_PIPELINE;
    }

    if (!obj->isType("SETELE"))
    {
        sendError("Received illegal type at port %s \n", m_portPoints->getName());
        sendError("Be sure you got more than one timestep");
        return STOP_PIPELINE;
    }
    //Get ID's at Port if connected
    coDistributedObject *objID = m_portID->getCurrentObject();

    if (!objID)
    {
        sendError("Did not receive an Object at Port %s", m_portID->getName());
        return STOP_PIPELINE;
    }

    if (!objID->isType("SETELE"))
    {
        sendError("Received illegal type at port %s \n", m_portID->getName());
        sendError("Be sure you got more than one timestep");
        return STOP_PIPELINE;
    }

    coDoSet *inData = (coDoSet *)obj;
    coDoSet *inIDs = (coDoSet *)objID;

    int numberSteps;
    int numberStepsID;
    coDistributedObject *const *points = inData->getAllElements(&numberSteps);
    coDistributedObject *const *IDs = inIDs->getAllElements(&numberStepsID);

    if (numberSteps != numberStepsID)
    {
        sendError("Points has not the same amount of timesteps as ID");
    }

    // get comparing coord triple
    if (m_pTimestep->getValue() > numberSteps)
    {
        sendError("Timestep to compare is too high");
        return STOP_PIPELINE;
    }

    int numberPoints;
    coDoPoints *tocompare = dynamic_cast<coDoPoints *>(points[m_pTimestep->getValue()]);
    numberPoints = tocompare->getNumPoints();

    coDoInt *IDstocompare = dynamic_cast<coDoInt *>(IDs[m_pTimestep->getValue()]);
    if (numberPoints != IDstocompare->getNumPoints())
    {
        sendError("Amount of Atoms and IDs is not the same in Timestep to compare");
    }

    float *x, *y, *z;
    float *tx, *ty, *tz;
    int *ID, *tID;
    float *erg_x, *erg_y, *erg_z;

    tocompare->getAddresses(&tx, &ty, &tz);
    IDstocompare->getAddress(&tID);

    pdisplacement = new coDistributedObject *[numberSteps + 1];
    pdisplacement[numberSteps] = NULL;

    int pointstocalc;
    for (int timestep = 0; timestep < numberSteps; timestep++)
    {
        //calculate displacement for each step
        cerr << "computing timestep: " << timestep << endl;
        coDoPoints *compare = dynamic_cast<coDoPoints *>(points[timestep]);
        compare->getAddresses(&x, &y, &z);
        pointstocalc = compare->getNumPoints();
        coDoInt *IDscompare = dynamic_cast<coDoInt *>(IDs[timestep]);
        IDscompare->getAddress(&ID);

        coDoVec3 *pdisplace = new coDoVec3(coObjInfo(m_portDisplacement->getName()), numberPoints);
        pdisplace->getAddresses(&erg_x, &erg_y, &erg_z);

        for (int j = 0; j < pointstocalc; j++)
        {
            //calculate displacement for each Point
            for (int k = 0; k < numberPoints; k++)
            {
                //search for the same ID
                if (tID[j] == ID[k])
                {
                    //here calculate the displacement
                    erg_x[k] = x[k] - tx[j];
                    erg_y[k] = y[k] - ty[j];
                    erg_z[k] = z[k] - tz[j];
                    break;
                }
            }
        }
        //again in step loop
        pdisplacement[timestep] = pdisplace;
    }

    coDoSet *DisplacementSet = new coDoSet(coObjInfo(m_portDisplacement->getObjName()), pdisplacement);
    char ts[100];
    sprintf(ts, "1 %d", numberSteps);
    DisplacementSet->addAttribute("TIMESTEP", ts);

    m_portDisplacement->setCurrentObject(DisplacementSet);

    /* Das ist ein Test im Umgang mit Objekten
	float *x,*y,*z;
	int no;
	inData->getAddresses(&x,&y,&z);
	no=inData->getNumPoints();
	coDoPoints *outData = new coDoPoints(coObjInfo(m_portPoints->getName()),no,x,y,z);

	//m_portTest->setCurrentObject(outData);*/

    return SUCCESS;
}

MODULE_MAIN(IO, Displacement)
