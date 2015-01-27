/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: get grid data as scalar values                         ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  2000-02-26  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "UsgIndex.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
USGdata::USGdata()
    : coSimpleModule("USG index analysis")
{
    // Parameters

    // create the parameter
    p_whatOut = addChoiceParam("whatout", "What output to deliver");

    // set the dafault values
    static const char *whatOut[] = { "CellNo", "VertexNo" };
    p_whatOut->setValue(2, whatOut, 0);

    // Ports
    p_inPort = addInputPort("inPort", "coDoUnstructuredGrid", "Grid input");
    p_outPortF = addOutputPort("outPortF", "coDoFloat", "Data output");
    p_outPortI = addOutputPort("outPortI", "coDoIntArr", "Data output int");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int USGdata::compute()
{
    int i;

    coDistributedObject *obj = p_inPort->getCurrentObject();

    // it should be the correct type
    if (!obj->isType("UNSGRD"))
    {
        sendError("Received illegal type at port '%s'", p_inPort->getName());
        return FAIL;
    }

    // So, this is an unstructured grid
    coDoUnstructuredGrid *inGrid = (coDoUnstructuredGrid *)obj;

    // retrieve the size and the list pointers
    int numElem, numConn, numCoord;
    int *inElemList, *inConnList;
    float *inXCoord, *inYCoord, *inZCoord;

    inGrid->getGridSize(&numElem, &numConn, &numCoord);

    inGrid->getAddresses(&inElemList, &inConnList,
                         &inXCoord, &inYCoord, &inZCoord);

    // only sizes differ...
    int numOut;
    switch (p_whatOut->getValue())
    {
    case 0: // "CellNo"
        numOut = numElem;
        break;

    case 1: // "VertexNo"
        numOut = numCoord;
        break;

    default:
        sendError("Illegal choice value");
        return STOP_PIPELINE;
    }

    coDoFloat *outDataF
        = new coDoFloat(p_outPortF->getObjName(), numOut);
    coDoIntArr *outDataI
        = new coDoIntArr(p_outPortI->getObjName(), 1, &numOut);
    if (!outDataI->objectOk() || !outDataF->objectOk())
    {
        sendError("Failed to create output object");
        return STOP_PIPELINE;
    }

    float *dataF;
    outDataF->getAddress(&dataF);

    int *dataI;
    outDataI->getAddress(&dataI);

    for (i = 0; i < numOut; i++)
    {
        dataI[i] = i;
        dataF[i] = i;
    }

    // finally, assign object to port
    p_outPortI->setCurrentObject(outDataI);
    p_outPortF->setCurrentObject(outDataF);

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    USGdata *application = new USGdata;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
