/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Filter program in COVISE API                           ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// COVISE data types
#include <do/coDoUnstructuredGrid.h>
// this includes our own class's headers
#include "Enlarge.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Enlarge::Enlarge(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Enlarge, world! program")
{
    // Parameters

    // create the parameter
    p_scale = addFloatSliderParam("scale", "Scale factor for Grid");

    // set the dafault values
    p_scale->setValue(0.1f, 5.0f, 2.0f);

    // Ports
    p_inPort = addInputPort("inPort", "UnstructuredGrid", "Grid input");
    p_outPort = addOutputPort("outPort", "UnstructuredGrid", "Grid output");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Enlarge::compute(const char *port)
{
    (void)port;

    const coDistributedObject *obj = p_inPort->getCurrentObject();

    // we should have an object
    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort->getName());
        return FAIL;
    }

    // it should be the correct type
    if (!obj->isType("UNSGRD"))
    {
        sendError("Received illegal type at port '%s'", p_inPort->getName());
        return FAIL;
    }

    // So, this is an unstructured grid
    const coDoUnstructuredGrid *inGrid = (const coDoUnstructuredGrid *)obj;

    // retrieve the size and the list pointers
    int numElem, numConn, numCoord, hasTypes;
    int *inElemList = NULL, *inConnList = NULL, *inTypeList = NULL;
    float *inXCoord, *inYCoord, *inZCoord;

    inGrid->getGridSize(&numElem, &numConn, &numCoord);
    hasTypes = inGrid->hasTypeList();

    inGrid->getAddresses(&inElemList, &inConnList,
                         &inXCoord, &inYCoord, &inZCoord);
    if (hasTypes)
        inGrid->getTypeList(&inTypeList);

    // allocate new Unstructured grid of same size
    coDoUnstructuredGrid *outGrid
        = new coDoUnstructuredGrid(p_outPort->getObjName(),
                                   numElem, numConn, numCoord,
                                   hasTypes);

    if (!outGrid->objectOk())
    {
        sendError("Failed to create object '%s' for port '%s'",
                  p_outPort->getObjName(), p_outPort->getName());
        return FAIL;
    }

    int *outElemList = NULL, *outConnList = NULL, *outTypeList = NULL;
    float *outXCoord, *outYCoord, *outZCoord;

    outGrid->getAddresses(&outElemList, &outConnList,
                          &outXCoord, &outYCoord, &outZCoord);
    if (hasTypes)
        outGrid->getTypeList(&outTypeList);

    /// copy element list
    for (int i = 0; i < numElem; i++)
        outElemList[i] = inElemList[i];

    /// if we have one, copy the types list
    if (hasTypes)
        for (int i = 0; i < numElem; i++)
            outTypeList[i] = inTypeList[i];

    /// copy connectivity list
    for (int i = 0; i < numConn; i++)
        outConnList[i] = inConnList[i];

    /// retrieve parameter
    float scale = p_scale->getValue();

    /// copy coordinates and scale them
    for (int i = 0; i < numCoord; i++)
    {
        outXCoord[i] = inXCoord[i] * scale;
        outYCoord[i] = inYCoord[i] * scale;
        outZCoord[i] = inZCoord[i] * scale;
    }

    // finally, assign object to port
    p_outPort->setCurrentObject(outGrid);

    return SUCCESS;
}

MODULE_MAIN(Examples, Enlarge)
