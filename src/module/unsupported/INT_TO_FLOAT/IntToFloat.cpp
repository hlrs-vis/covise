/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "IntToFloat.h"

IntToFloat::IntToFloat()
    : coSimpleModule("Int to float conversor")
{
    p_int = addInputPort("intsIn", "coDoIntArr", "input integer array");
    p_float = addOutputPort("floatsOut", "coDoFloat",
                            "output float array");
}

int
IntToFloat::compute()
{
    // test input
    if (p_int->getCurrentObject() == NULL || !p_int->getCurrentObject()->isType("INTARR"))
    {
        sendError("Could not get input object or is not an INTARR");
        return FAIL;
    }
    // get input info
    coDoIntArr *intInput = dynamic_cast<coDoIntArr *>(p_int->getCurrentObject());
    int size = 1; // = intInput->getSize() - 1; // strange, but necessary
    int num_dimensions = intInput->getNumDimensions();
    int dimension;
    for (dimension = 0; dimension < num_dimensions; ++dimension)
    {
        size *= intInput->get_dim(dimension);
    }
    int *intArray = intInput->getAddress();

    // create output object
    coDoFloat *floatOutput = new coDoFloat(
        p_float->getObjName(), size);
    // fill in output info
    float *f_addr;
    floatOutput->getAddress(&f_addr);
    int elem;
    for (elem = 0; elem < size; ++elem)
    {
        f_addr[elem] = intArray[elem];
    }
    // assign output object to port
    p_float->setCurrentObject(floatOutput);
    return SUCCESS;
}

int
main(int argc, char *argv[])
{
    // create the module
    IntToFloat *application = new IntToFloat;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
