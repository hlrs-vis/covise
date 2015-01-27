/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <do/coDoGeometry.h>
#include <do/coDoSet.h>
#include "SetOfReferences.h"

SetOfReferences::SetOfReferences(int argc, char *argv[])
    : coModule(argc, argv, "Create a set of an arbitrary number of elements by adding references to the provided object.")
{
    p_inport = addInputPort("element_in", "coDistributedObject", "input object");
    p_outport = addOutputPort("set_out", "coDistributedObject", "output object");

    p_count = addInt32Param("Count", "number of elements");
    p_count->setValue(10);

    p_timestep = addBooleanParam("TimeStep", "Add the TIMESTEP attribute");
    p_timestep->setValue(0);
}

int SetOfReferences::compute(const char *)
{
    // get and check count
    int count = p_count->getValue();
    if (count <= 0)
        return STOP_PIPELINE;

    // get and check input object
    const coDistributedObject *in_obj = p_inport->getCurrentObject();
    if (!in_obj)
        return STOP_PIPELINE;

    // create references
    coDistributedObject **out_objs = new coDistributedObject *[count + 1];
    out_objs[count] = NULL;
    for (int i = 0; i < count; ++i)
    {
        out_objs[i] = const_cast<coDistributedObject *>(in_obj);
        out_objs[i]->incRefCount();
    }

    // create set
    coDoSet *set_output = new coDoSet(p_outport->getObjName(), out_objs);

    // put timestep attribute
    if (p_timestep->getValue())
    {
        char buf[16];
        sprintf(buf, "1 %d", count);
        set_output->addAttribute("TIMESTEP", buf);
    }

    p_outport->setCurrentObject(set_output);
    delete[] out_objs;
    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Tools, SetOfReferences)
