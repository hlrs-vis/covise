/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoSet.h>
#include "MakeTransient.h"

MakeTransient::MakeTransient(int argc, char *argv[])
    : coModule(argc, argv, "Make a dynamic set out of a static object")
{
    p_inport = addInputPort("inport", "coDistributedObject", "input port");
    p_accordingTo = addInputPort("accordingTo", "coDistributedObject", "accordingTo");
    p_accordingTo->setRequired(0);
    p_outport = addOutputPort("outport", "coDistributedObject", "output port");
    p_timesteps = addInt32Param("timesteps", "number of time steps");
    p_timesteps->setValue(72);
};

int MakeTransient::Diagnose()
{
    if (!(in_obj_ = p_inport->getCurrentObject()))
    {
        sendError("Did not receive any input object");
        return -1;
    }
    else if (!in_obj_->objectOk())
    {
        sendError("Did not get a good input object");
        return -1;
    }
    /*else if( in_obj_->isType("SETELE") && in_obj_->getAttribute("TIMESTEP") ) {
      sendError("only static objects are acceptable for input");
      return -1;
   }*/
    if (p_timesteps->getValue() < 0 && p_accordingTo->getCurrentObject() == NULL)
    {
        sendWarning("Only a non-negative number of time steps is acceptable: assuming 0");
        p_timesteps->setValue(0);
    }
    return 0;
}

int MakeTransient::compute(const char *)
{
    int i;
    coDistributedObject *ret_set;
    const coDistributedObject *const *inObjs = NULL;
    bool is_timeset = false;

    if (Diagnose() < 0)
        return FAIL;

    // New: we have an input object at the 2nd port: take its # of elements
    int numSteps;
    int orgnumSteps;
    const coDistributedObject *accObj = p_accordingTo->getCurrentObject();
    if (accObj && accObj->isType("SETELE"))
    {
        numSteps = ((coDoSet *)accObj)->getNumElements();
    }
    else
        numSteps = p_timesteps->getValue();

    if (in_obj_->isType("SETELE") && in_obj_->getAttribute("TIMESTEP") != NULL)
    {
        inObjs = ((coDoSet *)in_obj_)->getAllElements(&orgnumSteps);
        numSteps *= orgnumSteps;
        is_timeset = true;
    }

    const coDistributedObject **set_list = new const coDistributedObject *[numSteps + 1];
    set_list[numSteps] = 0;
    for (i = 0; i < numSteps; ++i)
    {
        if (!is_timeset)
        {
            set_list[i] = in_obj_;
            in_obj_->incRefCount();
        }
        else
        {
            set_list[i] = inObjs[i % orgnumSteps];
            set_list[i]->incRefCount();
        }
    }
    ret_set = new coDoSet(p_outport->getObjName(), set_list);
    delete[] set_list;

    char buf[16];
    sprintf(buf, "1 %d", numSteps);
    ret_set->addAttribute("TIMESTEP", buf);
    p_outport->setCurrentObject(ret_set);
    return SUCCESS;
}

MODULE_MAIN(Tools, MakeTransient)
