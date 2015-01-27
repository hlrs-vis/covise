/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MakeBlockTransient.h"

MakeBlockTransient::MakeBlockTransient(int argc, char *argv[])
    : coModule(argc, argv, "Make a multiblock-grid transient (use same grid for all timesteps)")
{
    p_inport = addInputPort("inport", "coDistributedObject", "input port");
    p_accordingTo = addInputPort("accordingTo", "coDistributedObject", "accordingTo");
    p_accordingTo->setRequired(0);
    p_outport = addOutputPort("outport", "coDistributedObject", "output port");
    p_blocks = addInt32Param("blocks", "number of blocks");
    p_blocks->setValue(3);
    p_timesteps = addInt32Param("timesteps", "number of time steps");
    p_timesteps->setValue(100);
};

int MakeBlockTransient::Diagnose()
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
    if (p_blocks->getValue() < 0)
    {
        sendWarning("Only a non-negative number of blocks is acceptable!");
        return -1;
    }

    return 0;
}

int MakeBlockTransient::compute(const char *)
{
    int i, j;
    coDistributedObject *ret_set;
    coDistributedObject *timeset;
    coDistributedObject *const *inObjs = NULL;
    bool is_timeset = false;

    if (Diagnose() < 0)
        return FAIL;

    int numBlocks;
    // New: we have an input object at the 2nd port: take its # of elements
    int numSteps;
    int orgnumSteps;
    coDistributedObject *accObj = p_accordingTo->getCurrentObject();

    if (accObj && accObj->isType("SETELE"))
    {
        numSteps = ((coDoSet *)accObj)->getNumElements();
    }
    else
        numSteps = p_timesteps->getValue();

    if (in_obj_->isType("SETELE"))
    {
        inObjs = ((coDoSet *)in_obj_)->getAllElements(&orgnumSteps);
        numBlocks = orgnumSteps;
        is_timeset = true;
    }
    else
    {
        numBlocks = p_blocks->getValue();
    }

    coDistributedObject **block_list = new coDistributedObject *[numBlocks + 1];
    block_list[numBlocks] = NULL;

    coDistributedObject **set_list = new coDistributedObject *[numSteps + 1];
    set_list[numSteps] = NULL;

    for (i = 0; i < numBlocks; i++)
    {
        block_list[i] = NULL;
    }

    for (j = 0; j < numBlocks; j++)
    {
        set_list[j] = NULL;
    }

    char buf[16];
    char Name[300];

    sprintf(buf, "1 %d", numSteps);

    cerr << "NumBlocks=" << numBlocks << endl;

    for (j = 0; j < numBlocks; j++)
    {
        for (i = 0; i < numSteps; ++i)
        {
            if (coDoSet *set = dynamic_cast<coDoSet *>(inObjs[j]))
            {
                coDistributedObject *const *setChildren = NULL;
                int noChildren;
                setChildren = set->getAllElements(&noChildren);
                if (noChildren == 1)
                    set_list[i] = setChildren[0];
                else
                {
                }
                inObjs[j]->incRefCount();
            }
        }

        sprintf(Name, "%s_%d", p_outport->getObjName(), j);

        set_list[numSteps] = NULL;
        timeset = new coDoSet(Name, set_list);

        timeset->addAttribute("TIMESTEP", buf);

        block_list[j] = timeset;
        block_list[j]->incRefCount();
    }

    block_list[numBlocks] = NULL;

    ret_set = new coDoSet(p_outport->getObjName(), block_list);

    delete[] set_list;

    delete[] block_list;

    p_outport->setCurrentObject(ret_set);

    return SUCCESS;
}

MODULE_MAIN(Tools, MakeBlockTransient)
