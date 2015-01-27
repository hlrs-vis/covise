/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                (C)2001 VirCinity GmbH  **
 ** Description:   BlockCollect                                            **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date: 26.09.97                                                         **
 **       03.04.2001 Sven Kufer: changed to new API                        **
 **       27.09.2002 Sven Kufer: redesign
\**************************************************************************/

#include <do/coDoGeometry.h>
#include <do/coDoSet.h>
#include "BlockCollect.h"

using namespace covise;

BlockCollect::BlockCollect(int argc, char *argv[])
    : coModule(argc, argv, "create one big set out of many many little ones")
{
    int i;
    char portname[20];
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(portname, "inport_%d", i);
        p_inport[i] = addInputPort(portname, "coDistributedObject", "input object");
        if (i > 0)
            p_inport[i]->setRequired(0);
    }

    p_outport = addOutputPort("set_out", "coDistributedObject", "output object");

    p_mode = addChoiceParam("mode", "mode");
    const char *choLabels[] = { "Set of Elements", "Cat Timesteps", "Merge Blocks", "Set of Timesteps", "Set of Blocks", "Cat Blocks" };
    p_mode->setValue(6, choLabels, SetOfElements);
}

int BlockCollect::NoTimeSteps()
{
    int i, j, res = 0;
    const coDistributedObject *tmp_obj;
    aktivePorts = 0;
    int no_elements;

    if (mode == SetOfBlocks)
    {
        tmp_obj = p_inport[0]->getCurrentObject();
        if (dynamic_cast<const coDoSet *>(tmp_obj))
        {
            res = (static_cast<const coDoSet *>(tmp_obj))->getNumElements();
        }
        for (i = 0; i < NUM_PORTS; ++i)
        {
            tmp_obj = p_inport[i]->getCurrentObject();
            if (tmp_obj)
                aktivePorts++;
        }
    }
    else if (mode == CatBlocks)
    {
        for (i = 0; i < NUM_PORTS; ++i)
        {
            tmp_obj = p_inport[i]->getCurrentObject();
            if (!tmp_obj)
                continue;
            if (dynamic_cast<const coDoSet *>(tmp_obj))
            {
                const coDistributedObject *const *setList = ((coDoSet *)(tmp_obj))->getAllElements(&no_elements);
                for (j = 0; j < no_elements; ++j)
                {
                    if (dynamic_cast<const coDoSet *>(setList[j]))
                    {
                        res += (static_cast<const coDoSet *>(setList[j]))->getNumElements();
                    }
                    else
                    {
                        res++;
                    }
                }
            }
            else
            {
                res++;
            }
        }
    }
    else
    {
        for (i = 0; i < NUM_PORTS; ++i)
        {
            tmp_obj = p_inport[i]->getCurrentObject();
            if (!tmp_obj)
                continue;
            if (dynamic_cast<const coDoGeometry *>(tmp_obj))
                return 0; // handle objects always as static if one of them is a geometry object
            if (dynamic_cast<const coDoSet *>(tmp_obj) && mode != SetOfElements && mode != SetOfTimesteps)
            {
                // we have transient data
                res += (static_cast<const coDoSet *>(tmp_obj))->getNumElements();
            }
            else
            {
                // we have static data
                if (mode == MergeBlocks)
                {
                    return -1; // we have static data but also transient data
                }
                res++;
            }
        }
    }
    return res;
}

int BlockCollect::compute(const char *)
{
    int i;

    mode = (CollectMode)p_mode->getValue();

    const coDistributedObject *tmp_obj;
    const coDistributedObject *real_obj;

    coDoSet *set_output;

    real_obj = p_inport[0]->getCurrentObject();
    if (real_obj == NULL)
    {
        sendError("Object at inport_0 not found");
        return STOP_PIPELINE;
    }

    int nume = 0;
    const coDistributedObject **outobjs;
    const coDistributedObject *p_ForAttributes = NULL;

    // 1 if all elements
    int time_steps = NoTimeSteps(); // -1 : mix of static and transient data

    if (mode == SetOfBlocks)
    {
        outobjs = new const coDistributedObject *[time_steps + 1];
        outobjs[time_steps] = NULL;
        int numSets = time_steps * aktivePorts;
        const coDistributedObject **tmpobjs = new const coDistributedObject *[numSets + 1];
        tmpobjs[numSets] = NULL;
        char name[256];

        // now go through all ports
        for (i = 0; i < NUM_PORTS; i++)
        {
            tmp_obj = p_inport[i]->getCurrentObject();

            if (tmp_obj)
            {
                if (dynamic_cast<const coDoSet *>(tmp_obj))
                {
                    int j, no_elements;
                    const coDistributedObject *const *setList = ((coDoSet *)(tmp_obj))->getAllElements(&no_elements);
                    for (j = 0; j < no_elements; ++j)
                    {
                        tmpobjs[nume++] = setList[j];
                        setList[j]->incRefCount();
                    }
                }
                /*else
               {
                  tmpobjs[nume++] = tmp_obj;
                  tmp_obj->incRefCount();
               }*/
            }
        }
        const coDistributedObject **tmp_blocks;
        coDoSet *tmp_set;
        int att = 0;
        for (int step = 0; step < time_steps; step++)
        {
            int number = 0;
            int count = 0;
            for (i = step; i < numSets; i += time_steps)
            {
                if (dynamic_cast<const coDoSet *>(tmpobjs[i]))
                {
                    number += (static_cast<const coDoSet *>(tmpobjs[i]))->getNumElements();
                }
            }
            tmp_blocks = new const coDistributedObject *[number + 1];
            tmp_blocks[number] = NULL;
            for (i = step; i < numSets; i += time_steps)
            {
                if (dynamic_cast<const coDoSet *>(tmpobjs[i]))
                {
                    int elem, k;
                    const coDistributedObject *const *blockList = ((coDoSet *)(tmpobjs[i]))->getAllElements(&elem);
                    if (att == 0)
                    {
                        p_ForAttributes = tmpobjs[i];
                        att = -1;
                    }
                    for (k = 0; k < elem; ++k)
                    {
                        tmp_blocks[count++] = blockList[k];
                        blockList[k]->incRefCount();
                    }
                }
            }
            //in ein set packen
            sprintf(name, "%s_%d", p_outport->getObjName(), step);
            tmp_set = new coDoSet(name, tmp_blocks);
            //set in outobjs an stelle step
            outobjs[step] = tmp_set;
        }
    }
    else if (mode == CatBlocks)
    {
        outobjs = new const coDistributedObject *[time_steps + 1];
        outobjs[time_steps] = NULL;
        // now go through all ports
        for (i = 0; i < NUM_PORTS; i++)
        {
            tmp_obj = p_inport[i]->getCurrentObject();
            if (tmp_obj)
            {
                if (dynamic_cast<const coDoSet *>(tmp_obj))
                {
                    int j, no_elements;
                    const coDistributedObject *const *setList = ((coDoSet *)(tmp_obj))->getAllElements(&no_elements);
                    if (nume == 0)
                        p_ForAttributes = tmp_obj;
                    for (j = 0; j < no_elements; ++j)
                    {
                        if (dynamic_cast<const coDoSet *>(setList[j]))
                        {
                            int k, no_elements2;
                            const coDistributedObject *const *setList2 = ((coDoSet *)(setList[j]))->getAllElements(&no_elements2);
                            for (k = 0; k < no_elements2; k++)
                            {
                                outobjs[nume++] = setList2[k];
                                setList2[k]->incRefCount();
                            }
                        }
                        else
                        {
                            outobjs[nume++] = setList[j];
                            setList[j]->incRefCount();
                        }
                    }
                }
                else
                {
                    outobjs[nume++] = tmp_obj;
                    tmp_obj->incRefCount();
                }
            }
        }
    }
    else if (time_steps < 0 && mode == MergeBlocks)
    {
        // inconsistent data
        sendError("You may not mix static and transient data");
        return STOP_PIPELINE;
    }
    else if (time_steps == 0 || mode == SetOfElements || mode == SetOfTimesteps)
    {
        time_steps = 0;
        if (mode == MergeBlocks)
        {
            sendError("You need a set if you want to merge");
            return STOP_PIPELINE;
        }

        // static data
        outobjs = new const coDistributedObject *[NUM_PORTS + 1];

        // now go through all ports
        for (i = 0; i < NUM_PORTS; i++)
        {
            tmp_obj = p_inport[i]->getCurrentObject();

            if (tmp_obj != NULL)
            {
                outobjs[nume] = tmp_obj;
                outobjs[nume]->incRefCount();
                nume++;
            }
            outobjs[nume] = NULL;
            if (mode == CatTimesteps)
            {
                time_steps = nume;
            }
            p_ForAttributes = NULL;
        }
    }
    else
    {
        // transient data
        if (mode != MergeBlocks)
        {
            outobjs = new const coDistributedObject *[time_steps + 1];
            outobjs[time_steps] = NULL;
            // now go through all ports
            for (i = 0; i < NUM_PORTS; i++)
            {
                tmp_obj = p_inport[i]->getCurrentObject();

                if (tmp_obj)
                {
                    if (dynamic_cast<const coDoSet *>(tmp_obj))
                    {
                        int j, no_elements;
                        const coDistributedObject *const *setList = ((coDoSet *)(tmp_obj))->getAllElements(&no_elements);
                        if (nume == 0)
                            p_ForAttributes = tmp_obj;
                        for (j = 0; j < no_elements; ++j)
                        {
                            outobjs[nume++] = setList[j];
                            setList[j]->incRefCount();
                        }
                    }
                    else
                    {
                        outobjs[nume++] = tmp_obj;
                        tmp_obj->incRefCount();
                    }
                }
            }
        }
        else
        {
            const coDistributedObject *const *setList[NUM_PORTS];
            int dummy;
            for (i = 0; i < NUM_PORTS; i++)
            {
                tmp_obj = p_inport[i]->getCurrentObject();
                if (tmp_obj != NULL)
                {
                    setList[i] = ((coDoSet *)(tmp_obj))->getAllElements(&dummy);
                }
            }

            int real_steps = ((coDoSet *)p_inport[0]->getCurrentObject())->getNumElements();
            outobjs = new const coDistributedObject *[real_steps + 1];
            outobjs[real_steps] = NULL;
            int j;
            for (j = 0; j < real_steps; j++)
            {
                // assume all input objects have the same structure
                int used_ports = time_steps / real_steps;
                int part_num = 0;

                const coDistributedObject **parts = new const coDistributedObject *[used_ports + 1];
                parts[used_ports] = NULL;
                for (i = 0; i < NUM_PORTS; i++)
                {
                    tmp_obj = p_inport[i]->getCurrentObject();
                    if (tmp_obj != NULL)
                    {
                        parts[part_num++] = setList[i][j];
                        setList[i][j]->incRefCount();
                    }
                }
                char objname[512];
                sprintf(objname, "%s_%d", p_outport->getObjName(), j);
                outobjs[nume++] = new coDoSet(objname, parts);
            }
            if ((p_inport[0]->getCurrentObject())->getAttribute("TIMESTEP"))
            {
                time_steps = real_steps;
            }
            else //merge of blocks
            {
                time_steps = 0;
            }
        }
    }

    set_output = new coDoSet(p_outport->getObjName(), outobjs);
    if ((mode != CatBlocks)
        && (time_steps > 0
            || mode == CatTimesteps
            || mode == SetOfTimesteps
            || ((p_inport[0]->getCurrentObject())->getAttribute("TIMESTEP") && mode == MergeBlocks)))
    {
        // put timestep attribute
        char buf[16];
        sprintf(buf, "1 %d", time_steps);
        set_output->addAttribute("TIMESTEP", buf);

        // use p_ForAttributes to add additional attibutes
        if (p_ForAttributes)
        {
            const char **attributes;
            const char **values;
            int no_attributes = p_ForAttributes->getAllAttributes(&attributes, &values);
            int j;
            // add all attributes but do not repeat TIMESTEP
            for (j = 0; j < no_attributes; ++j)
            {
                if (strcmp("TIMESTEP", attributes[j]) != 0)
                {
                    set_output->addAttribute(attributes[j], values[j]);
                }
            }
        }
    }

    p_outport->setCurrentObject(set_output);
    delete[] outobjs;
    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Tools, BlockCollect)
