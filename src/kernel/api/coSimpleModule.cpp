/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDistributedObject.h>
#include <do/coDoSet.h>
#include "coSimpleModule.h"

#define __DEBUG_MSG 0

using namespace covise;

coSimpleModule::coSimpleModule(int argc, char *argv[], const char *desc, bool propagate)
    : coModule(argc, argv, desc, propagate)
{
    compute_timesteps = 0;
    compute_multiblock = 0;

    copy_attributes_flag = 1;
    copy_attributes_non_set_flag = 1;

    cover_interaction_flag = 0;
    object_level = 0;

    timestep_flag = 0;
    multiblock_flag = 0;

    portLeader = 0;
    return;
}

/////////////////////////////////////////////////////////////////////////////////////////

void coSimpleModule::copyAttributesToOutObj(coInputPort **input_ports,
                                            coOutputPort **output_ports, int i)
{
    int j;
    if (numInPorts == 0)
        return;
    if (i >= numInPorts)
        j = 0;
    else
        j = i;
    if (input_ports[j] && output_ports[i])
        copyAttributes(output_ports[i]->getCurrentObject(), input_ports[j]->getCurrentObject());
}

/////////////////////////////////////////////////////////////////////////////////////////

void coSimpleModule::copyAttributes(coDistributedObject *tgt, const coDistributedObject *src) const
{
    int n;
    const char **name, **setting;

#if __DEBUG_MSG
    cerr << "coSimpleModule::copyAttributes" << endl;
#endif

    if (src && tgt)
    {
        n = src->getAllAttributes(&name, &setting);
        if (n > 0)
            tgt->addAttributes(n, name, setting);
    }

    return;
}

/////////////////////////////////////////////////////////////////////////////////////////

void coSimpleModule::localCompute(void *callbackData)
{
    (void)callbackData;
    int i, ni, no;

    for (i = 0; i < d_numElem; i++)
        elemList[i]->preCompute();

    // this flag turns to false when errors occur
    bool continueExec = true;

    // get number of ports
    ni = no = 0;
    for (i = 0; i < d_numElem; i++)
    {
        if (elemList[i]->kind() == coUifElem::INPORT)
            ni++;
        else if (elemList[i]->kind() == coUifElem::OUTPORT)
            no++;
    }

#if __DEBUG_MSG
    cerr << "coSimpleModule::localCompute" << endl;
    cerr << "   in: " << ni << "   out: " << no << endl;
#endif
    // and get the ports
    originalInPorts = new coInputPort *[ni + 1];
    originalInPorts[ni] = NULL;
    originalOutPorts = new coOutputPort *[no + 1];
    originalOutPorts[no] = NULL;
    numInPorts = ni;
    numOutPorts = no;
    ni = no = 0;
    for (i = 0; i < d_numElem; i++)
    {
        if (elemList[i]->kind() == coUifElem::INPORT)
        {
            originalInPorts[ni] = (coInputPort *)elemList[i];

#if __DEBUG_MSG
            cerr << "   in: " << originalInPorts[ni]->getName() << endl;
#endif

            ni++;
        }
        else if (elemList[i]->kind() == coUifElem::OUTPORT)
        {
            originalOutPorts[no] = (coOutputPort *)elemList[i];

#if __DEBUG_MSG
            cerr << "   out: " << originalOutPorts[no]->getName() << endl;
#endif

            no++;
        }
    }

    // sl: Action called before handling the tree structure
    //  @@@ continueExec = ( preHandleObjects(originalInPorts) == CONTINUE_PIPELINE );
    preHandleObjects(originalInPorts);

    portLeader = 0;
    while (originalInPorts[portLeader] != NULL && originalInPorts[portLeader]->getCurrentObject() == NULL)
    {
        portLeader++;
        if (originalInPorts[portLeader] == NULL)
        {
            continueExec = false;
            break;
        }
    }

    // handle the objects
    if (continueExec)
    {
        continueExec = (handleObjects(originalInPorts, originalOutPorts) == CONTINUE_PIPELINE);
    };

    if (cover_interaction_flag == 1)
    {
        if (originalOutPorts[0]->getCurrentObject() != NULL)
            (originalOutPorts[0]->getCurrentObject())->addAttribute("FEEDBACK", INTattribute);
    }

    // sl:
    //  @@@ if ( continueExec )
    //  @@@ {
    //  @@@    continueExec = ( postHandleObjects(originalOutPorts) == CONTINUE_PIPELINE );
    //  @@@ }
    for (i = 0; i < no; i++)
    {
        coDistributedObject *obj = originalOutPorts[i]->getCurrentObject();
        if (NULL != obj)
        {
            if (!obj->checkObject())
            {
                sendError("Object Integrity check failed: see console");
                stopPipeline();
            }

            if (_propagateObjectName)
            {
                obj->addAttribute("OBJECTNAME", this->getTitle());
            }
        }
    }
    postHandleObjects(originalOutPorts);

    // clean up
    delete[] originalInPorts;
    delete[] originalOutPorts;

    for (i = 0; i < d_numElem; i++)
        elemList[i]->postCompute();

    if (!continueExec)
        send_stop_pipeline();
}

/////////////////////////////////////////////////////////////////////////////////////////

void coSimpleModule::swapObjects(coInputPort **inPorts, coOutputPort **outPorts)
{
    // swap:  inPorts>objs   outPorts>objnames
    coDistributedObject **outObjs;
    const coDistributedObject **inObjs;
    char **objNames;
    int i;

#if __DEBUG_MSG
    cerr << "coSimpleModule::swapObjects" << endl;
#endif

    inObjs = new const coDistributedObject *[numInPorts];
    objNames = new char *[numOutPorts];
    outObjs = new coDistributedObject *[numOutPorts];

    for (i = 0; i < numInPorts; i++)
        inObjs[i] = originalInPorts[i]->getCurrentObject();
    for (i = 0; i < numOutPorts; i++)
    {
        const char *outName = originalOutPorts[i]->getObjName();
        objNames[i] = strcpy(new char[strlen(outName) + 1], outName);
        outObjs[i] = originalOutPorts[i]->getCurrentObject();
    }

    // and swap
    for (i = 0; i < numInPorts; i++)
    {
        originalInPorts[i]->setCurrentObject(inPorts[i]->getCurrentObject());
        inPorts[i]->setCurrentObject(inObjs[i]);
    }
    for (i = 0; i < numOutPorts; i++)
    {
        originalOutPorts[i]->setObjName(outPorts[i]->getObjName());
        outPorts[i]->setObjName(objNames[i]);
        originalOutPorts[i]->setCurrentObject(outPorts[i]->getCurrentObject());
        outPorts[i]->setCurrentObject(outObjs[i]);
    }

    for (i = 0; i < numOutPorts; i++)
        delete[] objNames[i];
    delete[] objNames;
    delete[] inObjs;
    delete[] outObjs;

    // done
    return;
}

/////////////////////////////////////////////////////////////////////////////////////////

int coSimpleModule::handleObjects(coInputPort **inPorts, coOutputPort **outPorts)
{
    int compute_flag;
    const char *dataType;

    int i, j;

    // this flag is set to false if the execution should be terminated
    bool continueExec = true;

#if __DEBUG_MSG
    cerr << "coSimpleModule::handleObjects" << endl;
#endif
    if (inPorts[portLeader] == NULL)
    {
        compute_flag = 0;
    }
    else
    {

        // check for error
        if (NULL == inPorts || NULL == inPorts[portLeader] || NULL == inPorts[portLeader]->getCurrentObject())
        {
            return STOP_PIPELINE;
        }

        // handle current object
        dataType = (inPorts[portLeader]->getCurrentObject())->getType();
        if (0 == strcmp(dataType, "SETELE"))
        {
            // this is a set
            if ((inPorts[portLeader]->getCurrentObject())->getAttribute("TIMESTEP"))
            {
                compute_flag = !compute_timesteps; // transient
                timestep_flag = 1;
                multiblock_flag = 0;
            }
            else
            {
                compute_flag = !compute_multiblock; // multiblock
                multiblock_flag = 1;
            }
        }
        else
        {
            compute_flag = 0;
            multiblock_flag = 0;
            timestep_flag = 0;
        } // no set
    }
#if __DEBUG_MSG
    cerr << "   " << dataType << "   " << compute_flag << endl;
#endif

    if (!compute_flag)
    {
        // the user will handle this object
        swapObjects(inPorts, outPorts);
#if __DEBUG_MSG
        cerr << "   in-obj : " << originalInPorts[portLeader]->getCurrentObject()->getName() << endl;
        cerr << "   out-obj: " << originalOutPorts[0]->getObjName() << endl;
#endif

        // execute compute-callback
        if (compute(NULL) != CONTINUE_PIPELINE)
            continueExec = false;

#if __DEBUG_MSG
        cerr << "   computed out-obj (org.): " << originalOutPorts[0]->getCurrentObject()->getName() << endl;
#endif
        // restore original values
        swapObjects(inPorts, outPorts);
#if __DEBUG_MSG
        cerr << "   computed out-obj: " << outPorts[0]->getCurrentObject()->getName() << endl;
#endif

        // copy attributes
        if (copy_attributes_flag && copy_attributes_non_set_flag)
        {
            for (i = 0; i < numOutPorts; i++)
            {
                copyAttributesToOutObj(inPorts, outPorts, i);
            }
        }
    }
    else
    {
        // ack, we have to split the set and do the work
        const coDistributedObject ***setInObjs = new const coDistributedObject **[numInPorts];
        const coDistributedObject *const *setContent;
        coDistributedObject ***setOutObjs = new coDistributedObject **[numOutPorts];
        int numSetElem, t;
        coInputPort **newInPorts;
        coOutputPort **newOutPorts;
        char newObjName[2048];

        object_level++; // object is part of set

        element_counter.push_back(0);
        num_elements.push_back(1);

        // get input objects
        t = -1;
        // see sl: 1. below
        char *deleteNewInPorts = new char[numInPorts];
        memset(deleteNewInPorts, 'y', numInPorts);

        for (i = 0; i < numInPorts; i++)
        {
            if (inPorts[i] != NULL)
            {
                const coDistributedObject *in_obj = inPorts[i]->getCurrentObject();
                if (in_obj != NULL) // added 06/20/2000 by lf_te
                {
                    if (const coDoSet *set = dynamic_cast<const coDoSet *>(in_obj)) //added 03/01/2001 by sc
                    {
                        setContent = set->getAllElements(&numSetElem);
                        setInObjs[i] = new const coDistributedObject *[numSetElem];
                        for (j = 0; j < numSetElem; j++)
                        {
                            setInObjs[i][j] = setContent[j];
                        }

                        if (t == -1)
                        {
                            t = numSetElem;
                        }

                        if (numSetElem == 0) // added 03/12/2001 by sk
                        {
                            numSetElem = t;
                            setInObjs[i] = new const coDistributedObject *[numSetElem];
                            for (int j = 0; j < numSetElem; j++)
                                setInObjs[i][j] = NULL;
                        }
                        else if (numSetElem != t)
                        {
                            // serious error -> data inconsistency
                            Covise::sendError("coSimpleModule: stale object structure");
                            return STOP_PIPELINE;
                        }
                    }
                    else //added 03/01/2001 by sc
                        if (t == -1)
                        setInObjs[i] = NULL;
                    else
                    {
                        setInObjs[i] = new const coDistributedObject *[t];
                        // sl: 1. do not forget that setInObjs[i][j]
                        // will be associated below to newInPorts[i].
                        // So, be careful not to delete the object pointed
                        // at by in_obj when deleting newInPorts[i],
                        // because this object will be deleted elsewhere
                        // (coInputPort::postCompute)
                        deleteNewInPorts[i] = 'n';
                        // in this case remember also that setInObjs[i]
                        // has to be deleted-[], or else a memry leak appears
                        for (int j = 0; j < t; j++)
                            setInObjs[i][j] = in_obj;
                    }
                }
                else
                    setInObjs[i] = NULL;
            }
            else
                setInObjs[i] = NULL;
        }

#if __DEBUG_MSG
        cerr << "   numSetElem: " << numSetElem << endl;
#endif

        // create empty sets for output-objects
        for (i = 0; i < numOutPorts; i++)
        {
            setOutObjs[i] = new coDistributedObject *[numSetElem + 1];
            for (j = 0; j < numSetElem + 1; j++)
                setOutObjs[i][j] = NULL;
        }

        // create the ports for the recursive calls
        newInPorts = new coInputPort *[numInPorts];
        newOutPorts = new coOutputPort *[numOutPorts];
        for (i = 0; i < numInPorts; i++)
            newInPorts[i] = new coInputPort(inPorts[i]->getName(), "", "coSimpleModule - internal port");
        for (i = 0; i < numOutPorts; i++)
            newOutPorts[i] = new coOutputPort(outPorts[i]->getName(), "", "coSimpleModule - internal port");

        num_elements.back() = numSetElem;

        // call the compute callback
        for (t = 0; (t < numSetElem) && continueExec; t++)
        {
            element_counter.back() = t;
            setIterator(inPorts, t); //sl:
            // pre
            for (i = 0; i < numInPorts; i++)
            {
                if (setInObjs[i] != NULL)
                    newInPorts[i]->setCurrentObject(setInObjs[i][t]);
                else
                    newInPorts[i]->setCurrentObject(NULL);
            }
            for (i = 0; i < numOutPorts; i++)
            {
                sprintf(newObjName, "%s_%d", outPorts[i]->getObjName(), t);
                newOutPorts[i]->setObjName(newObjName);
                newOutPorts[i]->setCurrentObject(NULL);
            }

            // handle

            if (handleObjects(newInPorts, newOutPorts) != CONTINUE_PIPELINE)
                continueExec = false;

            // post
            if (continueExec)
            {
                for (i = 0; i < numOutPorts; i++)
                {
                    setOutObjs[i][t] = newOutPorts[i]->getCurrentObject();
                }
            }
        }

        // assemble set(s)
        for (i = 0; i < numOutPorts; i++)
        {
            if (outPorts[i]->getObjName())
            {
                coDistributedObject **setOutObjsCompacted = new coDistributedObject *[numSetElem + 1];
                memset(setOutObjsCompacted, 0, (numSetElem + 1) * sizeof(coDistributedObject *));
                int currentOutObj = 0;
                // Compact the array that no 0 elements are present
                for (t = 0; (t < numSetElem) && continueExec; ++t)
                {
                    setOutObjsCompacted[currentOutObj] = setOutObjs[i][t];
                    if (setOutObjsCompacted[currentOutObj] != 0)
                        ++currentOutObj;
                }

                if (currentOutObj > 0)
                {
                    outPorts[i]->setCurrentObject(new coDoSet(coObjInfo(outPorts[i]->getObjName()), setOutObjsCompacted));
                    if (copy_attributes_flag)
                    {
                        // sk, we: 10/01/01 removed wrong loop:
                        // sk:     for( i=0; i<numOutPorts; i++ ) changed '>' to '>='
                        // sl: 17.07.01 use virtual function copyAttributesToOutObj

                        copyAttributesToOutObj(inPorts, outPorts, i);
                    }
                }
                else
                    outPorts[i]->setCurrentObject(NULL);
            }
            else
                outPorts[i]->setCurrentObject(NULL);
        }

        // clean up
        for (i = 0; i < numInPorts; i++)
        {
            // see sl: 1.
            if (deleteNewInPorts[i] == 'n')
            {
                newInPorts[i]->setCurrentObject(NULL);
                // in this case remember also that setInObjs[i]
                // has to be deleted [], or else a memry leak appears
                delete[] setInObjs[i];
            }
            delete newInPorts[i];
        }
        delete[] newInPorts;
        for (i = 0; i < numOutPorts; i++)
        {
            newOutPorts[i]->setCurrentObject(NULL); // the associated object is destroed below
            delete newOutPorts[i];
        }
        delete[] newOutPorts;
        // Make sure that set element objects are destroyed!!!
        for (i = 0; i < numOutPorts; i++)
        {
            int set_elem;
            for (set_elem = 0; setOutObjs[i][set_elem]; ++set_elem)
            {
                delete setOutObjs[i][set_elem];
            }
        }
        for (i = 0; i < numOutPorts; i++)
            delete[] setOutObjs[i];
        delete[] setOutObjs;
        delete[] setInObjs;
        delete[] deleteNewInPorts;
        object_level--; // leave set
        element_counter.pop_back();
        num_elements.pop_back();
    }

    // done
    return continueExec ? CONTINUE_PIPELINE : STOP_PIPELINE;
}

int coSimpleModule::getObjectLevel() const
{
    return object_level;
}

int coSimpleModule::getElementNumber(int level) const
{
    if (level < -1 || level > object_level || object_level == 0)
        return -1;
    if (level == -1)
        return element_counter.back();
    return element_counter[level];
}

int coSimpleModule::getNumberOfElements(int level) const
{
    if (level < -1 || level > object_level || object_level == 0)
        return -1;
    if (level == -1)
        return num_elements.back();
    return num_elements[level];
}
