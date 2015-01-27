/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class GetSetElem                      ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 04.10.2001                                                    ++
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
#include "GetSetelem.h"
#include <string>

//#define VERBOSE

//
// Constructor
//
GetSetElem::GetSetElem(int argc, char *argv[])
#ifdef GET_SETELEM
    : coModule(argc, argv, "Extract set-element")
#else
    : coModule(argc, argv, "Extract subset")
#endif
{
// Parameters
#ifdef GET_SETELEM
    // create the parameters
    pIndex_ = addInt32Param("stepNo", "number of set-element");
    pIndex_->setValue(1);
#else
    pSelection_ = addStringParam("selection", "ranges of selected set elements");
    pSelection_->setValue("1-1");
#endif

    // Ports
    pInPorts_ = new coInputPort *[maxDataPortNm_];
    pOutPorts_ = new coOutputPort *[maxDataPortNm_];

    std::string pInBaseNm("DataIn");
    std::string pOutBaseNm("DataOut");

    const int notRequired = 0;

    for (int i = 0; i < maxDataPortNm_; ++i)
    {
        char cNm[3];
        sprintf(cNm, "%i", i);
        // input port
        std::string portName(pInBaseNm + std::string(cNm));
        pInPorts_[i] = addInputPort(portName.c_str(), "coDistributedObject", "data set");
        // output port
        portName = pOutBaseNm + std::string(cNm);
        pOutPorts_[i] = addOutputPort(portName.c_str(), "coDistributedObject", "data element");

        // port attributes
        pInPorts_[i]->setRequired(notRequired);

        pOutPorts_[i]->setDependencyPort(pInPorts_[i]);
    }
}

//
// compute method
//
int
GetSetElem::compute(const char *)
{
// retrieve parameters

#ifdef GET_SETELEM
    int thisElem = pIndex_->getValue() - 1; // we make it compatible with the old version of this module
#else
    //selection
    coRestraint selection;
    selection.add(pSelection_->getValue());

    //retrieve number of selected elements n_el;
    int n_el = 0;
    int min = selection.lower();
    int max = selection.upper();

    for (int loop = min; loop <= max; loop++)
    {
        if (selection(loop) != 0)
        {
            ++n_el;
        }
    }

#ifdef VERBOSE
    fprintf(stderr, "n_el=%d\n", n_el);
#endif

    //fill parameter list
    int *subsetElements = new int[n_el];
    int counter = selection.lower();
    int limit = selection.upper();
    int index = 0;

#ifdef VERBOSE
    fprintf(stderr, "mapping:");
#endif

    while (counter <= limit)
    {
        if (selection(counter) && (index < n_el))
        {
#ifdef VERBOSE
            fprintf(stderr, " %d->%d ", counter, index);
#endif
            subsetElements[index] = counter;
            ++index;
        }
        else if (selection(counter) && (index >= n_el))
        {
            cout << '\n' << flush << "error in subset element list\n" << flush;
        }
        else
        {
        }

        ++counter;
    }
#ifdef VERBOSE
    fprintf(stderr, "\n");
#endif
#endif

    ////////////////////////////////////////////////////
    //       P O R T S
    ////////////////////////////////////////////////////

    const coDistributedObject *gridInObj = NULL;

    for (int i = 0; i < maxDataPortNm_; ++i)
    {
        if (gridInObj == NULL || !gridInObj->isType("GEOMET"))
        {
            gridInObj = pInPorts_[i]->getCurrentObject();
        }
        // do we have a set ?
        if (gridInObj)
        {
            if (gridInObj->isType("SETELE") || gridInObj->isType("GEOMET"))
            {
                // get a pointer to the set in SHM
                const coDoSet *shmSet;
                if (!(shmSet = (const coDoSet *)(gridInObj)))
                {
                    // this should never occur
                    cerr << "FATAL error:  GetSetElem::compute( ) dynamic cast failed in line "
                         << __LINE__ << " of file " << __FILE__ << endl;
                }

// is the index reasonable?
#ifdef GET_SETELEM
                thisElem = adjustIndex(thisElem, shmSet);
#else
                for (int ind = 0; ind < n_el; ind++)
                {
                    subsetElements[ind] = adjustIndex(subsetElements[ind], shmSet);
                }

                // get the set-elem now
                if (n_el == 1)
                {
#endif
                const coDistributedObject *shmSetElem = NULL;
                if (const coDoGeometry *geo = dynamic_cast<const coDoGeometry *>(gridInObj))
                {
                    switch (i)
                    {
                    case 0:
                        shmSetElem = geo->getGeometry();
                        break;
                    case 1:
                        shmSetElem = geo->getColors();
                        break;
                    case 2:
                        shmSetElem = geo->getNormals();
                        break;
                    case 3:
                        shmSetElem = geo->getTexture();
                        break;
                    default:
                        shmSetElem = NULL;
                    }
                }
                else
                {
#ifdef GET_SETELEM
                    shmSetElem = shmSet->getElement(thisElem);
#else
                        shmSetElem = shmSet->getElement(subsetElements[0]);
#endif
                }
                if (shmSetElem)
                {

                    coDistributedObject *outObj = shmSetElem->clone(pOutPorts_[i]->getObjName());
/******************************************************************************
                                               - not tested yet -

                  const char *feedback=shmSet->getAttribute("FEEDBACK");
                  if( feedback )
                  {
                     outObj->addAttribute("FEEDBACK", feedback );
                  }
                  ********************************************************************************/

#ifdef GET_SETELEM
                    // add attribute for BlockCollect to execute pipeline step by step
                    // create module indification string
                    std::string module_id("!"); // just for compatibility
                    module_id += std::string(Covise::get_module()) + std::string("\n") + Covise::get_instance() + std::string("\n");
                    module_id += std::string(Covise::get_host()) + std::string("\n");
                    outObj->addAttribute("BLOCK_FEEDBACK", module_id.c_str());

                    int no = shmSet->getNumElements();

                    outObj->addAttribute("NEXT_STEP_PARAM", "stepNo\nIntScalar\n");
                    // increase stepNo by 1
                    char stepNr[32];
                    sprintf(stepNr, "%d", thisElem + 2);
                    outObj->addAttribute("NEXT_STEP", stepNr);

                    // show last elem
                    if (thisElem + 2 > no)
                    {
                        if (shmSet->getAttribute("TIMESTEP"))
                        {
                            outObj->addAttribute("LAST_STEP", " ");
                        }
                        else
                        {
                            outObj->addAttribute("LAST_BLOCK", " ");
                        }
                    }
#endif

                    pOutPorts_[i]->setCurrentObject(outObj);
                }

#ifndef GET_SETELEM
            }
            else
            {
                const coDistributedObject **subset = new const coDistributedObject *[n_el + 1];

                for (int index = 0; index < n_el; index++)
                {
                    subset[index] = shmSet->getElement(subsetElements[index]);
                    subset[index]->incRefCount();
                }
                subset[n_el] = NULL;
                coDoSet *outObj = new coDoSet(pOutPorts_[i]->getObjName(), subset);
                outObj->copyAllAttributes(shmSet);

                const char *timestep = shmSet->getAttribute("TIMESTEP");
                if (timestep)
                {
                    char steps[32];
                    sprintf(steps, "0 %i", n_el);
                    outObj->addAttribute("TIMESTEP", steps);
                }

                pOutPorts_[i]->setCurrentObject(outObj);

                delete[] subset;
            }
#endif
        }
        else
        {
            sendError("Did not receive a SET object at port %s. You don't need GetSetelem", pInPorts_[i]->getName());
            sendInfo("Did not receive a SET object at port %s. You don't need GetSetelem", pInPorts_[i]->getName());

            return STOP_PIPELINE;
        }
    }
}
return SUCCESS;
}

int
GetSetElem::adjustIndex(const int &idx, const coDoSet *obj)
{
    // is the chosen index reasonable
    int numElem = obj->getNumElements();
    int retIdx;
    if (idx >= numElem)
    {
        // if the given number is too big we use the number of sets
        retIdx = numElem - 1;
        char chNum[8];
        sprintf(chNum, "%i", numElem);
        sendInfo("The number of the set-element to extract is too big. We use %s instead.", chNum);
#ifdef GET_SETELEM
        pIndex_->setValue(numElem);
#endif
    }
    else if (idx < 0)
    {
        // if the given number is too small we use the number one
        retIdx = 0;
        sendInfo("The number of the set-element to extract is too small. We use 1 instead.");
#ifdef GET_SETELEM
        pIndex_->setValue(retIdx + 1);
#endif
    }
    else
    {
        retIdx = idx;
    }
    return retIdx;
}

//
// Destructor
//
GetSetElem::~GetSetElem()
{
}

MODULE_MAIN(Filter, GetSetElem)
