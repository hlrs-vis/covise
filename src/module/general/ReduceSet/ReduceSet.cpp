/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                              (C)2003 RUS  VirCinity ++
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

#include <do/coDoSet.h>
// this includes our own class's headers
#include "ReduceSet.h"

static const int maxDataPortNm_ = 8;
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReduceSet::ReduceSet(int argc, char *argv[])
    : coModule(argc, argv, "Skip data in transient datasets")
{

    // Parameters

    // create the parameter
    pFactor_ = addIntSliderParam("factor", "the factor the dataset is reduced by");

    // set the dafault values
    pFactor_->setValue(1, 10, 2);
    std::string pInBaseNm("input_");
    std::string pOutBaseNm("output_");
    // Ports

    pInPorts_ = new coInputPort *[maxDataPortNm_];
    pOutPorts_ = new coOutputPort *[maxDataPortNm_];
    int i;
    for (i = 0; i < maxDataPortNm_; ++i)
    {
        char cNm[3];
        sprintf(cNm, "%i", i);
        // input port
        std::string portName(pInBaseNm + std::string(cNm));
        pInPorts_[i] = addInputPort(portName.c_str(), "coDistributedObject", "data set");
        // output port
        portName = pOutBaseNm + std::string(cNm);
        pOutPorts_[i] = addOutputPort(portName.c_str(), "coDistributedObject", "data set");

        // port attributes
        pInPorts_[i]->setRequired(0);

        pOutPorts_[i]->setDependencyPort(pInPorts_[i]);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReduceSet::compute(const char *)
{
    int i;
    int factor = pFactor_->getValue();
    for (i = 0; i < maxDataPortNm_; ++i)
    {
        const coDistributedObject *gridInObj = pInPorts_[i]->getCurrentObject();
        // do we have a set ?
        if (gridInObj)
        {
            if (gridInObj->isType("SETELE"))
            {
                // get a pointer to the set in SHM
                const coDoSet *shmSet;
                if (!(shmSet = dynamic_cast<const coDoSet *>(gridInObj)))
                {
                    // this should never occur
                    cerr << "FATAL error:  GetSetElem::compute( ) dynamic cast failed in line "
                         << __LINE__ << " of file " << __FILE__ << endl;
                }
                shmSet = (const coDoSet *)pInPorts_[i]->getCurrentObject();
                int numSteps;

                const coDistributedObject *const *shmSetElem = shmSet->getAllElements(&numSteps);
                // get the set-elem now
                // coDistributedObject *shmSetElem = shmSet->getElement( thisElem );

                if (NULL != shmSetElem)
                {
                    int needed = 1 + numSteps / factor;
                    if (0 == numSteps % factor)
                    {
                        needed--;
                    }

                    const coDistributedObject **setList = new const coDistributedObject *[1 + needed];
                    setList[needed] = NULL;
                    int j;
                    int k = 0;
                    for (j = 0; j < numSteps; j++)
                    {
                        if (0 == j % factor)
                        {
                            setList[k] = shmSetElem[j];
                            shmSetElem[j]->incRefCount();
                            k++;
                        }
                    }
                    coDistributedObject *retSet;
                    retSet = new coDoSet(pOutPorts_[i]->getObjName(), setList);
                    delete[] setList;
                    char buf[16];
                    sprintf(buf, "1 %d", k);
                    retSet->addAttribute("TIMESTEP", buf);

                    pOutPorts_[i]->setCurrentObject(retSet);
                }
                else
                {
                    // this should never occur
                    cerr << "FATAL error:  GetSetElem::compute( ) Got NULL in line "
                         << __LINE__ << " of file " << __FILE__ << endl;
                }
            }
            else
            {
                sendError("Did not receive a SET object at port %s. You don'tneed ReduceSet", pInPorts_[i]->getName());
                sendInfo("Did not receive a SET object at port %s. You don't need ReduceSet", pInPorts_[i]->getName());

                return STOP_PIPELINE;
            }
        }
    }
    return SUCCESS;
}

MODULE_MAIN(Filter, ReduceSet)
