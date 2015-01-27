/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                               (C) 2002 VirCinity GmbH  **
 ** Description: PipelineCollect                                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Sven Kufer		                                          **
 **         (C)  VirCinity IT- Consulting GmbH                             **
 **         Nobelstrasse 15                               		  **
 **         D- 70569 Stuttgart    			       		  **
 **                                                                        **
 **  17.10.2001                                                            **
\**************************************************************************/

#ifndef _PIPELINE_COLLECT_H
#define _PIPELINE_COLLECT_H

// includes
#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <util/covise_list.h>

// defines
#define NUM_PORTS 5

class PipelineCollect : public coModule
{

private:
    // 'static' stuff
    int num_timesteps; // number of collected timesteps or blocks
    int first_step; // first step before recursion

    List<coDistributedObject> store[NUM_PORTS];

    //ports
    coInputPort *p_inport[NUM_PORTS];
    coOutputPort *p_outport[NUM_PORTS];

    // parameters
    coIntScalarParam *param_skip;

    // functions

    int NoTimeSteps(); // -1 if we mix static data and data sets
    // 0 if we have only static data
    // number of time steps for output
    //       with transient data
    /// create copy of object with new name
    coDistributedObject *createNewObj(const char *name, const coDistributedObject *obj) const;

public:
    virtual int compute(const char *port);
    PipelineCollect(int argc, char *argv[]);
};
#endif // _PIPELINE_COLLECT_H
