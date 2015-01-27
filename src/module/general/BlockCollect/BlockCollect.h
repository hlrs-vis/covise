/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                               (C) 2001 VirCinity GmbH  **
 ** Description: BlockCollect                                              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Lars Frenzel                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date: 03.04.2001 Sven Kufer: changed to new API                        **
\**************************************************************************/

#ifndef _BLOCK_COLLECT_H
#define _BLOCK_COLLECT_H

// includes
#include <api/coModule.h>
#include <util/coviseCompat.h>

// defines
#define NUM_PORTS 15

class BlockCollect : public covise::coModule
{

private:
    //ports
    covise::coInputPort *p_inport[NUM_PORTS];
    covise::coOutputPort *p_outport;

    //params
    covise::coChoiceParam *p_mode;

    enum CollectMode
    {
        SetOfElements = 0x00, // Just creates a set including all input objects. Not sure whether the input objects may be sets.
        CatTimesteps = 0x01, // Please explain !!!
        MergeBlocks = 0x02, // Please explain !!!
        SetOfTimesteps = 0x03, // Creates a set including all input objects and adds the TIMESTEP attribute. Not sure whether the input objects may be sets.
        SetOfBlocks = 0x04, // Please explain !!!
        CatBlocks = 0x05 // Creates one set including all input objects but removes any set hierarchy (up to two levels): (a1,a2),(b1,(c1,c2)),d1 -> (a1,a2,b1,c1,c2,d1)
    } mode;

    // functions

    int NoTimeSteps(); // -1 if we mix static data and data sets
    // 0 if we have only static data
    // number of time steps for output
    //       with transient data

    int aktivePorts;

public:
    virtual int compute(const char *port);
    BlockCollect(int argc, char *argv[]);
};
#endif // _BLOCK_COLLECT_H
