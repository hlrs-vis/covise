/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _REDUCE_H
#define _REDUCE_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Filter program in COVISE API                           ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Christof Schwenzer                       ++
// ++                             Vircinity GmbH                          ++
// ++                              Nobelstraße 15                         ++
// ++                             70569 Stuttgart                         ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReduceSet : public coModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    ////////// parameters
    coIntSliderParam *pFactor_;
    ////////// ports
    coInputPort **pInPorts_;
    coOutputPort **pOutPorts_;

public:
    ReduceSet(int argc, char *argv[]);
};
#endif
