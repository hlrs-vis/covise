/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ENLARGE_H
#define _ENLARGE_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
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

#include <api/coModule.h>
using namespace covise;

class SelectIdx : public coModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute();

    ////////// parameters
    coStringParam *p_selection;

    ////////// ports
    enum
    {
        NUM_PORTS = 8
    };
    coInputPort *p_in[NUM_PORTS];
    coOutputPort *p_out[NUM_PORTS];
    coInputPort *p_index;

    ////////// functions
    int getIndexField(int &numSelectElem, char *&selArr);

    coDistributedObject *selectObj(coDistributedObject *obj,
                                   const char *outName,
                                   int numElem, int numSelected,
                                   const char *selArr);

public:
    SelectIdx();
};
#endif
