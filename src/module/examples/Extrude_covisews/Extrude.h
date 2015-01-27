/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Extrude_H
#define _Extrude_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Extrude, world!" in COVISE API                          ++
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

#include <api/coSimpleModule.h>
using namespace covise;

class Extrude : public coSimpleModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    coFloatVectorParam *p_extrudeDirection;
    coFloatParam *p_extrudeLength;

    coInputPort *p_inLines;
    coOutputPort *p_outPolys;

public:
    Extrude(int argc, char *argv[]);
};
#endif
