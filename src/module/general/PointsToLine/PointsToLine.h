/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINTS_TO_LINE_H
#define POINTS_TO_LINE_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
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

class PointsToLine : public covise::coSimpleModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    covise::coInputPort *p_in;
    covise::coOutputPort *p_out;

public:
    PointsToLine(int argc, char *argv[]);
};
#endif
