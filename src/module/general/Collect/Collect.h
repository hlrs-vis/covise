/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COLLECT_H
#define _COLLECT_H

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

class Collect : public coModule
{
#ifdef VOLUME_COLLECT
    enum { CHAN_MAX = 8 };
#endif

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    ////////// ports
    coInputPort *p_grid;
#ifndef VOLUME_COLLECT
    coInputPort *p_color, *p_norm, *p_text, *p_vertex;
#else
    coInputPort *p_color[CHAN_MAX];
#endif
    coOutputPort *p_outPort;
#ifdef MATERIAL
    coMaterialParam *p_material;
#endif

    coStringParam *p_varName;
    coStringParam *p_attribute;
    coFloatVectorParam *p_boundsMinParam;
    coFloatVectorParam *p_boundsMaxParam;

public:
    Collect(int argc, char *argv[]);
};
#endif
