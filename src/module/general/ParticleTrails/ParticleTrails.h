/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2015 HLRS       ++
// ++ Description: connect particle positions through lines               ++
// ++                                                                     ++
// ++ Author:  Uwe Woessner                                               ++
// ++                                                                     ++
// ++               HLRS                                                  ++
// ++               Nobelstrasse 19                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 09.08.2015                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef ParticleTrails_H
#define ParticleTrails_H

#include <do/coDoSet.h>
#include <api/coModule.h>
using namespace covise;

class ParticleTrails : public coModule
{
public:
    ParticleTrails(int argc, char *argv[]);

    virtual ~ParticleTrails();

private:
    /// compute call-back
    virtual int compute(const char *port);

    coIntScalarParam *pTrailLength;

    /// ports
    coInputPort *pPoints;
    coOutputPort *pLines;

};
#endif
