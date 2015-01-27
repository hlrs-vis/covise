/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: Module to create a displacement field                  ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 12.02.01                                                      ++
// ++**********************************************************************/

#ifndef _CREATE_DISP_H
#define _CREATE_DISP_H

#include <api/coModule.h>
using namespace covise;

class CreateDisp : public coModule
{
public:
    CreateDisp(int argc, char **argv);

private:
    /// compute call-back
    virtual int compute(const char *);

    /// does the computation and returns a newly created object for output

    /// in port USG
    coInputPort *p_GridInPort_;

    coOutputPort *p_vDataOutPort_;

    /// parameters
    coFloatParam *pTrace1_;
    coFloatParam *pTrace2_;
    coFloatParam *pTrace3_;

    coFloatParam *pTrace4_;
    coFloatParam *pTrace5_;
    coFloatParam *pTrace6_;

    coFloatVectorParam *pRigDisp_;
};
#endif // CreateDisp
