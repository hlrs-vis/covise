/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: extract element of a set                               ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 04.10.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef GETSETELEM_H
#define GETSETELEM_H

#include <do/coDoSet.h>
#include <api/coModule.h>
using namespace covise;
#include <util/coRestraint.h>

class GetSetElem : public coModule
{
public:
    GetSetElem(int argc, char *argv[]);

    virtual ~GetSetElem();

    static const int maxDataPortNm_ = 8;

private:
    /// compute call-back
    virtual int compute(const char *port);

    // adjust the given index - returns the proper index
    int adjustIndex(const int &idx, const coDoSet *obj);

    /// ports
    coInputPort **pInPorts_;
    coOutputPort **pOutPorts_;

/// parameters
#ifdef GET_SETELEM
    coIntScalarParam *pIndex_;
#else
    coStringParam *pSelection_;
#endif
};
#endif
