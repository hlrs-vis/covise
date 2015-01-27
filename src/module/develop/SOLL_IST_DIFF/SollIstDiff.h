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
#ifndef SollIstDiff_H
#define SollIstDiff_H

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/CoNewString.h>

typedef struct
{
public:
    float x;
    float y;
    float z;
} Vector;

typedef struct
{
    Vector coord;
    Vector disp;
    float total;
} dinfo;

class SollIstDiff : public coSimpleModule
{
public:
    SollIstDiff();

    ~SollIstDiff();

private:
    /// compute call-back
    virtual int compute();

    int Bsearch(int begin, int end, float xkey, int *pos);
    int bestFit(Vector *p, int pos, int numPoints);

    /// ports
    coInputPort *p_polyIn;
    coOutputPort *p_polyOut, *p_diffOut, *p_totalOut, *p_pointsOut, *p_pointTotalOut;

    /// parameters
    coFloatParam *p_tol;
    coFileBrowserParam *p_file;
    coBooleanParam *p_onlyGeo;

    dinfo *data;
};
#endif
