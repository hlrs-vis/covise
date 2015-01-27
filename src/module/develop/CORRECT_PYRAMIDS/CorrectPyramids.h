/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE CORRECT_PYRAMIDS
//
//  Correct pyramids with the appearance of tetrahedra
//
//  Initial version: 2003-10-08 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _CORRECT_PYRAMIDS_H_
#define _CORRECT_PYRAMIDS_H_

#include "coSimpleModule.h"

class CorrectPyramids : public coSimpleModule
{
public:
    CorrectPyramids();
    virtual ~CorrectPyramids();

private:
    virtual int compute();
    int PyramidProblem(int element, const int *el, const int *vl,
                       const float *xc, const float *yc, const float *zc);
    static int NumberOfVertices(int);

    coInputPort *_p_in_grid;
    coOutputPort *_p_out_grid;
    coFloatParam *_p_volume;
};
#endif
