/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE DerivOperator
//
//  Apply derivative operators on vector and scalar fields
//
//  Initial version: 2001-09-09 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _DERIV_OPERATORS_H_
#define _DERIV_OPERATORS_H_

#include <api/coSimpleModule.h>
using namespace covise;

class DerivOperators : public coSimpleModule
{
public:
    DerivOperators(int argc, char *argv[]);
    virtual ~DerivOperators();

private:
    virtual int compute(const char *port);
    virtual void copyAttributesToOutObj(coInputPort **, coOutputPort **, int);
    static const char *operators[];
    coInputPort *p_grid_;
    coInputPort *p_inData_;
    coOutputPort *p_outData_;
    void outputDummy();
    int gradient(int no_el, int no_vl, int no_points,
                 const int *el, const int *vl, const int *tl,
                 const float *xc, const float *yc, const float *zc,
                 float *sdata);
    int gradientMagnitude(int no_el, int no_vl, int no_points,
                          const int *el, const int *vl, const int *tl,
                          const float *xc, const float *yc, const float *zc,
                          float *sdata);
    int divergence(int no_el, int no_vl, int no_points,
                   const int *el, const int *vl, const int *tl,
                   const float *xc, const float *yc, const float *zc,
                   float *vdata[3]);
    int curl(int no_el, int no_vl, int no_points,
             const int *el, const int *vl, const int *tl,
             const float *xc, const float *yc, const float *zc,
             float *vdata[3]);
    coChoiceParam *p_whatToDo_;
    //   coBooleanParam *p_perCell_;
};
#endif
