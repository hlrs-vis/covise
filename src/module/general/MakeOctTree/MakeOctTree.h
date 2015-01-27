/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE MakeOctTree
//
//  Make OctTrees for UNSGRD.
//  Convenient when the pipeline has more than 1 Tracer
//
//  Initial version:   28.06.2002 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _MAKE_OCTTREE_H_
#define _MAKE_OCTTREE_H_

#include <api/coSimpleModule.h>
using namespace covise;

class MakeOctTree : public coSimpleModule
{
public:
    MakeOctTree(int argc, char *argv[]);
    virtual ~MakeOctTree();

protected:
    virtual int compute(const char *port);

private:
    coInputPort *p_grids_;
    coOutputPort *p_octtrees_;
    coIntScalarParam *p_normal_size_;
    coIntScalarParam *p_max_no_levels_;
    coIntScalarParam *p_min_small_enough_;
    coIntScalarParam *p_crit_level_;
    coIntScalarParam *p_limit_fX_;
    coIntScalarParam *p_limit_fY_;
    coIntScalarParam *p_limit_fZ_;
};
#endif
