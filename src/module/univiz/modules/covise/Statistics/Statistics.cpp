/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Statistics
// Filip Sadlo 2007
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "Statistics.h"

#include "linalg.h"

#include "unstructured.h"
#include "unisys.h"

#include "statistics_impl.cpp" // ### including .cpp

static Unstructured *unst_in = NULL;

UniSys us = UniSys(NULL);

int main(int argc, char *argv[])
{
    myModule *application = new myModule(argc, argv);
    application->start(argc, argv);
    return 0;
}

void myModule::postInst() {}

void myModule::param(const char *, bool)
{
    // force min/max
    adaptMyParams();
}

int myModule::compute(const char *)
{
    // force min/max
    adaptMyParams();

    // system wrapper
    us = UniSys(this);

    // create unstructured wrapper for input
    int compScalar = -1;
    int compVector = -1;
    if (us.inputChanged("ucd", 0))
    {
        if (unst_in)
            delete unst_in;
        std::vector<coDoFloat *> svec;
        if (scalar->getCurrentObject())
        {
            svec.push_back((coDoFloat *)(scalar->getCurrentObject()));
            compScalar = 0;
        }
        std::vector<coDoVec3 *> vvec;
        if (vector->getCurrentObject())
        {
            vvec.push_back((coDoVec3 *)(vector->getCurrentObject()));
            compVector = compScalar + 1;
        }

        if (scalar->getCurrentObject())
        {
            unst_in = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                       &svec, NULL);
        }
        else
        {
            unst_in = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                       NULL, &vvec);
        }
    }

    if (compScalar < 0 && compVector < 0)
    {
        return FAIL;
    }

    if (compScalar >= 0 && vector->getCurrentObject())
    {
        us.warning("scalar and vector input available, computing statistics only for scalar");
    }

    // compute
    statistics_impl(&us, unst_in, (compScalar >= 0 ? compScalar : compVector));

    return SUCCESS;
}
