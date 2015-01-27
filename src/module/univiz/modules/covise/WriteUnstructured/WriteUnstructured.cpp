/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// WriteUnstructured
// Filip Sadlo 2007
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "WriteUnstructured.h"

#include "linalg.h"

#include "unstructured.h"
#include "unisys.h"

#include "write_unstructured_impl.cpp" // ### including .cpp

static Unstructured *unst_in = NULL;

UniSys us = UniSys(NULL);

static bool sequenceNameLast;
static int sequenceId;

int main(int argc, char *argv[])
{
    myModule *application = new myModule(argc, argv);
    application->start(argc, argv);
    return 0;
}

void myModule::postInst()
{
    sequenceNameLast = sequenceName->getValue();
    sequenceId = 0;
}

void myModule::param(const char *, bool)
{
    // force min/max
    adaptMyParams();

    if (sequenceName->getValue() != sequenceNameLast)
    {
        sequenceId = 0;
        sequenceNameLast = sequenceName->getValue();
    }
}

int myModule::compute(const char *)
{
    // force min/max
    adaptMyParams();

    // system wrapper
    us = UniSys(this);

    // create unstructured wrapper for input
    int compScalar = -1;
    int compVector0 = -1;
    int compVector1 = -1;
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
        if (vector0->getCurrentObject())
        {
            vvec.push_back((coDoVec3 *)(vector0->getCurrentObject()));
            compVector0 = compScalar + 1;
        }
        if (vector1->getCurrentObject())
        {
            vvec.push_back((coDoVec3 *)(vector1->getCurrentObject()));
            compVector1 = (compScalar > compVector0 ? compScalar + 1 : compVector0 + 1);
        }

        if (svec.size() > 0 && vvec.size() == 0)
        {
            unst_in = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                       &svec, NULL);
        }
        else if (svec.size() == 0 && vvec.size() > 0)
        {
            unst_in = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                       NULL, &vvec);
        }
        else if (svec.size() > 0 && vvec.size() > 0)
        {
            unst_in = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                       &svec, &vvec);
        }
    }

    if (compScalar < 0 && compVector0 < 0 && compVector1 < 0)
    {
        return FAIL;
    }

    // compute
    if (!sequenceName->getValue())
    {
        write_unstructured_impl(&us, unst_in, fileName->getValue());
    }
    else
    {
        char fname[1024];
        sprintf(fname, "%s%d", fileName->getValue(), sequenceId);
        write_unstructured_impl(&us, unst_in, fname);
        sequenceId++;
    }

    return SUCCESS;
}
