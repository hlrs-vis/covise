/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Field To Lines
// Filip Sadlo 2007
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "FieldToLines.h"

#include "linalg.h"

#include "unifield.h"
#include "unigeom.h"
#include "unisys.h"

#include "field_to_lines_impl.cpp" // ### including .cpp
using namespace covise;

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

    // wrapper for input field
    UniField *unif = new UniField((coDoStructuredGrid *)lineFieldGrid->getCurrentObject(),
                                  NULL,
                                  (coDoVec2 *)lineFieldData->getCurrentObject());

    // geometry wrapper for output
    UniGeom ugeom = UniGeom(lines);

    // compute
    if (!field_to_lines_impl(&us,
                             unif,
                             nodesX->getValue(),
                             nodesY->getValue(),
                             nodesZ->getValue(),
                             stride.getValue(),
                             &ugeom,
                             NULL,
                             NULL))
    {
        delete unif;
        return FAIL;
    }

    // output data already assigned to ports

    // delete field wrapper (but not the field)
    delete unif;

    return SUCCESS;
}
