/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "testingModule.h"

#include <do/coDoSet.h>
#include <do/coDoText.h>

using namespace covise;

testingModule::testingModule(int argc, char *argv[])
    : coModule(argc, argv, "testingModule")
{
}

int testingModule::compute(char const *str)
{

    coDistributedObject **cdo = new coDistributedObject *[101];

    for (int ii = 0; ii < 100; ii++)
    {
        char objname[100];
        sprintf(objname, "info_%d", ii);
        cdo[ii] = new coDoText(objname, "name");
    }
    cdo[100] = NULL;
    coDoSet set("set", cdo);

    for (int ii = 0; ii < 99; ii++)
    {
        delete cdo[ii];
    }
    delete[] cdo;

    for (int jj = 0; jj < 10000; jj++)
    {
        for (int ii = 0; ii < set.getNumElements(); ii++)
        {
            coDistributedObject const *getObj = set.getElement(ii);
            delete getObj; // das gehört dann hier wohl hin
        }
    }
    return CONTINUE_PIPELINE;
}

testingModule::~testingModule()
{
}

MODULE_MAIN(designTool, testingModule)
