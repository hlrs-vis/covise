/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "StarCD.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

coModule *StarCD::module = 0;

/// ----- Never forget the Destructor !! -------

StarCD::~StarCD()
{
}

// ----------------------------------------------------------------

// The constructor we use
StarCD::StarCD(int argc, char *argv[])
    : coSimLib(argc, argv, argv[0], "Star-CD coupling module")
{

    StarCD::module = this;

    // no automatic creation of ports - we do it later ourselves
    autoInitParam(0);

    // create the parameters
    createParam();

    // preset variables
    d_user = d_host = d_meshDir = d_compDir = d_case = d_script = d_creator = d_usr[0] = d_usr[1] = d_usr[2] = d_usr[3] = d_usr[4] = d_usr[5] = d_usr[6] = d_usr[7] = d_usr[8] = d_usr[9] = NULL;

    d_mesh = NULL;

    d_simState = NOT_STARTED;
    d_setupFileName = NULL;
    d_setupObjName = NULL;
    d_commObjName = NULL;

    d_useOldConfig = 0;

    // empty string if no args come in -> always have content for %s argument
    d_commArg = strcpy(new char[1], "");
}

// ----------------------------------------------------------------

int main(int argc, char *argv[])
{
    StarCD *application = new StarCD(argc, argv);
    application->start(argc, argv);
    return 0;
}
