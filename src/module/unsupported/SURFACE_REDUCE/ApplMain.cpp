/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#define APPLMAIN
#include "SurfaceReduce.h"

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}
