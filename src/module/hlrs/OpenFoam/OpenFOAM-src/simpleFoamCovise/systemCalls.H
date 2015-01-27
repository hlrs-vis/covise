/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __systemCalls_H
#define __systemCalls_H

#include "cstdlib"

int callRemoveLogs()
{
    return system("rm -rf log*");
}

int callSetSet()
{
    return system(". $WM_PROJECT_DIR/bin/tools/RunFunctions; runApplication setSet -batch setBatch");
}

int callSetsToZones()
{
    return system(". $WM_PROJECT_DIR/bin/tools/RunFunctions; runApplication setsToZones -noFlipMap");
}

int callDecomposePar()
{
    return system(". $WM_PROJECT_DIR/bin/tools/RunFunctions; runApplication decomposePar -force");
}

#endif
