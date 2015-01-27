/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "Tetrahedronize.h"
#include "Grid.h"

//////
////// we must provide main to init covise
//////

int main(int argc, char *argv[])
{
    // init
    new Tetrahedronize(argc, argv);

    // and back to covise
    Covise::main_loop();

    // done
    return 1;
}

coDistributedObject **Tetrahedronize::compute(const coDistributedObject **in, char **outNames)
{
    coDistributedObject **returnObject = NULL;
    Grid *gridIn, *gridOut;
    int regulate;

    // get input
    gridIn = new Grid(in[0], in[1]);
    Covise::get_boolean_param("regulate", &regulate);
    if (regulate)
    {
        Covise::sendInfo("regulation of irregular grids not yet available.");
        regulate = 0;
    }

    // prepare output
    gridOut = new Grid(outNames[0], outNames[1]);

    // compute output
    returnObject = gridOut->tetrahedronize(gridIn, regulate);

    // done
    return (returnObject);
}
