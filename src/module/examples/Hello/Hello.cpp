/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "Hello.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Hello::Hello(int argc, char *argv[])
    : coModule(argc, argv, "Hello, world! program")
{
    // no parameters, no ports...
    grid = addOutputPort("mesh", "UnstructuredGrid", "unstructured grid");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Hello::compute(const char *port)
{
    (void)port;
    sendInfo("Hello, Covise!");

    coDoUnstructuredGrid *g = new coDoUnstructuredGrid(grid->getObjName(), 2, 14, 10, 1);
    int *el, *cl, *tl;
    float *x, *y, *z;
    g->getAddresses(&el, &cl, &x, &y, &z);
    g->getTypeList(&tl);
    tl[0] = 7;
    tl[1] = 6;

    el[0] = 0;
    el[1] = 8;

    cl[0] = 0;
    cl[1] = 1;
    cl[2] = 2;
    cl[3] = 3;
    cl[4] = 4;
    cl[5] = 5;
    cl[6] = 6;
    cl[7] = 7;

    cl[8] = 1;
    cl[9] = 8;
    cl[10] = 2;
    cl[11] = 5;
    cl[12] = 9;
    cl[13] = 6;

    x[0] = 0.0f;
    y[0] = 0.0f;
    z[0] = 0.0f;
    x[1] = 1.0f;
    y[1] = 0.0f;
    z[1] = 0.0f;
    x[2] = 1.0f;
    y[2] = 1.0f;
    z[2] = 0.0f;
    x[3] = 0.0f;
    y[3] = 1.0f;
    z[3] = 0.0f;
    x[4] = 0.0f;
    y[4] = 0.0f;
    z[4] = 1.0f;
    x[5] = 1.0f;
    y[5] = 0.0f;
    z[5] = 1.0f;
    x[6] = 1.0f;
    y[6] = 1.0f;
    z[6] = 1.0f;
    x[7] = 0.0f;
    y[7] = 1.0f;
    z[7] = 1.0f;

    x[8] = 1.5f;
    y[8] = 0.7f;
    z[8] = 0.0f;
    x[9] = 1.5f;
    y[9] = 0.7f;
    z[9] = 1.0f;

    grid->setCurrentObject(g);

    return SUCCESS;
}

MODULE_MAIN(Examples, Hello)
