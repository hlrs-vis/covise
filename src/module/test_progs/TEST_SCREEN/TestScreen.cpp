/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TestScreen.h"
#include <assert.h>

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
TestScreen::TestScreen(const TestScreen &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
TestScreen &TestScreen::operator=(const TestScreen &)
{
    assert(0);
    return *this;
}

TestScreen::TestScreen()
    : coModule("USG index analysis")
{
    // Parameters

    // create the parameters
    p_numX = addIntSliderParam("numX", "Number of lines in X direction");
    p_numX->setValue(1, 40, 20);

    p_numY = addIntSliderParam("numY", "Number of lines in Y direction");
    p_numY->setValue(1, 40, 15);

    p_numZ = addIntSliderParam("numZ", "Number of lines in Z direction");
    p_numZ->setValue(1, 40, 1);

    // Ports
    p_outPort = addOutputPort("outPort", "coDoLines", "Data output int");
}

int TestScreen::compute()
{
    // read the parameters
    int numX = p_numX->getValue();
    int numY = p_numY->getValue();
    int numZ = p_numZ->getValue();
    cerr << numX << "," << numY << "," << numZ << endl;

    int numLines = 0;
    if (numZ > 1)
        numLines += numX * numY;
    if (numY > 1)
        numLines += numX * numZ;
    if (numX > 1)
        numLines += numY * numZ;

    int numPoints = numX * numY * numZ;

    coDoLines *lines = new coDoLines(p_outPort->getObjName(),
                                     numPoints, 2 * numLines, numLines);

    cerr << "alloc " << numLines << " lines" << endl;
    float *x, *y, *z;
    int *conn, *elem;
    lines->getAddresses(&x, &y, &z, &conn, &elem);

    // Points
    int ix, iy, iz;
    for (iz = 0; iz < numZ; iz++)
        for (iy = 0; iy < numY; iy++)
            for (ix = 0; ix < numX; ix++)
            {
                *x = ix;
                x++;
                *y = iy;
                y++;
                *z = iz;
                z++;
            }

    // line counter
    int connNo = 0;

// Lines in Z-Direction

#define IDX(x, y, z) ((x) + numX * (y) + numX * numY * (z))

    if (numZ > 1)
        for (ix = 0; ix < numX; ix++)
            for (iy = 0; iy < numY; iy++)
            {
                *elem++ = connNo;
                connNo += 2;

                *conn = IDX(ix, iy, 0);
                conn++;
                *conn = IDX(ix, iy, numZ - 1);
                conn++;
            }

    if (numY > 1)
        for (ix = 0; ix < numX; ix++)
            for (iz = 0; iz < numZ; iz++)
            {
                *elem++ = connNo;
                connNo += 2;

                *conn = IDX(ix, 0, iz);
                conn++;
                *conn = IDX(ix, numY - 1, iz);
                conn++;
            }

    if (numX > 1)
        for (iy = 0; iy < numY; iy++)
            for (iz = 0; iz < numZ; iz++)
            {
                *elem++ = connNo;
                connNo += 2;

                *conn = IDX(0, iy, iz);
                conn++;
                *conn = IDX(numX - 1, iy, iz);
                conn++;
            }

    cerr << "uesd  " << connNo / 2 << " lines" << endl;

#undef IDX
    p_outPort->setCurrentObject(lines);
    return CONTINUE_PIPELINE;
}

int main(int argc, char *argv[])

{
    // create the module
    TestScreen *application = new TestScreen;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
