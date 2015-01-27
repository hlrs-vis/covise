/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covWriteFiles.h"
#include <stdio.h>

int main()
{
    int pl[] = { 0, 4 };
    int cl[] = { 0, 2, 3, 1, 5, 7, 6, 4 };
    float x[] = { -1., -1., -1., -1., 1., 1., 1., 1. };
    float y[] = { -1., -1., 1., 1., -1., -1., 1., 1. };
    float z[] = { -1., 1., -1., 1., -1., 1., -1., 1. };
    float normalsx[] = { 1., 1., 1., 1., -1., -1., -1., -1. };
    float normalsy[] = { 1., 1., -1., -1., 1., 1., -1., -1. };
    float normalsz[] = { 1., -1., 1., -1., 1., -1., 1., -1. };
    int colors[] = { 65535, -65281, -671078401, -16318209, 65535, -3473153, -16318209, -65281 };
    char *ANames[] = { "vertexOrder", "Session", NULL };
    char *AValues[] = { "2", "FileLibTest" };

    int outD;

    outD = covOpenOutFile("polygon.covise");

    covWriteGeometryBegin(outD, 1, 1, 0);

    covWriteSetBegin(outD, 1);
    covWritePOLYGN(outD, 2, pl, 8, cl, 8, x, y, z, ANames, AValues, COUNT_ATTR);

    /* attributes of the set */
    covWriteSetEnd(outD, NULL, NULL, COUNT_ATTR);

    covWriteSetBegin(outD, 1);
    /* colors of geometry */
    covWriteRGBADT(outD, 8, colors, NULL, NULL, COUNT_ATTR);

    /* attributes of the set */
    covWriteSetEnd(outD, NULL, NULL, COUNT_ATTR);

    covWriteSetBegin(outD, 1);

    /* normals of geometry */
    covWriteUSTVDT(outD, 8, normalsx, normalsy, normalsz, NULL, NULL, COUNT_ATTR);

    /* attributes of the set */
    covWriteSetEnd(outD, NULL, NULL, COUNT_ATTR);

    /* attributes of the geometry container */
    covWriteGeometryEnd(outD, NULL, NULL, COUNT_ATTR);

    covCloseOutFile(outD);
}
