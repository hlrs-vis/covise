/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                                     ++
// ++ Description: Extrude lines to quads                                 ++
// ++                                                                     ++
// ++**********************************************************************/

// this includes our own class's headers
#include "Extrude.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Extrude::Extrude(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Extrude")
{
    p_extrudeDirection = addFloatVectorParam("extrude_vector", "extrude direction");
    p_extrudeDirection->setValue(0, 0, 1);

    p_extrudeLength = addFloatParam("extrude_legth", "extrude length");
    p_extrudeLength->setValue(1.0);

    p_inLines = addInputPort("linesOut", "Lines", "output lines");
    p_outPolys = addOutputPort("polysOut", "Polygons", "output polygons");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Extrude::compute(const char *port)
{
    (void)port;

    coDistributedObject *obj = p_inLines->getCurrentObject();
    coDoLines *inLines = dynamic_cast<coDoLines *>(obj);

    float *lx, *ly, *lz;
    int *lcornerlist, *llinelist;
    int numLines = inLines->getNumLines();
    int numPoints = inLines->getNumPoints();
    int numVertices = inLines->getNumVertices();
    inLines->getAddresses(&lx, &ly, &lz, &lcornerlist, &llinelist);

    fprintf(stderr, "numLines=%d\nnumPoints=%d\nnumVertices=%d\n", numLines, numPoints, numVertices);
    int numLineSegments = numVertices - numLines;

    float *px, *py, *pz;
    int *pcornerlist, *plinelist;
    coDoPolygons *outPolygons = new coDoPolygons(p_outPolys->getObjName(), 2 * numPoints, 4 * numLineSegments, numLineSegments);
    outPolygons->getAddresses(&px, &py, &pz, &pcornerlist, &plinelist);

    float vec[3];
    vec[0] = p_extrudeDirection->getValue(0);
    vec[1] = p_extrudeDirection->getValue(1);
    vec[2] = p_extrudeDirection->getValue(2);

    for (int i = 0; i < numPoints; i++)
    {
        px[i] = lx[i];
        py[i] = ly[i];
        pz[i] = lz[i];

        px[i + numPoints] = lx[i] + vec[0];
        py[i + numPoints] = ly[i] + vec[1];
        pz[i + numPoints] = lz[i] + vec[2];
    }

    int segments;
    int pos = 0;

    for (int i = 0; i < numLines; i++)
    {
        if (i < numLines - 1)
        {
            // 2 vertices - 1 line segment
            // n vertices - n-1 line segment
            segments = llinelist[i + 1] - llinelist[i] - 1;
        }
        else
        {
            // n_vertices of last line = length of list - last entry
            segments = numVertices - llinelist[i] - 1;
        }

        fprintf(stderr, "segments=%d\n", segments);
        for (int j = 0; j < segments; j++)
        {
            //fprintf(stderr,"seg %d: %d %d\n",j,lcornerlist[llinelist[i]+j],lcornerlist[llinelist[i]+j+1]);

            // cornerlist: nodes of segment, translated nodes
            pcornerlist[4 * pos] = lcornerlist[llinelist[i] + j];
            pcornerlist[4 * pos + 1] = lcornerlist[llinelist[i] + j + 1];
            pcornerlist[4 * pos + 2] = lcornerlist[llinelist[i] + j + 1] + numPoints;
            pcornerlist[4 * pos + 3] = lcornerlist[llinelist[i] + j] + numPoints;

            plinelist[pos] = 4 * pos;

            pos++;
        }
    }

    p_outPolys->setCurrentObject(outPolygons);

    return SUCCESS;
}

MODULE_MAIN(Examples, Extrude)
