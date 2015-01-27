/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2009 HLRS ++
// ++                                                                     ++
// ++ Description: Program for writing iso-surfaces in ABAQUS .inp format ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Ralf Schneider                           ++
// ++               High Performance Computing Center Stuttgart           ++
// ++                           Nobelstrasse 19                           ++
// ++                           70569 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  25.02.2009  V1.0                                             ++
// ++**********************************************************************/

// COVISE data types
#include <stdio.h>
#include <do/coDoUnstructuredGrid.h>
// this includes our own class's headers
#include "WriteAbaqusInp.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
WriteAbaqusInp::WriteAbaqusInp(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Write Isosurface in ABAQUS Input Format")
{

    // Parameters
    p_outFile = addFileBrowserParam("outFile", "File for ABAQUS output");

    p_outFile->setValue("~/OutFile.inp", "*.inp");

    // Ports
    p_inPort = addInputPort("inPort", "Polygons", "Iso Surface input");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int WriteAbaqusInp::compute(const char *port)
{
    (void)port;

    int ii;

    coDistributedObject *obj = p_inPort->getCurrentObject();

    // we should have an object
    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort->getName());
        return FAIL;
    }

    // it should be the correct type
    if (!obj->isType("POLYGN"))
    {
        sendError("Received illegal type at port '%s'", p_inPort->getName());
        return FAIL;
    }

    // So, this is a Polygon List
    coDoPolygons *inIsoSurf = (coDoPolygons *)obj;

    // retrieve the size and the list pointers
    int numPoints = inIsoSurf->getNumPoints();
    int numVertices = inIsoSurf->getNumVertices();
    int numPolygons = inIsoSurf->getNumPolygons();

    sendInfo("NumPoints= %d -- NumVerices= %d -- NumPolygons= %d", numPoints, numVertices, numPolygons);

    int *CornerList, *PolygonList;
    float *XCoord, *YCoord, *ZCoord;

    inIsoSurf->getAddresses(&XCoord, &YCoord, &ZCoord, &CornerList, &PolygonList);

    FILE *fd = fopen(p_outFile->getValue(), "w");
    if (fd != NULL)
    {

        fprintf(fd, "**\n");
        fprintf(fd, "**Written with covise\n");
        fprintf(fd, "**\n");

        // Printout node list ***************************************************
        sendInfo("Writing node list ....");
        fprintf(fd, "*NODE\n");

        for (ii = 0; ii < numPoints; ii++)
        {
            fprintf(fd, "%d, %f, %f, %f\n", ii + 1, XCoord[ii], YCoord[ii], ZCoord[ii]);
        }

        // Printout element list ************************************************
        sendInfo("Writing element list ....");
        fprintf(fd, "*ELEMENT, TYPE=S3R, ELSET=Covise_Iso_Surface\n");

        for (ii = 0; ii < numPolygons; ii++)
        {
            if (!((CornerList[PolygonList[ii]] == CornerList[PolygonList[ii] + 1]) || (CornerList[PolygonList[ii] + 1] == CornerList[PolygonList[ii] + 2]) || (CornerList[PolygonList[ii]] == CornerList[PolygonList[ii] + 2])))
            {

                fprintf(fd, "%d, %d, %d, %d\n", (PolygonList[ii] + 1), (CornerList[PolygonList[ii]] + 1),
                        (CornerList[PolygonList[ii] + 1] + 1), (CornerList[PolygonList[ii] + 2]) + 1);
            }
        }

        fprintf(fd, "**\n");
        fprintf(fd, "*SHELL SECTION, ELSET=Covise_Iso_Surface, MATERIAL=MAT_Iso_surface\n");
        fprintf(fd, "0.001\n");
        fprintf(fd, "**\n");
        fprintf(fd, "*MATERIAL, NAME=MAT_Iso_surface\n");
        fprintf(fd, "*ELASTIC, TYPE=ISOTROPIC\n");
        fprintf(fd, "14000.,  0.3\n");
        fprintf(fd, "**\n");

        fclose(fd);

        sendInfo("... done writing ABAQUS input.");
    }
    else
    {
        sendError("Couldn't open file: %s", p_outFile->getValue());
    }

    return SUCCESS;
}

MODULE_MAIN(IO, WriteAbaqusInp)
