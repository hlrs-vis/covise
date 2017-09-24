/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)2015 HLRS **
 **                                                                        **
 ** Description: Simple Reader CSV files storing polygon information       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 ** History:                                                               **
 ** December 15         v1                                                 **
 **                                                                        **
\**************************************************************************/

//lenght of a line
#define LINE_SIZE 8192

// portion for resizing data
#define CHUNK_SIZE 4096

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadCSVPoly.h"
#include <do/coDoPolygons.h>

ReadCSVPoly::ReadCSVPoly(int argc, char *argv[])
    : coModule(argc, argv, "Read polygons stored as CSV")
{
    // the output port
    polygonPort = addOutputPort("polygons", "Polygons", "geometry polygons");

    // select the CSV file name with a file browser
    coordFileParam = addFileBrowserParam("coordFile", "Coord file");
    coordFileParam->setValue("data", "*.csv");
    facesFileParam = addFileBrowserParam("facesFile", "Faces file");
    facesFileParam->setValue("data", "*.csv");
}

ReadCSVPoly::~ReadCSVPoly()
{
}

void ReadCSVPoly::quit()
{
}

int ReadCSVPoly::compute(const char *port)
{
    (void)port;

    // get the file name
    coordFilename = coordFileParam->getValue();
    facesFilename = facesFileParam->getValue();

    if (coordFilename != NULL && facesFilename!=NULL)
    {
        // open the file
        if (openFile())
        {
            sendInfo("File %s %s open", coordFilename, facesFilename);

            // read the file, create the lists and create a COVISE polygon object
            readFiles();
        }
        else
        {
            sendError("Error opening file %s %s", coordFilename, facesFilename);
            return FAIL;
        }
    }
    else
    {
        sendError("ERROR: fileName is NULL");
        return FAIL;
    }
    return SUCCESS;
}

void
ReadCSVPoly::readFiles()
{
    int numCoords = 0, numVertices = 0, numPolys = 0;

    float *cx, *cy, *cz; // vertex coordinate lists
    float *cxTmp, *cyTmp, *czTmp; // temp. variables for resizing the lists
    int newCSize, oldCSize = CHUNK_SIZE; // size of the vertex lists
    static int numCAlloc = 1; // number of 'allocs' for the coordinate lists

    int *ci; // vertex coordinate index list
    int *ciTmp;
    int newCiSize, oldCiSize = CHUNK_SIZE;
    static int numCiAlloc = 1; // number of 'allocs' for the coordinate index list

    int *pi; // polygon list
    int *piTmp;
    int newPiSize, oldPiSize = CHUNK_SIZE;
    static int numPiAlloc = 1;

    char line[LINE_SIZE]; // line in an obj file
    char *first; // current position in line
    coDoPolygons *polygonObject; // output object
    const char *polygonObjectName; // output object name assigned by the controller

    // allocate memory for the lists
    cx = new float[CHUNK_SIZE];
    cy = new float[CHUNK_SIZE];
    cz = new float[CHUNK_SIZE];
    ci = new int[CHUNK_SIZE];
    pi = new int[CHUNK_SIZE];

    while (fgets(line, LINE_SIZE, fp1) != NULL)
    {
        // find first non-space character
        first = line;
        while (*first != '\0' && isspace(*first))
            first++;

        // skip blank lines and comments
        if (*first == '\0' || *first == '#')
            // read the next line
                continue;

        //-> create the coordinate lists

        numCoords++;

        // test if we allocated enough memory
        if (numCoords > (oldCSize))
        {
            // allocate more memory
            numCAlloc++;
            newCSize = numCAlloc * CHUNK_SIZE;
            cxTmp = new float[newCSize];
            cyTmp = new float[newCSize];
            czTmp = new float[newCSize];
            memcpy(cxTmp, cx, oldCSize * sizeof(float));
            memcpy(cyTmp, cy, oldCSize * sizeof(float));
            memcpy(czTmp, cz, oldCSize * sizeof(float));
            delete[] cx;
            delete[] cy;
            delete[] cz;
            cx = cxTmp;
            cy = cyTmp;
            cz = czTmp;
            oldCSize = newCSize;
        }
        // scan the line
        if (sscanf(first, "%f,%f,%f", &(cx[numCoords - 1]), &(cy[numCoords - 1]), &(cz[numCoords - 1])) != 3)
        {
            cerr << "ReadCSVPoly::readFile:: sscanf2 failed" << endl;
        }
    }


    // read one line after another
    while (fgets(line, LINE_SIZE, fp2) != NULL)
    {
        // find first non-space character
        first = line;
        while (*first != '\0' && isspace(*first))
            first++;

        // skip blank lines and comments
        if (*first == '\0' || *first == '#')
            // read the next line
                continue;


        // -> create the coordinate index list and the polygon list

        numPolys++;

        // create the polygons lists
        if (numPolys > (oldPiSize))
        {
            numPiAlloc++;
            newPiSize = numPiAlloc * CHUNK_SIZE;
            piTmp = new int[newPiSize];
            memcpy(piTmp, pi, oldPiSize * sizeof(int));
            delete[] pi;
            pi = piTmp;
            oldPiSize = newPiSize;
        }
        pi[numPolys - 1] = numVertices;

        // scan the line string for string "ci/ti/ni"
        int numv=3;
        int v[4];
        numv = sscanf(first, "%d,%d,%d,%d", v,v+1,v+2,v+3);
        if(numv>0 && numv < 5)
        {
       
            // test if we have enough memory
            if (numVertices+numv > (oldCiSize))
            {
                // allocate more memory
                numCiAlloc++;
                newCiSize = numCiAlloc * CHUNK_SIZE;
                ciTmp = new int[newCiSize];
                memcpy(ciTmp, ci, oldCiSize * sizeof(int));
                delete[] ci;
                ci = ciTmp;
                oldCiSize = newCiSize;
            }
            for(int i=0;i<numv;i++)
            {
                // obj indices start with 1, COVISE indices start with 0
                ci[numVertices +i] = v[i] - 1;
            }
            numVertices+=numv;
        }
    }

    sendInfo("found %d coordinates, %d vertices, %d polygons", numCoords, numVertices, numPolys);

    // get the COVISE output object name from the controller
    polygonObjectName = polygonPort->getObjName();

    // create the COVISE output object
    polygonObject = new coDoPolygons(polygonObjectName, numCoords, cx, cy, cz, numVertices, ci, numPolys, pi);
    polygonPort->setCurrentObject(polygonObject);

    // set the vertex order for twosided lighting in the renderer
    // 1=clockwise 2=counterclockwise
    // missing vertex order -> no twosided lighting (inner surface not lighted)
    // wrong vertex order -> wrong lighting for surfaces with normals
    polygonObject->addAttribute("vertexOrder", "2");

    // delete the lists
    delete[] cx;
    delete[] cy;
    delete[] cz;
    delete[] ci;
    delete[] pi;
    fclose(fp1);
    fclose(fp2);
}

bool ReadCSVPoly::openFile()
{
    sendInfo("Opening file %s", coordFilename);

    // open the obj file
    if ((fp1 = Covise::fopen((char *)coordFilename, "r")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", coordFilename);
        return false;
    }
    else
    {
    sendInfo("Opening file %s", facesFilename);
        if ((fp2 = Covise::fopen((char *)facesFilename, "r")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", facesFilename);
        return false;
    }
    else
    {
        return true;
    }
    }
}

MODULE_MAIN(Examples, ReadCSVPoly)
