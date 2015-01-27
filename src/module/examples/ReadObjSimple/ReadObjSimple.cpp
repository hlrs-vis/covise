/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Simple Reader for Wavefront OBJ Format	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 ** History:                                                               **
 ** April 99         v1                                                    **
 ** September 99     new covise api                                        **                               **
 **                                                                        **
\**************************************************************************/

//lenght of a line
#define LINE_SIZE 8192

// portion for resizing data
#define CHUNK_SIZE 4096

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadObjSimple.h"
#include <do/coDoPolygons.h>

ReadObjSimple::ReadObjSimple(int argc, char *argv[])
    : coModule(argc, argv, "Simple Wavefront OBJ Reader")
{
    // the output port
    polygonPort = addOutputPort("polygons", "Polygons", "geometry polygons");

    // select the OBJ file name with a file browser
    objFileParam = addFileBrowserParam("objFile", "OBJ file");
    objFileParam->setValue("data/objFile", "*.obj");
}

ReadObjSimple::~ReadObjSimple()
{
}

void ReadObjSimple::quit()
{
}

int ReadObjSimple::compute(const char *port)
{
    (void)port;

    // get the file name
    filename = objFileParam->getValue();

    if (filename != NULL)
    {
        // open the file
        if (openFile())
        {
            sendInfo("File %s open", filename);

            // read the file, create the lists and create a COVISE polygon object
            readFile();
        }
        else
        {
            sendError("Error opening file %s", filename);
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
ReadObjSimple::readFile()
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
    char vertexString[LINE_SIZE];

    int *pi; // polygon list
    int *piTmp;
    int newPiSize, oldPiSize = CHUNK_SIZE;
    static int numPiAlloc = 1;

    char line[LINE_SIZE]; // line in an obj file
    char *first; // current position in line
    char key[LINE_SIZE]; // keyword
    int numScanned; // number of characters scanned with sscanf
    coDoPolygons *polygonObject; // output object
    const char *polygonObjectName; // output object name assigned by the controller

    // allocate memory for the lists
    cx = new float[CHUNK_SIZE];
    cy = new float[CHUNK_SIZE];
    cz = new float[CHUNK_SIZE];
    ci = new int[CHUNK_SIZE];
    pi = new int[CHUNK_SIZE];

    // read one line after another
    while (fgets(line, LINE_SIZE, fp) != NULL)
    {
        // find first non-space character
        first = line;
        while (*first != '\0' && isspace(*first))
            first++;

        // skip blank lines and comments
        if (*first == '\0' || *first == '#')
            // read the next line
            continue;

        // read the keyword
        if (sscanf(first, "%s%n", key, &numScanned) < 1)
        {
            cerr << "ReadObjSimple::readFile:: sscanf1 failed" << endl;
        }
        first += numScanned;

        if (strcasecmp("v", key) == 0)
        {
            // found an obj vertex definition
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
            if (sscanf(first, "%f %f %f", &(cx[numCoords - 1]), &(cy[numCoords - 1]), &(cz[numCoords - 1])) != 3)
            {
                cerr << "ReadObjSimple::readFile:: sscanf2 failed" << endl;
            }
        }

        else if (strcasecmp("f", key) == 0)
        {
            // found an obj polygon definition
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
            while (sscanf(first, "%s%n", vertexString, &numScanned) == 1)
            {
                // position the pointer behind the scanned string
                first += numScanned;

                numVertices++;

                // test if we have enough memory
                if (numVertices > (oldCiSize))
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

                // obj indices start with 1, COVISE indices start with 0
                ci[numVertices - 1] = ((int)strtol(vertexString, NULL, 10)) - 1;
            }
        }

        else if (strcasecmp(key, "vn") || strcasecmp(key, "vt") || strcasecmp(key, "g") || strcasecmp(key, "usemtl") || strcasecmp(key, "mtllib") || strcasecmp(key, "bevel") || strcasecmp(key, "bmat") || strcasecmp(key, "bsp") || strcasecmp(key, "bzp") || strcasecmp(key, "c_interp") || strcasecmp(key, "cdc") || strcasecmp(key, "con") || strcasecmp(key, "cstype") || strcasecmp(key, "ctech") || strcasecmp(key, "curv") || strcasecmp(key, "curv2") || strcasecmp(key, "d_interp") || strcasecmp(key, "deg") || strcasecmp(key, "end") || strcasecmp(key, "hole") || strcasecmp(key, "l") || strcasecmp(key, "lod") || strcasecmp(key, "maplib") || strcasecmp(key, "mg") || strcasecmp(key, "o") || strcasecmp(key, "p") || strcasecmp(key, "param") || strcasecmp(key, "parm") || strcasecmp(key, "res") || strcasecmp(key, "s") || strcasecmp(key, "scrv") || strcasecmp(key, "shadow_obj") || strcasecmp(key, "sp") || strcasecmp(key, "stech") || strcasecmp(key, "step") || strcasecmp(key, "surf") || strcasecmp(key, "trace_obj") || strcasecmp(key, "trim") || strcasecmp(key, "usemap") || strcasecmp(key, "vp"))
        {
            //fprintf(stderr,"ReadObj:: skipping key %s\n", key);
        }
        else
        {
            sendInfo("unrecognized key %s", key);
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
}

bool ReadObjSimple::openFile()
{
    sendInfo("Opening file %s", filename);

    // open the obj file
    if ((fp = Covise::fopen((char *)filename, "r")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", filename);
        return false;
    }
    else
    {
        return true;
    }
}

MODULE_MAIN(Examples, ReadObjSimple)
