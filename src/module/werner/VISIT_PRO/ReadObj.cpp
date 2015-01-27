/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#define DEBUGMODE 1

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

//lenght of a line
#define LINE_SIZE 8192

// portion for resizing data
#define CHUNK_SIZE 4096

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadObj.h"
#include <api/coModule.h>

char *ReadObj::objFile, *ReadObj::mtlFile; // obj file name
FILE *ReadObj::objFp, *ReadObj::mtlFp;
int ReadObj::numMtls; // no of materials in the mtl file
int *ReadObj::pcList; // list of packed colors from the mtl file
int ReadObj::currentColor; // current color in packed format
ReadObj::mtlNameType *ReadObj::mtlNameList; // list of material names in the mtl file

coDistributedObject *ReadObj::read(const char *objFile,
                                   const char *objName,
                                   const char *unit)
{
    float scale;

    if (0 == strcmp("mm", unit))
        scale = 1.0;
    else if (0 == strcmp("cm", unit))
        scale = 10.0;
    else if (0 == strcmp("ft", unit))
        scale = 25.4;
    else if (0 == strcmp("m", unit))
        scale = 1000.0;
    else
        scale = 1.0;

    numMtls = 0;

    char infobuf[500]; // buffer for COVISE info and error messages

    // get the name of the material file
    //Covise::get_browser_param("mtlFile", &mtlFile);
    mtlFile = NULL;

    if (mtlFile != NULL)
    {
        // open the material file
        if (mtlFp = openFile(mtlFile))
        {
            coModule::sendInfo("File %s open", mtlFile);
            Covise::sendInfo(infobuf);

            // read the file, create a list of colors
            readMtlFile();
        }
        else
        {
            coModule::sendError("Error opening file %s", mtlFile);
            // set a default color
            numMtls = 1;
            currentColor = makePackedColor(0.9, 0.9, 0.9, 1.0);
        }
    }
    else
    {
        // set a default color
        numMtls = 1;
        currentColor = makePackedColor(0.9, 0.9, 0.9, 1.0);
    }

    // get the file name
    // Covise::get_browser_param("objFile", &objFile);

    if (objFile != NULL)
    {
        // open the file
        if (objFp = openFile(objFile))
        {
            coModule::sendInfo("File %s open", objFile);

            // read the file, create the lists and create a COVISE polygon object
            return readObjFile(objName, scale);
        }
        else
        {
            coModule::sendError("Error opening file %s", objFile);
            return NULL;
        }
    }
    else
    {
        Covise::sendError("ERROR: objFile is NULL");
    }

    return NULL;
}

coDistributedObject *
ReadObj::readObjFile(const char *objName, float scale)
{
    int numCoords = 0, numVertices = 0, numPolys = 0, numNormals = 0;

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

    float *normals_x, *normals_y, *normals_z; // normal list
    float *normals_xTmp, *normals_yTmp, *normals_zTmp;
    //float *new_normals_x, *new_normals_y, *new_normals_z;
    //char *normalName;

    int newNSize, oldNSize = CHUNK_SIZE; // size of the normal lists
    static int numNAlloc = 1; // number of 'allocs' for the normal lists
    int *norm, numCoordOrig = 0, *co;

    int newPiSize, oldPiSize = CHUNK_SIZE;
    static int numPiAlloc = 1;
    coDoPolygons *polygonObject; // output object
    //char *polygonObjectName; // output object name assignet by the controller

    char line[LINE_SIZE]; // line in an obj file
    char *first; // current position in line
    char key[LINE_SIZE]; // keyword
    char infobuf[300]; // buffer for COVISE info and error messages
    int numScanned; // number of characters scanned with sscanf
    mtlNameType currentMtl;
    //char *colorObjectName;
    int *colorList, *colorListTmp; // list of packed colors

    typedef struct
    {
        List<int> duplCoord;
        List<int> duplNormals;
    } duplListElement;
    duplListElement *duplList; // duplicate vertexes to handle normals right

    // allocate memory for the lists
    cx = new float[CHUNK_SIZE];
    cy = new float[CHUNK_SIZE];
    cz = new float[CHUNK_SIZE];
    ci = new int[CHUNK_SIZE];
    pi = new int[CHUNK_SIZE];
    normals_x = new float[CHUNK_SIZE];
    normals_y = new float[CHUNK_SIZE];
    normals_z = new float[CHUNK_SIZE];
    colorList = new int[CHUNK_SIZE];

    currentColor = makePackedColor(0.9, 0.9, 0.9, 1.0);

    // read one line after another
    while (fgets(line, LINE_SIZE, objFp) != NULL)
    {
        // find first non-space character
        first = line;
        while (*first != '\0' && isspace(*first))
        {
            first++;
        }

        // skip blank lines and comments
        if (*first == '\0' || *first == '#')
        {
            // read the next line
            continue;
        }

        // read the keyword
        sscanf(first, "%s%n", key, &numScanned);
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
            sscanf(first, "%f %f %f", &(cx[numCoords - 1]), &(cy[numCoords - 1]), &(cz[numCoords - 1]));
            cx[numCoords - 1] *= scale;
            cy[numCoords - 1] *= scale;
            cz[numCoords - 1] *= scale;
        }
        else if (strcasecmp("vn", key) == 0)
        {
            // found an obj vertex definition
            //-> create the coordinate lists

            if (numNormals == 0) // only for the first time
            {
                duplList = new duplListElement[numCoords];
                norm = new int[10 * numCoords];
                co = new int[10 * numCoords];
            }
            numNormals++;

            // test if we allocated enough memory
            if (numNormals > (oldNSize))
            {
                // allocate more memory
                numNAlloc++;
                newNSize = numNAlloc * CHUNK_SIZE;
                normals_xTmp = new float[newNSize];
                normals_yTmp = new float[newNSize];
                normals_zTmp = new float[newNSize];
                memcpy(normals_xTmp, normals_x, oldNSize * sizeof(float));
                memcpy(normals_yTmp, normals_y, oldNSize * sizeof(float));
                memcpy(normals_zTmp, normals_z, oldNSize * sizeof(float));
                delete[] normals_x;
                delete[] normals_y;
                delete[] normals_z;
                normals_x = normals_xTmp;
                normals_y = normals_yTmp;
                normals_z = normals_zTmp;
                oldNSize = newNSize;
            }
            // scan the line
            sscanf(first, "%f %f %f", &(normals_x[numNormals - 1]), &(normals_y[numNormals - 1]), &(normals_z[numNormals - 1]));
        }
        else if (strcasecmp("f", key) == 0)
        {
            // found an obj polygon definition
            // -> create the coordinate index list and the polygon list
            // -> create the color list

            numPolys++;

            // create the polygons lists
            if (numPolys > (oldPiSize))
            {
                // resize polygon list
                numPiAlloc++;
                newPiSize = numPiAlloc * CHUNK_SIZE;
                piTmp = new int[newPiSize];
                memcpy(piTmp, pi, oldPiSize * sizeof(int));
                delete[] pi;
                pi = piTmp;

                // resize color list
                colorListTmp = new int[newPiSize];
                memcpy(colorListTmp, colorList, oldPiSize * sizeof(int));
                delete[] colorList;
                colorList = colorListTmp;

                // now use this list lenght
                oldPiSize = newPiSize;
            }
            pi[numPolys - 1] = numVertices;
            colorList[numPolys - 1] = currentColor;

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
                // normal index at the end of ci//ti//ni
                char *normString = strrchr(vertexString, '/');

                if (normString)
                    norm[numVertices] = ((int)strtol(normString + 1, NULL, 10)) - 1;
                else
                    norm[numVertices] = 0;

                int found = 0;
                duplList[ci[numVertices - 1]].duplNormals.reset();
                while (!found && duplList[ci[numVertices - 1]].duplNormals.current() != 0L)
                {
                    if (*(duplList[ci[numVertices - 1]].duplNormals.current()) == norm[numVertices])
                    {
                        found = 1;
                    }
                    duplList[ci[numVertices - 1]].duplNormals.next();
                }

                if (!found)
                {
                    if (duplList[ci[numVertices - 1]].duplNormals.get_first() != NULL)
                    {
                        if (numCoordOrig == 0) // save original number of coordinates
                        {
                            numCoordOrig = numCoords;
                        }

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

                        // copy coords
                        cx[numCoords - 1] = cx[ci[numVertices - 1]];
                        cy[numCoords - 1] = cy[ci[numVertices - 1]];
                        cz[numCoords - 1] = cz[ci[numVertices - 1]];

                        co[numVertices] = numCoords - 1;
                        duplList[ci[numVertices - 1]].duplCoord.add(co + numVertices);
                        duplList[ci[numVertices - 1]].duplNormals.add(norm + numVertices);
                        ci[numVertices - 1] = numCoords - 1; // new point in polygon
                    }
                    else
                    {
                        duplList[ci[numVertices - 1]].duplNormals.add(norm + numVertices);
                    }
                }
            }
        }
        else if (strcasecmp("usemtl", key) == 0)
        {
            sscanf(first, "%s", currentMtl);
            currentColor = getCurrentColor(currentMtl);
        }
        else if (strcasecmp(key, "vn") || strcasecmp(key, "vt") || strcasecmp(key, "g") || strcasecmp(key, "mtllib") || strcasecmp(key, "bevel") || strcasecmp(key, "bmat") || strcasecmp(key, "bsp") || strcasecmp(key, "bzp") || strcasecmp(key, "c_interp") || strcasecmp(key, "cdc") || strcasecmp(key, "con") || strcasecmp(key, "cstype") || strcasecmp(key, "ctech") || strcasecmp(key, "curv") || strcasecmp(key, "curv2") || strcasecmp(key, "d_interp") || strcasecmp(key, "deg") || strcasecmp(key, "end") || strcasecmp(key, "hole") || strcasecmp(key, "l") || strcasecmp(key, "lod") || strcasecmp(key, "maplib") || strcasecmp(key, "mg") || strcasecmp(key, "o") || strcasecmp(key, "p") || strcasecmp(key, "param") || strcasecmp(key, "parm") || strcasecmp(key, "res") || strcasecmp(key, "s") || strcasecmp(key, "scrv") || strcasecmp(key, "shadow_obj") || strcasecmp(key, "sp") || strcasecmp(key, "stech") || strcasecmp(key, "step") || strcasecmp(key, "surf") || strcasecmp(key, "trace_obj") || strcasecmp(key, "trim") || strcasecmp(key, "usemap") || strcasecmp(key, "vp"))
        {
            //fprintf(stderr,"ReadObj:: skipping key %s\n", key);
        }
        else
        {
            strcpy(infobuf, "unrecognized key ");
            strcat(infobuf, key);
            Covise::sendInfo(infobuf);
        }
    }

    sprintf(infobuf, "found %d coordinates, %d vertices, %d polygons, %d normals", numCoords, numVertices, numPolys, numNormals);
    Covise::sendInfo(infobuf);

    // get the COVISE output object name from the controller
    //polygonObjectName = Covise::get_object_name("polygons");

    // create the COVISE output object
    polygonObject = new coDoPolygons(objName, numCoords,
                                     cx, cy, cz, numVertices,
                                     ci, numPolys, pi);

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
    delete[] normals_x;
    delete[] normals_y;
    delete[] normals_z;

    if (numNormals > 0)
    {
        delete[] duplList;
        delete[] norm;
        delete[] co;
    }
    delete[] colorList;

    return polygonObject;
}

FILE *ReadObj::openFile(const char *filename)
{
    char infobuf[300];
    FILE *fp;

    //strcpy(infobuf, "Opening file ");
    //strcat(infobuf, filename);
    //Covise::sendInfo(infobuf);

    // open the obj file
    if ((fp = Covise::fopen(filename, "r")) == NULL)
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, filename);
        Covise::sendError(infobuf);
        return (NULL);
    }
    else
    {
#ifdef DEBUGMODE
        fprintf(stderr, "File %s open\n", filename);
#endif
        return (fp);
    }
}

void ReadObj::readMtlFile()
{
#ifdef DEBUGMODE
    fprintf(stderr, "ReadObj::readMtlFile\n");
#endif
    char line[LINE_SIZE]; // line in an obj file
    char *first; // current position in line
    char key[LINE_SIZE]; // keyword
    int numScanned;
    int newCSize, oldCSize = CHUNK_SIZE; // sizes of the lists
    numMtls = 0;
    static int numCAlloc = 1;
    float *rList, *gList, *bList, *aList; // arrays with the rgba colors
    // temporary arrays for resizing the lists
    float *rListTmp, *gListTmp, *bListTmp, *aListTmp;
    mtlNameType *mtlNameListTmp;
    int i;
    int *pcListTmp; // temporary list of packed colors

    // alloc memory
    pcList = new int[CHUNK_SIZE];
    rList = new float[CHUNK_SIZE];
    gList = new float[CHUNK_SIZE];
    bList = new float[CHUNK_SIZE];
    aList = new float[CHUNK_SIZE];
    mtlNameList = new mtlNameType[CHUNK_SIZE];

    // give it default values
    for (i = 0; i < CHUNK_SIZE; i++)
    {
        rList[i] = gList[i] = bList[i] = 0.9;
        aList[i] = 1.0;
        strcpy(mtlNameList[i], "None");
    }

    // read one line after another
    while (fgets(line, LINE_SIZE, mtlFp) != NULL)
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
        sscanf(first, "%s%n", key, &numScanned);
        first += numScanned;

        if (strcasecmp("newmtl", key) == 0)
        {
            numMtls++;
            if (numMtls > (oldCSize))
            {

                // allocate more memory
                numCAlloc++;
                newCSize = numCAlloc * CHUNK_SIZE;
                pcListTmp = new int[newCSize];
                rListTmp = new float[newCSize];
                gListTmp = new float[newCSize];
                bListTmp = new float[newCSize];
                aListTmp = new float[newCSize];
                mtlNameListTmp = new mtlNameType[newCSize];
                memcpy(pcListTmp, pcList, oldCSize * sizeof(int));
                memcpy(rListTmp, rList, oldCSize * sizeof(float));
                memcpy(gListTmp, gList, oldCSize * sizeof(float));
                memcpy(bListTmp, bList, oldCSize * sizeof(float));
                memcpy(aListTmp, aList, oldCSize * sizeof(float));
                memcpy(mtlNameListTmp, mtlNameList, oldCSize * sizeof(mtlNameType));
                delete[] pcList;
                delete[] rList;
                delete[] gList;
                delete[] bList;
                delete[] aList;
                delete[] mtlNameList;
                pcList = pcListTmp;
                rList = rListTmp;
                gList = gListTmp;
                bList = bListTmp;
                aList = aListTmp;
                mtlNameList = mtlNameListTmp;
                // give it default values
                for (i = oldCSize; i < newCSize; i++)
                {
                    rList[i] = gList[i] = bList[i] = 0.9;
                    aList[i] = 1.0;
                    strcpy(mtlNameList[i], "None");
                }

                oldCSize = newCSize;
            }

            sscanf(first, "%s", mtlNameList[numMtls - 1]);
        }
        else if (strcasecmp("Kd", key) == 0)
        {
            sscanf(first, "%f %f %f", &(rList[numMtls - 1]), &(gList[numMtls - 1]), &(bList[numMtls - 1]));
        }
        else if ((strcasecmp("Tr", key) == 0) || (strcasecmp("Tf", key) == 0))
        {
            sscanf(first, "%f", &(aList[numMtls - 1]));
        }
    }
    for (i = 0; i < numMtls; i++)
        pcList[i] = makePackedColor(rList[i], gList[i], bList[i], 1.0 - aList[i]);

#ifdef DEBUGMODE
    fprintf(stderr, "found %d material\n", numMtls);

    for (i = 0; i < numMtls; i++)
    {
        fprintf(stderr, "name=%s pc=%d rgba=[%f %f %f %f]\n", mtlNameList[i], pcList[i], rList[i], gList[i], bList[i], aList[i]);
    }
#endif
}

int ReadObj::makePackedColor(float r, float g, float b, float a)
{
    unsigned int rgba;

    unsigned char rc, gc, bc, ac;
    rc = r * 255;
    gc = g * 255;
    bc = b * 255;
    ac = a * 255;

    rgba = (rc << 24) | (gc << 16) | (bc << 8) | ac;

    return rgba;
}

int ReadObj::getCurrentColor(char *mtlName)
{
    int i;

    for (i = 0; i < numMtls; i++)
    {
        if (strcmp(mtlName, mtlNameList[i]) == 0)
            return (pcList[i]);
    }
    return 0;
}
