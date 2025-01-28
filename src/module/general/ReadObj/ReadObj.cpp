/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Reader for Wavefront OBJ Format	    	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Rainer		                                          **
 **                                                                        **
 ** History:  						                  **
 ** 01-September-99	v1			                          **
 **                                                                        **
 **                                                                        **
\**************************************************************************/
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

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <ctype.h>
#include "ReadObj.h"
#include <util/covise_list.h>
#include <api/coFileBrowserParam.h>
#include <do/coDoPixelImage.h>
#include <do/coDoTexture.h>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

ReadObj::ReadObj(int argc, char *argv[])
    : coModule(argc, argv, "Wavefront OBJ Reader")
{
    // the output ports
    polyOut = addOutputPort("GridOut0", "Polygons", "geometry polygons");
    colorOut = addOutputPort("DataOut0", "RGBA", "polygon colors");
    normalOut = addOutputPort("DataOut1", "Vec3", "polygon normals");
    textureOut = addOutputPort("TextureOut0", "Texture",
        "diffuse texture map (only that of the first material and first mesh!)");

    // select the OBJ file name with a file browser
    objFileBrowser = addFileBrowserParam("objFile", "OBJ file");
    objFileBrowser->setValue("data/", "*.obj/*");

    // select the MTL file name with a file browser
    mtlFileBrowser = addFileBrowserParam("mtlFile", "MTL file");
    mtlFileBrowser->setValue("data/", "*.mtl/*");

    numMtls = 0;
}

int ReadObj::compute(const char *)
{
    // get the file name
    const char *objFile = objFileBrowser->getValue();

    // get the base path so we can find textures etc.
    fs::path p(objFile);
    basePath = p.parent_path();

    // get the name of the material file
    const char *mtlFile = mtlFileBrowser->getValue();

    if (mtlFile != NULL)
    {
        // open the material file
        mtlFp = openFile(mtlFile);
        if (mtlFp)
        {
            Covise::sendInfo("File %s open", mtlFile);

            // read the file, create a list of colors
            readMtlFile();
        }
        else
        {
            Covise::sendInfo("Error opening file %s", mtlFile);
            // set a default color
            numMtls = 0;
            currentColor = makePackedColor(0.9f, 0.9f, 0.9f, 1.0f);
        }
    }
    else
    {
        Covise::sendInfo("ERROR: mtlFile is NULL");
        // set a default color
        numMtls = 0;
        currentColor = makePackedColor(0.9f, 0.9f, 0.9f, 1.0f);
    }

    if (objFile != NULL)
    {
        // open the file
        objFp = openFile(objFile);
        if (objFp)
        {
            Covise::sendInfo("File %s open", objFile);

            // read the file, create the lists and create a COVISE polygon object
            readObjFile();
        }
        else
        {
            Covise::sendError("Error opening file %s", objFile);
            return STOP_PIPELINE;
        }
    }
    else
    {
        Covise::sendError("ERROR: objFile is NULL");
        return STOP_PIPELINE;
    }

    // Try to load the first kd-map:
    int w, h, n;
    if (mapKdList[0][0] != '\0') {
        void *image = stbi_load(mapKdList[0], &w, &h, &n, 0);
        if (image) {
            Covise::sendInfo("Load texture map %s", mapKdList[0]);
            coDoPixelImage *pix = new coDoPixelImage("kdPix", w, h, n, n, (const char *)image);

            std::vector<int> txIndex(tcList[0].size());
            float *texCoords[2];
            std::vector<float> texCoordsX(tcList[0].size());
            std::vector<float> texCoordsY(tcList[0].size());
            texCoords[0] = texCoordsX.data();
            texCoords[1] = texCoordsY.data();


            for (size_t i=0; i<tcList[0].size(); ++i) {
                txIndex[i] = i;
                texCoords[0][i] = tcList[0][i].x;
                texCoords[1][i] = tcList[0][i].y;
            }

            std::vector<int> indices(tcList.size()/2);
            for (size_t i=0;i<indices.size();++i) indices[i] = i;
            coDoTexture *texture = new coDoTexture(textureOut->getNewObjectInfo(), pix, 0, n, 0,
                    tcList[0].size(), txIndex.data(), txIndex.size(), texCoords);
            texture->addAttribute("WRAP_MODE", "repeat");
            texture->addAttribute("MIN_FILTER", "linear");
            texture->addAttribute("MAG_FILTER", "linear");

            textureOut->setCurrentObject(texture);
            Covise::sendInfo("Texture map of size %i x %i loaded", w, h);
        } else {
            Covise::sendInfo("Error opening texture map %s", mapKdList[0]);
        }
    }

    return CONTINUE_PIPELINE;
}

void
ReadObj::readObjFile()
{
    int numCoords = 0, numTexCoords = 0, numVertices = 0, numPolys = 0, numNormals = 0;

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
    float *new_normals_x, *new_normals_y, *new_normals_z;
    coDoVec3 *normalObject;

    int newNSize, oldNSize = CHUNK_SIZE; // size of the normal lists
    static int numNAlloc = 1; // number of 'allocs' for the normal lists
    int *norm = NULL, numCoordOrig = 0, *co = NULL;

    int newPiSize, oldPiSize = CHUNK_SIZE;
    static int numPiAlloc = 1;
    coDoPolygons *polygonObject; // output object

    char line[LINE_SIZE]; // line in an obj file
    char *first; // current position in line
    char key[LINE_SIZE]; // keyword
    int numScanned; // number of characters scanned with sscanf
    mtlNameType currentMtl;
    coDoRGBA *colorObject;
    int *colorList, *colorListTmp; // list of packed colors

    typedef struct
    {
        List<int> duplCoord;
        List<int> duplNormals;
    } duplListElement;
    duplListElement *duplList = NULL; // duplicate vertexes to handle normals right

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

    currentColor = makePackedColor(0.9f, 0.9f, 0.9f, 1.0f);

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
            std::cout << key << '\n';
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
        }
        else if (strcasecmp("vt", key) == 0)
        {
            std::cout << key << '\n';
            // found an obj vertex definition
            //-> create the coordinate lists


            if (numTexCoords == 0) // only for the first time
            {
                tcList.push_back(TexCoordList{});
            }
            numTexCoords++;

            // scan the line
            TexCoord tc;
            sscanf(first, "%f %f", &tc.x, &tc.y);
            tcList.back().push_back(tc);
        }
        else if (strcasecmp("vn", key) == 0)
        {
            std::cout << key << '\n';
            // found an obj vertex definition
            //-> create the coordinate lists

            if (numNormals == 0) // only for the first time
            {
                duplList = new duplListElement[numCoords];
                norm = new int[50 * numCoords];
                co = new int[50 * numCoords];
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
            if ((numPolys == 0) && (norm == NULL))
            {
                duplList = new duplListElement[numCoords];
                norm = new int[50 * numCoords];
                co = new int[50 * numCoords];
            }
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
            Covise::sendInfo("unrecognized key %s", key);
        }
    }

    Covise::sendInfo("found %d coordinates, %d vertices, %d polygons, %d normals", numCoords, numVertices, numPolys, numNormals);

    // create the COVISE output object
    polygonObject = new coDoPolygons(polyOut->getNewObjectInfo(), numCoords, cx, cy, cz, numVertices, ci, numPolys, pi);
    //@@@    polygonObject->addAttribute("TRANSPARENCY","0.001");

    // set the vertex order for twosided lighting in the renderer
    // 1=clockwise 2=counterclockwise
    // missing vertex order -> no twosided lighting (inner surface not lighted)
    // wrong vertex order -> wrong lighting for surfaces with normals
    polygonObject->addAttribute("vertexOrder", "2");

    polyOut->setCurrentObject(polygonObject);

    //
    // construct the right array of normals
    //
    if (numNormals > 0)
    {
        int i, pos, c;
        normalObject = new coDoVec3(normalOut->getNewObjectInfo(), numCoords);
        normalObject->getAddresses(&new_normals_x, &new_normals_y, &new_normals_z);
        for (i = 0; i < numCoordOrig; i++)
        {
            duplList[i].duplNormals.reset();
            int *posPtr = duplList[i].duplNormals.current();
            if (posPtr)
            {
                pos = *(duplList[i].duplNormals.current());
                new_normals_x[i] = normals_x[pos];
                new_normals_y[i] = normals_y[pos];
                new_normals_z[i] = normals_z[pos];
                duplList[i].duplNormals.next();
                duplList[i].duplCoord.reset();
                while (duplList[i].duplNormals.current() != 0L)
                {
                    c = *(duplList[i].duplCoord.current());
                    pos = *(duplList[i].duplNormals.current());
                    new_normals_x[c] = normals_x[pos];
                    new_normals_y[c] = normals_y[pos];
                    new_normals_z[c] = normals_z[pos];
                    duplList[i].duplNormals.next();
                    duplList[i].duplCoord.next();
                }
            }
            else
            {
                new_normals_x[i] = 0.0;
                new_normals_y[i] = 0.0;
                new_normals_z[i] = 0.0;
            }
        }
        normalOut->setCurrentObject(normalObject);
    }

    // delete the lists
    delete[] cx;
    delete[] cy;
    delete[] cz;
    delete[] ci;
    delete[] pi;
    delete[] normals_x;
    delete[] normals_y;
    delete[] normals_z;

    delete[] duplList;
    delete[] norm;
    delete[] co;

    colorObject = new coDoRGBA(colorOut->getNewObjectInfo(), numPolys, colorList);
    colorOut->setCurrentObject(colorObject);

    delete[] colorList;
}

FILE *ReadObj::openFile(const char *filename)
{
    FILE *fp;

    //strcpy(infobuf, "Opening file ");
    //strcat(infobuf, filename);
    //Covise::sendInfo(infobuf);

    // open the obj file
    if ((fp = Covise::fopen(filename, "r")) == NULL)
    {
        Covise::sendWarning("ERROR: Can't open file >> %s", filename);
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
    float *rList, *gList, *bList, *tList; // arrays with the rgba colors
    // temporary arrays for resizing the lists
    float *rListTmp, *gListTmp, *bListTmp, *tListTmp;
    mtlNameType *mtlNameListTmp;
    int i;
    int *pcListTmp; // temporary list of packed colors

    // alloc memory
    pcList = new int[CHUNK_SIZE];
    rList = new float[CHUNK_SIZE];
    gList = new float[CHUNK_SIZE];
    bList = new float[CHUNK_SIZE];
    tList = new float[CHUNK_SIZE];
    mapKdList = new PATH[CHUNK_SIZE];
    mtlNameList = new mtlNameType[CHUNK_SIZE];

    // give it default values
    for (i = 0; i < CHUNK_SIZE; i++)
    {
        rList[i] = gList[i] = bList[i] = 0.9f;
        tList[i] = 0.0;
        strcpy(mtlNameList[i], "None");
        memset(mapKdList[i], '\0', MAX_PATH_LEN);
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
                tListTmp = new float[newCSize];
                mtlNameListTmp = new mtlNameType[newCSize];
                memcpy(pcListTmp, pcList, oldCSize * sizeof(int));
                memcpy(rListTmp, rList, oldCSize * sizeof(float));
                memcpy(gListTmp, gList, oldCSize * sizeof(float));
                memcpy(bListTmp, bList, oldCSize * sizeof(float));
                memcpy(tListTmp, tList, oldCSize * sizeof(float));
                memcpy(mtlNameListTmp, mtlNameList, oldCSize * sizeof(mtlNameType));
                delete[] pcList;
                delete[] rList;
                delete[] gList;
                delete[] bList;
                delete[] tList;
                delete[] mapKdList;
                delete[] mtlNameList;
                pcList = pcListTmp;
                rList = rListTmp;
                gList = gListTmp;
                bList = bListTmp;
                tList = tListTmp;
                mtlNameList = mtlNameListTmp;
                // give it default values
                for (i = oldCSize; i < newCSize; i++)
                {
                    rList[i] = gList[i] = bList[i] = 0.9f;
                    tList[i] = 0.0f;
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
            sscanf(first, "%f", &(tList[numMtls - 1]));
        }
        else if (strcasecmp("map_Kd", key) == 0)
        {
            sscanf(first, "%s", (char *)&mapKdList[numMtls - 1]);
            fs::path p(mapKdList[numMtls - 1]);
            if (p.is_relative()) {
                fs::path absolutePath = basePath / fs::path(mapKdList[numMtls - 1]);
                memcpy(&mapKdList[numMtls - 1], absolutePath.string().c_str(),
                    absolutePath.string().length()*sizeof(char));
                mapKdList[numMtls - 1][absolutePath.string().length()] = '\0';
            }
        }
    }
    for (i = 0; i < numMtls; i++)
        pcList[i] = makePackedColor(rList[i], gList[i], bList[i], 1.0f - tList[i]);

#ifdef DEBUGMODE
    fprintf(stderr, "found %d material\n", numMtls);

    for (i = 0; i < numMtls; i++)
    {
        fprintf(stderr, "name=%s pc=%d rgba=[%f %f %f %f]\n", mtlNameList[i], pcList[i], rList[i], gList[i], bList[i], tList[i]);
    }
#endif

  
}

int ReadObj::makePackedColor(float r, float g, float b, float a)
{
    unsigned int rgba;

    //    *chptr     = (unsigned char)(a*255.0); chptr++;
    //    *(chptr)   = (unsigned char)(b*255.0); chptr++;
    //    *(chptr)   = (unsigned char)(g*255.0); chptr++;
    //    *(chptr)   = (unsigned char)(r*255.0);

    /*
       unsigned char *chptr;
       chptr      = (unsigned char *)&rgba;
       *(chptr)   = (unsigned char)(r*255.0); chptr++;
       *(chptr)   = (unsigned char)(g*255.0); chptr++;
       *(chptr)   = (unsigned char)(b*255.0); chptr++;
       *chptr     = (unsigned char)(a*255.0);
   */
    unsigned char rc, gc, bc, ac;
    rc = (unsigned char)(r * 255);
    gc = (unsigned char)(g * 255);
    bc = (unsigned char)(b * 255);
    ac = (unsigned char)(a * 255);

    rgba = (rc << 24) | (gc << 16) | (bc << 8) | ac;

    return (int)rgba;
}

int ReadObj::getCurrentColor(const char *mtlName)
{
    int i;

    for (i = 0; i < numMtls; i++)
    {
        if (strcmp(mtlName, mtlNameList[i]) == 0)
            return (pcList[i]);
    }
    return makePackedColor(0.9f, 0.9f, 0.9f, 1.0f);
}

MODULE_MAIN(IO, ReadObj)
