/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadMoldFlow
//
//
// Initial version: 2003-09-01 Uwe
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2003 by Uwe Woessner
// +++++++++++++++++++++++++++++++++++++++++
// Changes:

#include "ReadMoldFlow.h"
#include <inttypes.h>
#include <errno.h>
#include <util/coviseCompat.h>
#include <string.h>

// remove trailing path from filename
inline const char *coBasename(const char *str)
{
    const char *lastslash = strrchr(str, '/');
    if (lastslash)
        return lastslash + 1;
    else
        return str;
}

// Module start
int
main(int argc, char *argv[])
{
    ReadMoldFlow *application = new ReadMoldFlow();
    application->start(argc, argv);
    return 0;
}

// Module set-up in Constructor
ReadMoldFlow::ReadMoldFlow()
    : coModule("Read MoldFlow")
{
    // file browser parameter
    p_nodeFilename = addFileBrowserParam("NodeFile", "Node file path");
    p_nodeFilename->setValue("/mnt/cod/sfb374/ikp/testmfl.nda", "*.nda");
    p_nodeFilename->setImmediate(1);
    p_resultFilename = addFileBrowserParam("ResultFile", "Result file path");
    p_resultFilename->setValue("/mnt/cod/sfb374/ikp/testpatran.nod", "*.nod;*.ele;*.xml");
    p_resultFilename->setImmediate(1);
    p_elementFilename = addFileBrowserParam("ElementFile", "Element file path");
    p_elementFilename->setValue("/mnt/cod/sfb374/ikp/testmfl.ela", "*.ela");
    p_elementFilename->setImmediate(1);
    p_numT = addInt32Param("numTimesteps", "Number of Timesteps");
    p_numT->setValue(1);
    p_numT->setImmediate(1);

    // Output ports
    p_polyOut = addOutputPort("mesh", "coDoPolygons", "Polygons");
    p_resultOut = addOutputPort("data", "Float", "Nodal data");

    // to be added later for coloured binary files
    //colorOutPort= addOutputPort("color","DO_RGBA_Colors","color data");
    d_resultFile = NULL;
    d_nodeFile = NULL;
    d_elementFile = NULL;
}

ReadMoldFlow::~ReadMoldFlow()
{
}

// param callback read header again after all changes
void
ReadMoldFlow::param(const char *paraName)
{
    if (in_map_loading())
        return;
    else if (strcmp(paraName, "SetModuleTitle"))
    {
    }
}

// taken from old ReadMoldFlow module: 2-Pass reading
int ReadMoldFlow::readASCII()
{
    float *x_coord, *y_coord, *z_coord;
    int *vl, *el, i;
    char buf[600], nodeBuf[600];
    char buf2[900];

    // 1st pass: count sizes
    n_coord = 0;
    n_elem = 0;
    //int n_vert=0;
    int node = 0;
    int nodeNum = 0;
    int currt, t, endt, numNumbers;

    const char *resultFileName = p_resultFilename->getValue();

    rewind(d_nodeFile);
    rewind(d_elementFile);
    while (!feof(d_elementFile))
    {
        if (fgets(buf, 600, d_elementFile))
        {
            n_elem++;
        }
    }
    rewind(d_elementFile);
    while (!feof(d_nodeFile))
    {
        if (fgets(buf, 600, d_nodeFile))
        {
            n_coord++;
        }
    }
    rewind(d_nodeFile);

    char dp[900];
    char dpend[900];
    currt = 0;

    strcpy(dp, resultFileName);
    i = strlen(dp) - 1;
    while (dp[i] && ((dp[i] < '0') || (dp[i] > '9')))
        i--;
    // dp[i] ist jetzt die letzte Ziffer, alles danach ist Endung
    if (dp[i])
    {
        strcpy(dpend, dp + i + 1); // dpend= Endung;
        dp[i + 1] = '\0';
    }
    else
    {
        dpend[0] = '\0';
    }
    numNumbers = 0;
    bool zeros = false;
    while ((dp[i] >= '0') && (dp[i] <= '9'))
    {
        if (dp[i] == '0')
            zeros = true;
        else
            zeros = false;
        i--;
        numNumbers++;
    }
    if (dp[i])
    {
        sscanf(dp + i + 1, "%d", &currt); //currt = Aktueller Zeitschritt
        endt = currt + numt;
        dp[i + 1] = 0; // dp = basename
    }
    else
    {
        currt = 0;
    }

    int fileNumber = currt;

    coDistributedObject **Data_sets = new coDistributedObject *[numt + 1];
    Data_sets[0] = NULL;
    coDistributedObject **Grid_sets = new coDistributedObject *[numt + 1];
    Grid_sets[0] = NULL;
    const char *Results = p_resultOut->getObjName();
    if (numt == 1)
    {
        d_resultFile = fopen(resultFileName, "r");
        if (!d_resultFile)
        {
            sprintf(buf, "Could not read %s: %s", resultFileName, strerror(errno));
            sendError(buf);
            return STOP_PIPELINE;
        }
        coDoFloat *dataObj = NULL;
        if ((dataObj = readResults(Results)) == NULL)
        {
            return STOP_PIPELINE;
        }
        if (d_resultFile)
            fclose(d_resultFile);

        p_resultOut->setCurrentObject(dataObj);
    }
    else
    {
        for (t = currt; t < endt; t++)
        {

            d_resultFile = NULL;
            int numTries = 0;
            while (numTries < 100)
            {
                if (zeros)
                {
                    sprintf(buf, "%s%0*d%s", dp, numNumbers, fileNumber, dpend);
                    //fprintf(stderr,"Opening file %s\n",buf);
                }
                else
                    sprintf(buf, "%s%d%s", dp, fileNumber, dpend);
                d_resultFile = fopen(buf, "r");
                if (d_resultFile)
                {
                    sprintf(buf2, "Reading file %s\n", buf);
                    Covise::sendInfo(buf2);
                    break;
                }
                numTries++;

                fileNumber++;
            }
            if (d_resultFile)
            {
                sprintf(buf, "%s_%d", Results, fileNumber);
                coDoFloat *dataObj = NULL;
                if ((dataObj = readResults(buf)) == NULL)
                {
                    break;
                }
                if (d_resultFile)
                    fclose(d_resultFile);
                d_resultFile = NULL;
                for (i = 0; Data_sets[i]; i++)
                    ;
                Data_sets[i] = dataObj;
                Data_sets[i + 1] = NULL;
            }
            else
            {
                break;
            }
            fileNumber++;
        }

        coDoSet *Data_set = NULL;
        if (Data_sets[0])
            Data_set = new coDoSet(Results, Data_sets);
        p_resultOut->setCurrentObject(Data_set);
    }

    const char *Surface = p_polyOut->getObjName();

    coDoPolygons *surface = NULL;
    //coDoPoints                  *points = NULL;
    //coDoFloat   *results = NULL;

    if (numt == 1)
    {
        surface = new coDoPolygons(Surface, n_coord, n_elem * 3, n_elem);
    }
    else
    {
        sprintf(buf, "%s_Grid", Surface);
        surface = new coDoPolygons(buf, n_coord, n_elem * 3, n_elem);
    }

    if (!surface || !surface->objectOk())
    {
        sendError("Error creating objects");
        return STOP_PIPELINE;
    }
    surface->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
    node = 0;
    while (!feof(d_nodeFile))
    {
        fgets(nodeBuf, 600, d_nodeFile);
        sscanf(nodeBuf, "%d %f %f %f", &nodeNum, &x_coord[node], &y_coord[node], &z_coord[node]);
        node++;
    }
    n_elem = 0;
    while (!feof(d_elementFile))
    {
        if (fgets(buf, 600, d_elementFile))
        {
            int v1 = 0, v2 = 0, v3 = 0;
            sscanf(buf, "%d %d %d %d", &nodeNum, &v1, &v2, &v3);
            vl[(n_elem * 3)] = v1 - 1;
            vl[(n_elem * 3) + 1] = v2 - 1;
            vl[(n_elem * 3) + 2] = v3 - 1;
            el[n_elem] = n_elem * 3;
            n_elem++;
        }
    }

    surface->addAttribute("vertexOrder", "2");
    surface->addAttribute("COLOR", "white");
    if (numt == 1)
    {
        p_polyOut->setCurrentObject(surface);
    }
    else
    {
        coDoSet *Grid_set = NULL;
        Grid_sets[0] = surface;
        for (i = 1; Data_sets[i]; i++)
        {
            Grid_sets[i] = surface;
            surface->incRefCount();
            Grid_sets[i + 1] = NULL;
        }

        if (Grid_sets[0])
        {
            Grid_set = new coDoSet(Surface, Grid_sets);
            Grid_set->addAttribute("TIMESTEP", "1 100");
        }
        p_resultOut->setCurrentObject(Grid_set);
        delete surface;
        delete[] Grid_sets;
        for (i = 0; Data_sets[i]; i++)
            delete Data_sets[i];
        delete[] Data_sets;
    }

    // DONE reading Polygons

    if (d_nodeFile)
        fclose(d_nodeFile);
    if (d_elementFile)
        fclose(d_elementFile);
    d_resultFile = NULL;
    d_nodeFile = NULL;
    d_elementFile = NULL;
    return CONTINUE_PIPELINE;
}

coDoFloat *ReadMoldFlow::readResults(const char *objName)
{
    char buf[600], *tmpbuf;
    float *values;
    int node = 0;
    int nodeNum = 0;
    coDoFloat *dataObj = NULL;
    fgets(buf, 600, d_resultFile); //header
    dataType = TYPE_PATRAN;
    bool nodalData = true;
    if (strncmp(buf, "<?xml", 5) == 0)
    {
        dataType = TYPE_XML;
        while (strstr(buf, "<Data>") == NULL)
            fgets(buf, 600, d_resultFile); //header
        fgets(buf, 600, d_resultFile); //header
        if (strstr(buf, "<ElementData") != NULL)
        {
            nodalData = false;
        }
    }
    else
    {
        int numInFile = 0;
        int maxitems = 0;
        fgets(buf, 600, d_resultFile); // number of nodes
        sscanf(buf, "%d %d", &numInFile, &maxitems);
        fgets(buf, 600, d_resultFile); //header
        if (strstr(buf, "EleID") != NULL)
        {
            nodalData = false;
        }
    }
    //sscanf(buf,"%d",&n_coord);
    if (nodalData)
    {
        dataObj = new coDoFloat(objName, n_coord);
    }
    else
    {
        dataObj = new coDoFloat(objName, n_elem);
    }
    //points = new coDoPoints(Surface, n_coord);

    dataObj->getAddress(&values);

    if (nodalData)
    {
        memset(values, 0, n_coord * sizeof(float));
    }
    else
    {
        memset(values, 0, n_elem * sizeof(float));
    }
    //points->getAddresses(&x_coord,&y_coord,&z_coord);
    if (dataType == TYPE_XML)
    {

        while (!feof(d_resultFile))
        {
            fgets(buf, 600, d_resultFile);
            if ((tmpbuf = strchr(buf, '>')))
            {
                tmpbuf++;
                sscanf(tmpbuf, "%f", &values[node]);
                fgets(buf, 600, d_resultFile);
                fgets(buf, 600, d_resultFile);
            }
            node++;
        }
    }
    else
    {
        while (!feof(d_resultFile))
        {
            fgets(buf, 600, d_resultFile);
            float tmpVal;
            sscanf(buf, "%d %f", &nodeNum, &tmpVal);
            values[nodeNum - 1] = tmpVal;
            node++;
        }
    }
    return dataObj;
}

int ReadMoldFlow::compute()
{
    numt = p_numT->getValue();
    openFiles();

    // Now, this must be an error:
    //     No message, readHeader already cries if problems occur
    if (!d_elementFile || !d_nodeFile)
        return STOP_PIPELINE;

    return readASCII();
}

// utility functions
void ReadMoldFlow::openFiles()
{
    char buffer[512];

    if (d_nodeFile)
        fclose(d_nodeFile);
    if (d_elementFile)
        fclose(d_elementFile);

    const char *nodeFileName = p_nodeFilename->getValue();
    const char *elementFileName = p_elementFilename->getValue();

    // Try to open file
    d_nodeFile = fopen(nodeFileName, "r");
    if (!d_nodeFile)
    {
        sprintf(buffer, "Could not read %s: %s", nodeFileName, strerror(errno));
        sendError(buffer);
        return;
    }
    // Try to open file
    d_elementFile = fopen(elementFileName, "r");
    if (!d_elementFile)
    {
        sprintf(buffer, "Could not read %s: %s", elementFileName, strerror(errno));
        sendError(buffer);
        return;
    }

    return;
}
