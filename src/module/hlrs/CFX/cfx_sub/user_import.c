/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
* ImportTemplate.c - Patran Neutral File Import
* reads packets 1 (nodes), 2 (elements) and 21 (named groups)
* and optionally packet 6 (loads), and sends data to TfC
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <fstream.h>
//#include <iostream.h>

#include "cfxImport.h"
#include "getargs.h"
#include "coSimClient.h"

static char options[] = "velF:";

static char *usgmsg[] = {
    "usage  : ImportTemplate.c [options] Patran_file",
    "options:",
    "   -v      = verbose output",
    "   -l      = process packet 6 - distributed loads",
    NULL
};

/*---------- print_error -------------------------------------------
* print error message and line number
*------------------------------------------------------------------*/

static int lineno = 0;

static void print_error(
#ifdef PROTOTYPE
    char *errmsg)
#else
    errmsg) char *errmsg;
#endif
{
    fprintf(stderr, "%s on line %d\n", errmsg, lineno);
}

/*---------- add_face -----------------------------------------------
* add an element face to the region list
*-------------------------------------------------------------------*/

static void add_face(
#ifdef PROTOTYPE
    int elemid, char *data)
#else
    elemid, data) int elemid;
char *data;
#endif
{
    int n, nnodes, nodes[8];
    ID_t nodeid[8];
    char errmsg[81];

    /* check for node flags set */

    for (nnodes = 0, n = 0; n < 8; n++)
    {
        if ('1' == data[n])
            nodes[nnodes++] = n;
    }

    /* if node flags set, use the node values */

    if (nnodes)
    {
        ID_t elemnodes[8];
        int elemtype = cfxImportGetElement(elemid, elemnodes);
        if (!elemtype)
        {
            sprintf(errmsg,
                    "element %d not found for packet 6\n", elemid);
            cfxImportFatal(errmsg);
        }
        for (n = 0; n < nnodes; n++)
        {
            if (nodes[n] >= elemtype)
            {
                sprintf(errmsg,
                        "invalid node flags for element %d\n", elemid);
                cfxImportFatal(errmsg);
            }
            nodeid[n] = elemnodes[nodes[n]];
        }
    }

    /* else get nodes from face number */

    else
    {
        int faceid = atoi(&data[8]);
        nnodes = cfxImportGetFace(elemid, faceid, nodeid);
        if (nnodes < 0)
        {
            sprintf(errmsg,
                    "element %d not found for packet 6\n", elemid);
            cfxImportFatal(errmsg);
        }
        if (0 == nnodes)
        {
            sprintf(errmsg,
                    "invalid face number for element %d\n", elemid);
            cfxImportFatal(errmsg);
        }
    }

    cfxImportAddReg(nnodes, nodeid);
}

/*========== main ===================================================*/

#define getline()                                      \
    {                                                  \
        if (NULL == fgets(buffer, sizeof(buffer), fp)) \
            cfxImportFatal("premature EOF");           \
        lineno++;                                      \
    }

int main(int argc, char **argv)
{

    int initconn;
    int numElem, numCoord, numConn;
    int *elem, *conn;
    float *x, *y, *z;
    int mtype, numblades;
    int *nodeInfo, *elemInfo, *wall;
    int *balance, *press;
    int colDiriclet, numDiriclet, colDiricletVals;
    int *diricletIndex;
    float *diricletVal;
    int colWall, numWall;
    int colBalance, numBalance;
    int n, packet, nlines;
    int nnodes;
    int elemid;
    ID_t nodeid[8];
    int lastid = -1, loadid;
    int verbose = 0, do_loads = 0;
    double xyz[3];
    char *p, buffer[256];
    char *testfile = NULL;
    FILE *fp, *outfp;
    struct stat st;
    int i, j;
    char regname[100];
    unsigned int nodes[4];
    int numbalanceregions = 0; // number of different balances
    int balanceregions[100]; // balanceidentifiers (100,200,...)
    int balanceregionscount[100]; //number of elements in each region
    int k;
    char filename[30];
    char balancename[30];
    FILE *file;

    int numwallregions = 0; // number of different wall regions
    int wallregions[50]; // wallidentifiers
    int wallregionscount[50]; // number of elements in each wall region
    int found;
    int nr, oldnr;
    unsigned int *objlist;
    unsigned int testvalue;
    unsigned int lokfacenr;
    unsigned int globfacenr;

    char outfile[200];

    //fprintf (stderr,"sleeping to allow to attach debugger ...\n");
    //_sleep(15000);
    //fprintf (stderr,"sleep end\n");

    //strcpy(outfile,"c:\\tmp\\user_import_out.txt");
    strcpy(outfile, "/tmp/user_import_out.txt");
    if ((outfp = fopen(&outfile[0], "w")) == NULL)
    {
        fprintf(stderr, "cannot open '%s' for writing!\n", outfile);
        return 0;
    }
    fprintf(stderr, "writing to %s\n", outfile);
    fprintf(outfp, "user_import: writing to outfile.\n");

    fprintf(stderr, "start user import\n");
    fprintf(outfp, "start user import\n");
    /* Creating Connection to Covise */

    fprintf(stderr, "Creating Connection to Covise\n");
    fprintf(outfp, "Creating Connection to Covise\n");
    if (coNotConnected() != 0)
    {
        coWSAInit();
        initconn = coInitConnect();
        if (initconn != 0)
        {
            fprintf(stderr, "Could not connect to Covise\n");
            fprintf(stderr, "initconn: %d\n", initconn);
            fprintf(outfp, "Could not connect to Covise\n");
            fprintf(outfp, "initconn: %d\n", initconn);
            return 0;
        }

        else
        {
            fprintf(stderr, "Connection ok\n");
            fprintf(outfp, "Connection ok\n");
        }
    }

    //////////////////// Get Grid ////////////////////

    fprintf(stderr, "receiving mtype\n");
    fprintf(outfp, "receiving mtype\n");
    recvData(&mtype, sizeof(int));
    recvData(&numblades, sizeof(int));
    fprintf(stderr, "Gitter wird empfangen\n");
    recvData(&numElem, sizeof(int));
    fprintf(stderr, "Sim: numElem: %d\n", numElem);
    fprintf(outfp, "Sim: numElem: %d\n", numElem);
    recvData(&numConn, sizeof(int));
    fprintf(stderr, "Sim: numConn: %d\n", numConn);
    fprintf(outfp, "Sim: numConn: %d\n", numConn);
    recvData(&numCoord, sizeof(int));
    fprintf(stderr, "Sim: numCoord: %d\n", numCoord);
    fprintf(outfp, "Sim: numCoord: %d\n", numCoord);

    elem = (int *)malloc(numElem * sizeof(int));
    conn = (int *)malloc(numConn * sizeof(int));

    x = (float *)malloc(numCoord * sizeof(float));
    y = (float *)malloc(numCoord * sizeof(float));
    z = (float *)malloc(numCoord * sizeof(float));

    recvData(elem, numElem * sizeof(int));
    recvData(conn, numConn * sizeof(int));

    recvData(x, numCoord * sizeof(float));
    recvData(y, numCoord * sizeof(float));
    recvData(z, numCoord * sizeof(float));

    fprintf(stderr, "Sim: Gitter empfangen\n");
    fprintf(outfp, "Sim: Gitter empfangen\n");
    fprintf(stderr, "Sim: elem[3]=%d\n", elem[3]);
    fprintf(stderr, "Sim: conn[4]=%d\n", conn[4]);
    fprintf(stderr, "Sim: x[0]=%6.3f\n", x[0]);

    //////////////Get Boundary Conditions/////////////////

    //// Diriclet Nodes

    fprintf(outfp, "Sim: Diriclet Nodes wird empfangen.....\n");

    recvData(&colDiriclet, sizeof(int));
    recvData(&numDiriclet, sizeof(int));
    recvData(&colDiricletVals, sizeof(int));

    diricletIndex = (int *)malloc(colDiriclet * numDiriclet * sizeof(int));
    recvData(diricletIndex, (colDiriclet * numDiriclet) * sizeof(int));

    fprintf(outfp, "Sim: colDiriclet: %d\n", colDiriclet);
    fprintf(outfp, "Sim: diricletIndex[8]=%d\n", diricletIndex[8]);
    fprintf(outfp, "Sim: Diriclet Nodes empfangen\n");

    //// Diriclet Values
    fprintf(outfp, "Sim: Diriclet Values wird empfangen.....\n");
    diricletVal = (float *)malloc((numDiriclet) * sizeof(float));
    recvData(diricletVal, (numDiriclet) * sizeof(float));
    fprintf(outfp, "Sim: Diriclet Values empfangen\n");

    //// Wall indices

    fprintf(outfp, "Sim: Wall indices wird empfangen.....\n");

    recvData(&colWall, sizeof(int));
    recvData(&numWall, sizeof(int));
    wall = (int *)malloc((colWall * numWall) * sizeof(int));
    recvData(wall, (colWall * numWall) * sizeof(int));
    fprintf(outfp, "Sim: wall[8]=%d\n", wall[8]);
    fprintf(outfp, "Sim: Wall indices empfangen\n");

    //// Balance indices

    fprintf(outfp, "Sim: Balance indices wird empfangen.....\n");

    recvData(&colBalance, sizeof(int));
    recvData(&numBalance, sizeof(int));
    balance = (int *)malloc((colBalance * numBalance) * sizeof(int));
    recvData(balance, (colBalance * numBalance) * sizeof(int));
    fprintf(outfp, "Sim: Balance indices empfangen\n");
    fprintf(outfp, "Sim: Balance indices empfangen\n");

    ///////////////// einsortieren //////////////////

    //// grid

    cfxImportInit();

    for (j = 0; j < numCoord; j++)
    {
        cfxImportNode(j + 1, x[j], y[j], z[j]);
    }

    //        int nodeid[8];

    for (i = 0; i < numElem; i++)
    {
        if (i < numElem - 1)
        {
            nnodes = elem[i + 1] - elem[i];
        }
        else
        {
            nnodes = numConn - elem[i];
        }

        for (n = 0; n < nnodes; n++)
        {
            nodeid[n] = conn[elem[i] + n] + 1;
        }

        cfxImportElement(i + 1, nnodes, nodeid);
    }

    //// regions

    objlist = (unsigned int *)malloc(15000000 * sizeof(int));

    if (mtype == 0 || mtype == 2 || mtype == 5 || mtype == 6) // MachineType = radial, axial, rechenraum or surfacedemo
    {
        // get wall IDs out of wall list
        oldnr = -1;
        for (i = 0; i < numWall; i++)
        {
            nr = wall[colWall * i + 5];

            if (nr != oldnr)
            {
                // potentiell neue Nummer
                // testen, ob Nummer bereits auftaucht
                found = 0;
                for (j = 0; j < numwallregions; j++)
                {
                    if (wallregions[j] == nr)
                    {
                        found = 1;
                        break;
                    }
                }
                if (found == 0)
                {
                    // new nr!
                    wallregions[numwallregions] = nr;
                    numwallregions++;
                    if (numwallregions >= 50)
                    {
                        fprintf(outfp, "maximum number for wall regions exceeded (50). exiting\n");
                        exit(0);
                    }
                }
            }

            oldnr = nr;
        }

        for (i = 0; i < numwallregions; i++)
        {
            nr = wallregions[i];
            wallregionscount[i] = 0;

            sprintf(regname, "wall%d", nr);
            fprintf(outfp, "adding region %s\n", regname);
            cfxImportBegReg(regname, cfxImpREG_FACES);

            for (j = 0; j < numWall; j++)
            {
                if (nr == wall[colWall * j + 5])
                {
                    nodes[0] = wall[colWall * j + 0];
                    nodes[1] = wall[colWall * j + 1];
                    nodes[2] = wall[colWall * j + 2];
                    nodes[3] = wall[colWall * j + 3];

                    lokfacenr = cfxImportFindFace(wall[colWall * j + 4], 4, nodes); //lokale face number
                    //fprintf(stderr,"lokfacenr: %d\n",lokfacenr);
                    globfacenr = cfxImportFaceID(wall[colWall * j + 4], lokfacenr); //globale face number
                    //fprintf(stderr,"globfacenr: %d\n",globfacenr);
                    //testvalue = cfxImportAddReg(1,&globfacenr);
                    //if (testvalue !=1) {fprintf(stderr,"testvalue:%d\n",testvalue);}

                    objlist[wallregionscount[i]] = globfacenr;
                    wallregionscount[i]++;
                    //cfxImportAddReg(4,nodes);
                }
            }
            testvalue = cfxImportAddReg(wallregionscount[i], objlist);
            if (testvalue != 1)
            {
                fprintf(outfp, "testvalue:%d\n", testvalue);
            }
            cfxImportEndReg();
            fprintf(outfp, "wallregion %d: %d\n", wallregions[i], wallregionscount[i]);
        }
    }

    if (mtype == 1 || mtype == 3 || mtype == 4) // MachineType is radial_machine or axial_machine or complete_machine
    {

        // get wall IDs out of wall list
        oldnr = -1;
        for (i = 0; i < numWall; i++)
        {
            nr = wall[colWall * i + 5];
            if (nr != oldnr)
            {
                // potentiell neue Nummer
                // testen, ob Nummer bereits auftaucht
                found = 0;
                for (j = 0; j < numwallregions; j++)
                {
                    if (wallregions[j] == nr)
                    {
                        found = 1;
                        break;
                    }
                }
                if (found == 0)
                {
                    // new nr!
                    wallregions[numwallregions] = nr;
                    numwallregions++;
                    if (numwallregions >= 50)
                    {
                        fprintf(outfp, "maximum number for wall regions exceeded (50). exiting\n");
                        exit(0);
                    }
                }
            }

            oldnr = nr;
        }

        for (i = 0; i < numwallregions; i++)
        {
            nr = wallregions[i];
            wallregionscount[i] = 0;

            sprintf(regname, "wall%d", nr);
            fprintf(outfp, "adding region %s\n", regname);
            cfxImportBegReg(regname, cfxImpREG_FACES);

            for (j = 0; j < numWall; j++)
            {
                if ((nr == wall[colWall * j + 5]))
                {
                    nodes[0] = wall[colWall * j + 0];
                    nodes[1] = wall[colWall * j + 1];
                    nodes[2] = wall[colWall * j + 2];
                    nodes[3] = wall[colWall * j + 3];

                    lokfacenr = cfxImportFindFace(wall[colWall * j + 4], 4, nodes); //lokale face number
                    //fprintf(stderr,"lokfacenr: %d\n",lokfacenr);
                    globfacenr = cfxImportFaceID(wall[colWall * j + 4], lokfacenr); //globale face number
                    //fprintf(stderr,"globfacenr: %d\n",globfacenr);
                    //testvalue = cfxImportAddReg(1,&globfacenr);
                    //if (testvalue !=1) {fprintf(stderr,"testvalue:%d\n",testvalue);}
                    objlist[wallregionscount[i]] = globfacenr;
                    wallregionscount[i]++;
                }
            }
            testvalue = cfxImportAddReg(wallregionscount[i], objlist);
            if (testvalue != 1)
            {
                fprintf(outfp, "testvalue:%d\n", testvalue);
            }
            cfxImportEndReg();
            fprintf(outfp, "wallregionscount[%d]:%d\n", i, wallregionscount[i]);
            //cfxImportRegion(regname,cfxImpREG_FACES,wallregionscount[i],objlist);
            fprintf(outfp, "wallregion %d: %d\n", wallregions[i], wallregionscount[i]);
        }

    } // if mtype is radial_machine, axial_machine, complete_machine

    ///// balances

    if (mtype == 0 || mtype == 2 || mtype == 5 || mtype == 6) // MachineType is radial, axial, rechenraum or surfacedemo
    {
        // get balance IDs out of balance list
        oldnr = -1;
        for (i = 0; i < numBalance; i++)
        {
            nr = balance[colBalance * i + 5];

            if (nr != oldnr)
            {
                // potentiell neue Nummer
                // testen, ob Nummer bereits auftaucht
                found = 0;
                for (j = 0; j < numbalanceregions; j++)
                {
                    if (balanceregions[j] == nr)
                    {
                        found = 1;
                        break;
                    }
                }
                if (found == 0)
                {
                    // new nr!
                    balanceregions[numbalanceregions] = nr;
                    numbalanceregions++;
                    if (numbalanceregions >= 100)
                    {
                        fprintf(outfp, "maximum number for balance regions exceeded (100). exiting\n");
                        exit(0);
                    }
                }
            }

            oldnr = nr;
        }

        for (i = 0; i < numbalanceregions; i++)
        {
            nr = balanceregions[i];
            balanceregionscount[i] = 0;

            sprintf(regname, "balance%d", nr);
            fprintf(outfp, "adding region %s\n", regname);
            cfxImportBegReg(regname, cfxImpREG_FACES);

            for (j = 0; j < numBalance; j++)
            {
                if (nr == balance[colBalance * j + 5])
                {
                    nodes[0] = balance[colBalance * j + 0];
                    nodes[1] = balance[colBalance * j + 1];
                    nodes[2] = balance[colBalance * j + 2];
                    nodes[3] = balance[colBalance * j + 3];

                    /*
if (nr==107)
{
      fprintf(stderr,"balance107 %d: %d %d %d %d %d\n",j,nodes[0]-1,nodes[1]-1,nodes[2]-1,nodes[3]-1,nr);
}
*/

                    lokfacenr = cfxImportFindFace(balance[colBalance * j + 4], 4, nodes); //lokale face number
                    //fprintf(stderr,"lokfacenr: %d\n",lokfacenr);
                    globfacenr = cfxImportFaceID(balance[colBalance * j + 4], lokfacenr); //globale face number
                    //fprintf(stderr,"globfacenr: %d\n",globfacenr);
                    //testvalue = cfxImportAddReg(1,&globfacenr);
                    //if (testvalue !=1) {fprintf(stderr,"testvalue:%d\n",testvalue);}
                    //balanceregionscount[i]++;
                    objlist[balanceregionscount[i]] = globfacenr;
                    balanceregionscount[i]++;

                    //cfxImportAddReg(4,nodes);
                }
            }
            testvalue = cfxImportAddReg(balanceregionscount[i], objlist);
            if (testvalue != 1)
            {
                fprintf(outfp, "testvalue:%d\n", testvalue);
            }
            cfxImportEndReg();
            fprintf(outfp, "balanceregion %d: %d\n", balanceregions[i], balanceregionscount[i]);
        }
    }

    if (mtype == 1 || mtype == 3 || mtype == 4) // MachineType is radial_machine,axial_machine, complete_machine
    {
        // get balance IDs out of balance list
        oldnr = -1;
        for (i = 0; i < numBalance; i++)
        {
            nr = balance[colBalance * i + 5];

            if (nr != oldnr)
            {
                // potentiell neue Nummer
                // testen, ob Nummer bereits auftaucht
                found = 0;
                for (j = 0; j < numbalanceregions; j++)
                {
                    if (balanceregions[j] == nr)
                    {
                        found = 1;
                        break;
                    }
                }
                if (found == 0)
                {
                    // new nr!
                    balanceregions[numbalanceregions] = nr;
                    numbalanceregions++;
                    if (numbalanceregions >= 20)
                    {
                        fprintf(outfp, "maximum number for balance regions exceeded (20). exiting\n");
                        exit(0);
                    }
                }
            }

            oldnr = nr;
        }

        for (i = 0; i < numbalanceregions; i++)
        {
            nr = balanceregions[i];
            balanceregionscount[i] = 0;

            sprintf(regname, "balance%d", nr);
            fprintf(outfp, "adding region %s\n", regname);
            cfxImportBegReg(regname, cfxImpREG_FACES);

            for (j = 0; j < numBalance; j++)
            {
                if ((nr == balance[colBalance * j + 5]))
                {
                    nodes[0] = balance[colBalance * j + 0];
                    nodes[1] = balance[colBalance * j + 1];
                    nodes[2] = balance[colBalance * j + 2];
                    nodes[3] = balance[colBalance * j + 3];

                    lokfacenr = cfxImportFindFace(balance[colBalance * j + 4], 4, nodes); //lokale face number
                    //fprintf(stderr,"lokfacenr %d\n",lokfacenr);
                    globfacenr = cfxImportFaceID(balance[colBalance * j + 4], lokfacenr); //globale face number
                    //fprintf(stderr,"globfacenr %d\n",globfacenr);
                    //testvalue = cfxImportAddReg(1,&globfacenr);
                    //if (testvalue < 1) {fprintf(stderr,"testvalue:%d\n",testvalue);}
                    objlist[balanceregionscount[i]] = globfacenr;
                    balanceregionscount[i]++;
                }
            }
            testvalue = cfxImportAddReg(balanceregionscount[i], objlist);
            if (testvalue < 1)
            {
                fprintf(outfp, "testvalue:%d\n", testvalue);
            }
            cfxImportEndReg();
            fprintf(outfp, "balanceregion %d: %d\n", balanceregions[i], balanceregionscount[i]);
        }

    } // if machinetype is radial_machine, axial_machine,complete_machine

    //// inlet bocos in File schreiben

    if (mtype == 0 || mtype == 2 || mtype == 5 || mtype == 6) // MachineType is radial, axial, rechenraum or surfacedemo
    {
        fprintf(outfp, "inletboco.csv wird geschrieben. \n");

        file = fopen("inletboco.csv", "w");
        fprintf(file, "[Name]\n");
        fprintf(file, "balance100\n");
        fprintf(file, "\n");
        fprintf(file, "[Spatial Fields]\n");
        fprintf(file, "x,y,z\n");
        fprintf(file, "\n");
        fprintf(file, "[Data]\n");
        fprintf(file, "x [ m ], y [ m ], z [ m ], Velocity u [ m s^-1 ], Velocity v [ m s^-1 ], Velocity w [ m s^-1 ]\n");

        //	fprintf(file,"colDiriclet: %i\n",colDiriclet);
        fprintf(outfp, "numDiriclet: %d \n", numDiriclet);

        for (i = 0; i < (numDiriclet / 5); i++)
        {
            fprintf(file, "%3.8f,%3.8f,%3.8f,%8.5f,%8.5f,%8.5f\n", x[diricletIndex[(10 * i)] - 1], y[diricletIndex[(10 * i)] - 1], z[diricletIndex[10 * i] - 1], diricletVal[5 * i], diricletVal[5 * i + 1], diricletVal[5 * i + 2]);
        }
        fclose(file);
        file = NULL;
    }

    if (mtype == 1 || mtype == 3) // MachineType is radial_machine or axial_machine
    {
        k = 0;
        fprintf(outfp, "inletboco.csv wird geschrieben. \n");

        if (mtype == 1)
        {
            sprintf(filename, "inletboco.csv");
        }
        if (mtype == 3)
        {
            sprintf(filename, "inletboco.csv");
        }
        fprintf(outfp, "filename: %s\n", filename);
        fprintf(outfp, "numDiriclet: %d\n", numDiriclet);

        file = fopen(filename, "w");
        fprintf(file, "[Name]\n");
        sprintf(balancename, "balance100");
        fprintf(file, balancename, "\n");
        fprintf(file, "\n");
        fprintf(file, "[Spatial Fields]\n");
        fprintf(file, "x,y,z\n");
        fprintf(file, "\n");
        fprintf(file, "[Data]\n");
        fprintf(file, "x [ m ], y [ m ], z [ m ], Velocity u [ m s^-1 ], Velocity v [ m s^-1 ], Velocity w [ m s^-1 ]\n");
        for (k = 0; k < numblades; k++)
        {
            for (i = 0; i < (numDiriclet / (5 * numblades)); i++) //numDiriclet 2700  ->36
            {

                fprintf(file, "%f,%f,%f,%8.5f,%8.5f,%8.5f\n", x[diricletIndex[10 * i + k * numDiriclet * colDiriclet / (numblades)] - 1], y[diricletIndex[10 * i + k * numDiriclet * colDiriclet / (numblades)] - 1], z[diricletIndex[10 * i + k * numDiriclet * colDiriclet / (numblades)] - 1], diricletVal[5 * i + 0 + k * numDiriclet / (numblades)], diricletVal[5 * i + 1 + k * numDiriclet / (numblades)], diricletVal[5 * i + 2 + k * numDiriclet / (numblades)]);
            }
        }
        fclose(file);
        file = NULL;
    }

    cfxImportDone();

    /* print summary */
    /*
   if (verbose) {
      size_t stats[cfxImpCNT_SIZE];
      long bytes;
      static char *statname[] = {
         "imported nodes    ",
         "imported elements ",
         "imported regions  ",
         "unreferenced nodes",
         "duplicate nodes   ",
         "tet elements      ",
         "pyramid elements  ",
         "wedge elements    ",
         "hex elements      ",
         "total bytes sent  "
      };
      bytes = cfxImportTotals (stats);
      putchar ('\n');
      for (n = 0; n < 9; n++)
         printf ("%s = %ld\n", statname[n], stats[n]);
      printf ("%s = %ld\n", statname[9], bytes);
   }
*/
    free(elem);
    free(conn);

    free(x);
    free(y);
    free(z);

    //	free (press);
    //	free (balance);
    //	free (nodeInfo);
    //	free (elemInfo);
    //	free (diricletIndex);
    //	free (diricletVal);
    //	free (wall);

    fprintf(stderr, "end user import\n");
    fprintf(outfp, "end user import\n");

    coWSAEnd();

    fclose(outfp);

    return (0);
}

/*
#include "stdafx.h"


int _tmain(int argc, _TCHAR* argv[])
{
	return 0;
}
*/
