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
 ** Author: U.Woessner                                                     **
 **                                                                        **
\**************************************************************************/

#define BUF_SIZE 32768

#include <stdlib.h>
#include <stdio.h>
#include "ReadVisteon.h"

void main(int argc, char *argv[])
{
    ReadVisteon *application = new ReadVisteon();
    application->start(argc, argv);
}

ReadVisteon::ReadVisteon()
{

    // this info appears in the module setup window
    set_module_description("Reader for Visteon ASCII data");

    // the output port
    polygonPort = addOutputPort("polygons", "coDoPolygons", "geometry polygons");
    polygonPressurePort = addOutputPort("polygonPressure", "coDoFloat", "Pressure on polygons");
    polygonTemperaturePort = addOutputPort("polygonTemperature", "coDoFloat", "Temperature on polygons");
    polygonVisPort = addOutputPort("polygonEddyViscosity", "coDoFloat", "Eddy Viscosity on polygons");
    gridPort = addOutputPort("grid", "coDoUnstructuredGrid", "the Tetrahedra Grid");
    gridVelocityPort = addOutputPort("gridVelocity", "coDoVec3", "Velocity on the Grid");
    gridPressurePort = addOutputPort("gridPressure", "coDoFloat", "Pressure on the Grid");
    gridTemperaturePort = addOutputPort("gridTemperature", "coDoFloat", "Temperature on the Grid");
    gridVisPort = addOutputPort("gridEddyViscosity", "coDoFloat", "Eddy Viscosity on the Grid");

    // the parameters
    crdFileParam = addFileBrowserParam("crdFile", "Coordinate file");
    crdFileParam->setValue("data/visteon/acusol.crd", "*.crd");
    outFileParam = addFileBrowserParam("outFile", "Solution file");
    outFileParam->setValue("data/visteon/peugeot_5_step91.out", "*.out");
    cfgFileParam = addFileBrowserParam("cfgFile", "Config file");
    cfgFileParam->setValue("data/visteon/peugeot_5.cfg", "*.cfg");

    doGrid = addBooleanParam("doGrid", "generate Grid, or not");
    doGrid->setValue(true);
    doSurface = addBooleanParam("doSurface", "generate Surface, or not");
    doSurface->setValue(true);
    doVel = addBooleanParam("doVel", "generate Velocity, or not");
    doVel->setValue(true);
    doP = addBooleanParam("doP", "generate Pessure, or not");
    doP->setValue(true);
    doVis = addBooleanParam("doVis", "generate Viscosity, or not");
    doVis->setValue(true);
    doT = addBooleanParam("doT", "generate Temperature, or not");
    doT->setValue(true);

    //Surfaces = addStringParam("Surfaces","Surfaces to read");
    //Surfaces->setValue("WT_VEHICLE_BOX_G_D_AIR.EBC0 WHEEL_L_B_G-WT_VEHICLE_BOX_G_interface.EBC0");
    //Grids = addStringParam("Grids","Grids to read");
    //Grids->setValue("TURBO_G.con SHRD_DOOR_FPUSH_G.con");

    grids = NULL;
    surfaces = NULL;
    directory = NULL;
}

ReadVisteon::~ReadVisteon()
{
}

void ReadVisteon::quit()
{
}

/*
void ReadVisteon::parseString(const char *string,char **&values,int &num)
{
    int i;
    if(values)
    {
        for(i=0;i<num;i++)
        {
            delete[] values[i];
        }
        delete[] values;
}
if(strlen(string)>0)
num=1;
else
num=0;
i=0;
while(string[i])
{
i++;
if(isspace(string[i]))
{
num++;
while(string[i] && isspace(string[i]))
i++;
}
}
values = new char *[num+1];
values[num]=NULL;
i=0;
int n=0;
while(string[i])
{
int m=0;
while(string[i+m] && !isspace(string[i+m]))
m++;
values[n]=new char[m+1];
strcpy(values[n],string+i);
if(values[n][m])
values[n][m]='\0';
n++;
while(string[i] && !isspace(string[i]))
i++;
while(string[i] && isspace(string[i]))
i++;
}
}*/
// get the number of lines in this file
int ReadVisteon::numLines(const char *filename)
{
    /* int fd = open(filename,O_RDONLY);
    if(fd<=0)
    {
        sprintf(infobuf, "Could not open File %s for reading", filename);
        sendInfo(infobuf);
        return 0;
    }
    char buf[BUF_SIZE];
    int numRead,num=0,i;
    while((numRead=read(fd,buf,BUF_SIZE))>0)
    {
   for(i=0;i<numRead;i++)
   {
   if(buf[i]=='\n')
   num++;
   }
   }
   return num;*/
    int num, n;
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp)
    {
        num = 0;
        while (fgets(buf, LINELENGTH, fp))
        {
            sscanf(buf, "%d", &n);
            if (n > num)
                num = n;
        }
    }
    fclose(fp);
    return n;
}

// get the selected Files
void ReadVisteon::parseCfg(const char *filename)
{
    numSurfaces = 0;
    numGrids = 0;
    delete[] surfaces;
    delete[] grids;
    surfaces = new char *[1000];
    grids = new char *[1000];
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp)
    {
        while (fgets(buf, LINELENGTH, fp))
        {
            if ((strlen(buf) > 4) && (buf[0] != '#'))
            {
                while ((buf[strlen(buf) - 1] == '\n') || (buf[strlen(buf) - 1] == '\r') || (buf[strlen(buf) - 1] == ' '))
                    buf[strlen(buf) - 1] = '\0';
                if (strcasecmp(buf + strlen(buf) - 4, ".con") != 0)
                {
                    surfaces[numSurfaces] = new char[strlen(buf) + 1];
                    strcpy(surfaces[numSurfaces], buf);
                    numSurfaces++;
                }
                else
                {
                    grids[numGrids] = new char[strlen(buf) + 1];
                    strcpy(grids[numGrids], buf);
                    numGrids++;
                }
            }
        }
    }
    fclose(fp);
}

int ReadVisteon::readSurfaces()
{
    int number, num, dummy, n1, n2, n3, *vl, *pl;
    char *filename;
    FILE *fp;
    numPolygons = 0;
    for (number = 0; number < numSurfaces; number++)
    {
        sprintf(infobuf, "Parsing %s", surfaces[number]);
        sendInfo(infobuf);
        filename = new char[strlen(directory) + strlen(surfaces[number]) + 2];
        strcpy(filename, directory);
        strcat(filename, "/");
        strcat(filename, surfaces[number]);
        fp = fopen(filename, "r");
        if (fp)
        {
            num = 0;
            while (fgets(buf, LINELENGTH, fp))
            {
                sscanf(buf, "%d %d %d %d %d", &dummy, &dummy, &n1, &n2, &n3);
                if (polygonCoordNumbers[n1] < 0)
                {
                    polygonCoordNumbers[n1] = numPolygonCoords;
                    numPolygonCoords++;
                }
                if (polygonCoordNumbers[n2] < 0)
                {
                    polygonCoordNumbers[n2] = numPolygonCoords;
                    numPolygonCoords++;
                }
                if (polygonCoordNumbers[n3] < 0)
                {
                    polygonCoordNumbers[n3] = numPolygonCoords;
                    numPolygonCoords++;
                }
                num++;
            }
            numPolygons += num;
            fclose(fp);
        }
        delete[] filename;
    }

    // create the COVISE output object
    coDoPolygons *polygonObject = new coDoPolygons(polygonPort->getObjName(), numPolygonCoords, numPolygons * 3, numPolygons);
    polygonPort->setCurrentObject(polygonObject);
    polygonObject->getAddresses(&x_p, &y_p, &z_p, &vl, &pl);
    polygonObject->addAttribute("vertexOrder", "2");
    //polygonObject->addAttribute("MATERIAL","metal metal.20");

    int numv = 0;
    numPolygons = 0;
    for (number = 0; number < numSurfaces; number++)
    {
        sprintf(infobuf, "Reading %s", surfaces[number]);
        sendInfo(infobuf);
        filename = new char[strlen(directory) + strlen(surfaces[number]) + 2];
        strcpy(filename, directory);
        strcat(filename, "/");
        strcat(filename, surfaces[number]);
        fp = fopen(filename, "r");
        if (fp)
        {
            num = 0;
            while (fgets(buf, LINELENGTH, fp))
            {
                pl[numPolygons + num] = numv;
                sscanf(buf, "%d %d %d %d %d", &dummy, &dummy, &n1, &n2, &n3);
                vl[numv] = polygonCoordNumbers[n1];
                numv++;
                vl[numv] = polygonCoordNumbers[n2];
                numv++;
                vl[numv] = polygonCoordNumbers[n3];
                numv++;
                num++;
            }
            numPolygons += num;
            fclose(fp);
        }
        delete[] filename;
    }

    return SUCCESS;
}

int ReadVisteon::readGrid()
{
    int number, num, dummy, n1, n2, n3, n4, *el, *vl, *tl;
    char *filename;
    FILE *fp;
    numElements = 0;
    for (number = 0; number < numGrids; number++)
    {
        sprintf(infobuf, "Parsing %s", grids[number]);
        sendInfo(infobuf);
        filename = new char[strlen(directory) + strlen(grids[number]) + 2];
        strcpy(filename, directory);
        strcat(filename, "/");
        strcat(filename, grids[number]);
        fp = fopen(filename, "r");
        if (fp)
        {
            num = 0;
            while (fgets(buf, LINELENGTH, fp))
            {
                sscanf(buf, "%d %d %d %d %d", &dummy, &n1, &n2, &n3, &n4);
                if (gridCoordNumbers[n1] < 0)
                {
                    gridCoordNumbers[n1] = numGridCoords;
                    numGridCoords++;
                }
                if (gridCoordNumbers[n2] < 0)
                {
                    gridCoordNumbers[n2] = numGridCoords;
                    numGridCoords++;
                }
                if (gridCoordNumbers[n3] < 0)
                {
                    gridCoordNumbers[n3] = numGridCoords;
                    numGridCoords++;
                }
                if (gridCoordNumbers[n4] < 0)
                {
                    gridCoordNumbers[n4] = numGridCoords;
                    numGridCoords++;
                }
                num++;
            }
            numElements += num;
            fclose(fp);
        }
        delete[] filename;
    }
    // create the COVISE output object
    coDoUnstructuredGrid *gridObject = new coDoUnstructuredGrid(gridPort->getObjName(), numElements, numElements * 4, numGridCoords, 1);
    gridPort->setCurrentObject(gridObject);
    gridObject->getAddresses(&el, &vl, &x_g, &y_g, &z_g);
    gridObject->getTypeList(&tl);

    int numv = 0;
    numElements = 0;
    for (number = 0; number < numGrids; number++)
    {
        sprintf(infobuf, "Reading %s", grids[number]);
        sendInfo(infobuf);
        filename = new char[strlen(directory) + strlen(grids[number]) + 2];
        strcpy(filename, directory);
        strcat(filename, "/");
        strcat(filename, grids[number]);
        fp = fopen(filename, "r");
        if (fp)
        {
            num = 0;
            while (fgets(buf, LINELENGTH, fp))
            {
                el[numElements + num] = numv;
                tl[numElements + num] = TYPE_TETRAHEDER;
                sscanf(buf, "%d %d %d %d %d", &dummy, &n1, &n2, &n3, &n4);
                vl[numv] = gridCoordNumbers[n1];
                numv++;
                vl[numv] = gridCoordNumbers[n2];
                numv++;
                vl[numv] = gridCoordNumbers[n3];
                numv++;
                vl[numv] = gridCoordNumbers[n4];
                numv++;
                num++;
            }
            numElements += num;
            fclose(fp);
        }
        delete[] filename;
    }

    return SUCCESS;
}

int ReadVisteon::compute()
{
    int i;
    // get the file names
    crdFilename = crdFileParam->getValue();
    outFilename = outFileParam->getValue();
    parseCfg(cfgFileParam->getValue());
    if (numSurfaces == 0 && numGrids == 0)
    {
        sendError("ERROR: No Grid or surface files in cfg File");
        return FAIL;
    }
    if (crdFilename && outFilename)
    {
        sendInfo("Parsing Nodes");
        tNumCoords = numLines(crdFilename);
        if (tNumCoords)
        {
            sprintf(infobuf, "Total Number of Coordinates: %d", tNumCoords);
            sendInfo(infobuf);
            polygonCoordNumbers = NULL;
            gridCoordNumbers = NULL;

            delete[] directory;
            directory = new char[strlen(crdFileParam->getValue()) + 1];
            strcpy(directory, crdFileParam->getValue());
            int n = strlen(crdFileParam->getValue());
            while (n >= 0)
            {
                if ((directory[n] == '\\') || (directory[n] == '/'))
                {
                    directory[n] = '\0';
                    break;
                }
                n--;
            }

            if (doGrid->getValue())
            {
                gridCoordNumbers = new int[tNumCoords + 1];
                if (gridCoordNumbers == NULL)
                {
                    sendError("Out of Memory allocating gridCoordNumbers");
                    return FAIL;
                }
                for (i = 0; i < tNumCoords + 1; i++)
                {
                    gridCoordNumbers[i] = -1;
                }
            }
            if (doSurface->getValue())
            {
                polygonCoordNumbers = new int[tNumCoords + 1];
                if (polygonCoordNumbers == NULL)
                {
                    sendError("Out of Memory allocating gridCoordNumbers");
                    delete[] gridCoordNumbers;
                    return FAIL;
                }
                for (i = 0; i < tNumCoords + 1; i++)
                {
                    polygonCoordNumbers[i] = -1;
                }
            }
            numPolygonCoords = 0;
            numGridCoords = 0;

            if (doGrid->getValue())
            {
                //parseString(Grids->getValue(),grids,numGrids);
                sprintf(infobuf, "Reading %d Grids", numGrids);
                sendInfo(infobuf);
                if (readGrid() == FAIL)
                {
                    sendInfo("reading Grids Failed");
                }
            }
            if (doSurface->getValue())
            {
                //parseString(Surfaces->getValue(),surfaces,numSurfaces);
                sprintf(infobuf, "Reading %d Surfaces", numSurfaces);
                sendInfo(infobuf);
                if (readSurfaces() == FAIL)
                {
                    sendInfo("reading Surfaces Failed");
                }
            }
            sendInfo("Reading Nodes");
            if (numPolygonCoords || numGridCoords)
            {
                int cnum;
                float x, y, z;
                FILE *fp = fopen(crdFilename, "r");
                if (fp)
                {
                    while (fgets(buf, LINELENGTH, fp))
                    {
                        sscanf(buf, "%d %f %f %f", &cnum, &x, &y, &z);
                        if (numPolygonCoords)
                        {
                            if (polygonCoordNumbers[cnum] >= 0)
                            {
                                x_p[polygonCoordNumbers[cnum]] = x;
                                y_p[polygonCoordNumbers[cnum]] = y;
                                z_p[polygonCoordNumbers[cnum]] = z;
                            }
                        }
                        if (numGridCoords)
                        {
                            if (gridCoordNumbers[cnum] >= 0)
                            {
                                x_g[gridCoordNumbers[cnum]] = x;
                                y_g[gridCoordNumbers[cnum]] = y;
                                z_g[gridCoordNumbers[cnum]] = z;
                            }
                        }
                    }
                }
                fclose(fp);
            }

            coDoVec3 *VG;
            coDoFloat *PG;
            coDoFloat *TG;
            coDoFloat *VisG;
            coDoFloat *PP;
            coDoFloat *TP;
            coDoFloat *VisP;
            bool readP = false, readV = false, readVis = false, readT = false;
            float *pres, *presG, *visco, *viscoG, *temp, *tempG, *vx, *vy, *vz;

            if (doP->getValue())
            {
                readP = true;
                PG = new coDoFloat(gridPressurePort->getObjName(), numGridCoords);
                PG->getAddress(&presG);
                PP = new coDoFloat(polygonPressurePort->getObjName(), numPolygonCoords);
                PP->getAddress(&pres);
            }

            if (doT->getValue())
            {
                readT = true;
                TG = new coDoFloat(gridTemperaturePort->getObjName(), numGridCoords);
                TG->getAddress(&tempG);
                TP = new coDoFloat(polygonTemperaturePort->getObjName(), numPolygonCoords);
                TP->getAddress(&temp);
            }

            if (doVis->getValue())
            {
                readVis = true;
                VisG = new coDoFloat(gridVisPort->getObjName(), numGridCoords);
                VisG->getAddress(&viscoG);
                VisP = new coDoFloat(polygonVisPort->getObjName(), numPolygonCoords);
                VisP->getAddress(&visco);
            }

            if (doVel->getValue())
            {
                readV = true;
                VG = new coDoVec3(gridVelocityPort->getObjName(), numGridCoords);
                VG->getAddresses(&vx, &vy, &vz);
            }
            if ((numPolygonCoords || numGridCoords) && (readV || readVis || readT || readP))
            {
                sendInfo("Reading Nodal Values");
                int cnum;
                float u, v, w, p, t, vis;
                FILE *fp = fopen(outFilename, "r");
                if (fp)
                {
                    fgets(buf, LINELENGTH, fp);
                    hasTemperature = false;
                    if (sscanf(buf, "%d %f %f %f %f %f %f", &cnum, &u, &v, &w, &p, &t, &vis) == 7)
                        hasTemperature = true;
                    fseek(fp, 0, SEEK_SET);
                    if (hasTemperature)
                    {
                        while (fgets(buf, LINELENGTH, fp))
                        {
                            sscanf(buf, "%d %f %f %f %f %f %f", &cnum, &u, &v, &w, &p, &t, &vis);
                            if (numPolygonCoords)
                            {
                                if (polygonCoordNumbers[cnum] >= 0)
                                {
                                    if (readP)
                                        pres[polygonCoordNumbers[cnum]] = p;
                                    if (readT)
                                        temp[polygonCoordNumbers[cnum]] = t;
                                    if (readVis)
                                        visco[polygonCoordNumbers[cnum]] = vis;
                                }
                            }
                            if (numGridCoords)
                            {
                                if (gridCoordNumbers[cnum] >= 0)
                                {
                                    if (readP)
                                        presG[gridCoordNumbers[cnum]] = p;
                                    if (readT)
                                        tempG[gridCoordNumbers[cnum]] = t;
                                    if (readVis)
                                        viscoG[gridCoordNumbers[cnum]] = vis;
                                    if (readV)
                                    {
                                        vx[gridCoordNumbers[cnum]] = u;
                                        vy[gridCoordNumbers[cnum]] = v;
                                        vz[gridCoordNumbers[cnum]] = w;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        while (fgets(buf, LINELENGTH, fp))
                        {
                            sscanf(buf, "%d %f %f %f %f %f", &cnum, &u, &v, &w, &p, &vis);
                            if (numPolygonCoords)
                            {
                                if (polygonCoordNumbers[cnum] >= 0)
                                {
                                    if (readP)
                                        pres[polygonCoordNumbers[cnum]] = p;
                                    if (readVis)
                                        visco[polygonCoordNumbers[cnum]] = vis;
                                }
                            }
                            if (numGridCoords)
                            {
                                if (gridCoordNumbers[cnum] >= 0)
                                {
                                    if (readP)
                                        presG[gridCoordNumbers[cnum]] = p;
                                    if (readVis)
                                        viscoG[gridCoordNumbers[cnum]] = vis;
                                    if (readV)
                                    {
                                        vx[gridCoordNumbers[cnum]] = u;
                                        vy[gridCoordNumbers[cnum]] = v;
                                        vz[gridCoordNumbers[cnum]] = w;
                                    }
                                }
                            }
                        }
                    }
                    fclose(fp);
                }
                else
                {
                    sprintf(infobuf, "Could not open %s", outFilename);
                    sendInfo(infobuf);
                }
            }

            delete[] polygonCoordNumbers;
            delete[] gridCoordNumbers;
        }
        else
        {
            sendInfo("Could not read Coordinates");
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
