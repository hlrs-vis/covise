/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
 **                                                   	      (C)2000 RUS **
 **                                                                        **
 ** Description:  Trans3D  Simulation Module            	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 **                                                                        **
 ** History:                                                               **
 ** Apr 00         v1                                                      **                               **
 **                                                                        **
\**************************************************************************/
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
#include "error.h"
#include "Trans3D.h"
#include "trans3DInterface.h"
#include <api/coFeedback.h>

extern void read_file(const char *pname);

void CoviseMain(int argc, char *argv[])
{
    Trans3D *application = new Trans3D();
    application->start(argc, argv);
}

Trans3D::Trans3D()
{

    // this info appears in the module setup window
    set_module_description("Read Trans3D(IFSW) Grid and Data in ASCII Format");

    // the output ports
    gridPort = addOutputPort("grid", "coDoStructuredGrid", "Structured Grid");
    TDataPort = addOutputPort("T", "coDoFloat", "Temperature");
    QDataPort = addOutputPort("Q", "coDoFloat", "Q");

    fileParam = addFileBrowserParam("Trans3DInputFile", "Trans3D INPUT File");
    fileParam->setValue("data/ifsw/Trans3D/alufid.in", "*.in");

    iniFileParam = addFileBrowserParam("Trans3DInitFile", "Trans3D Initialization File");
    iniFileParam->setValue("data/ifsw/Trans3D/trinput.fil", "*.fil");

    numVisStepsParam = addInt32Param("numVisSteps", "number of steps to calculate before visualization");
    numVisStepsParam->setValue(5);

    intensitaet = addFloatParam("intensitaet", "Intensitaet des Lasers");
    intensitaet->setValue(1.0);

    radius = addFloatParam("radius", "Radius des Lasers");
    radius->setValue(1.0);

    divergenz = addFloatParam("divergenz", "Divergenz des Lasers");
    divergenz->setValue(4.286000e-02);

    wellenlaenge = addFloatParam("wellenlaenge", "Wellenlaenge des Lasers");
    wellenlaenge->setValue(1.064000e-06);

    strahlradius = addFloatParam("strahlradius", "Strahlradius des Lasers");
    strahlradius->setValue(1.000000e-05);

    fokuslage = addFloatParam("fokuslage", "Fokuslage des Lasers");
    fokuslage->setValue(0.000000e+00);

    laserPos = addFloatVectorParam("laserPos", "Position des Lasers");
    laserPos->setValue(1.0, 1.0, 0.0);

    timestep = 0;
    calculating = false;
}

Trans3D::~Trans3D()
{
}

void Trans3D::quit()
{
}

void Trans3D::updateParameters()
{
    trans3D.setRadius(radius->getValue());
    float x, y, z;
    laserPos->getValue(x, y, z);
    trans3D.setLaserPos(x, y, z);
}

float Trans3D::idle()
{
    if (calculating)
    {

        updateParameters();

        int res = trans3D.Calculate(1);
        if (res == ERR_ABORTED)
        {
        }
        if (res < 0)
        {
            sendError("ERROR: Simulation failed");
            calculating = false;
            return -1;
        }
        if (res == 0)
        {
            if (trans3D.executeScript() == 2) // 2 = TScript::START
            {
                calculating = true;
                return 0;
            }
            else
            {
                calculating = false;
                return -1;
            }
        }
        timestep++;
        if (timestep >= numVisStepsParam->getValue())
        {
            timestep = 0;
            char buf[1000];
            sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(),
                    Covise::get_instance(),
                    Covise::get_host());
            Covise::set_feedback_info(buf);

            // send execute message
            Covise::send_feedback_message("EXEC", "");
            return -1; // wait for the next message (probably the execute)
            // if no, so what...
        }
        return 0; // continue calculation
    }
    else
    {
        return -1; // stop calculation
    }
}

int Trans3D::compute()
{
    char infobuf[500]; // buffer for COVISE info and error messages
    char infobuf2[500]; // buffer for COVISE info and error messages
    static char *oldFile = NULL;
    // get the file name
    filename = fileParam->getValue();
    getname(infobuf, filename);
    filename = infobuf;

    inifilename = iniFileParam->getValue();
    getname(infobuf2, inifilename);
    inifilename = infobuf2;

    // if a new input file was selected and exists
    if ((inifilename[0] != '\0') && (filename[0] != '\0') && ((oldFile == NULL) || (strcmp(oldFile, filename) != 0)))
    {
        delete oldFile;
        oldFile = new char[strlen(filename) + 1];
        strcpy(oldFile, filename);
        trans3D.init(inifilename);
        calculating = false;
        timestep = 0;
        read_file(filename); // read input file
        int res = trans3D.initCalculation();
        if (res == ERR_ABORTED)
        {
        }
        if (res < 0)
        {
            sendError("ERROR: Simulation failed");
            return -1;
        }
        calculating = true;
        trans3D.executeScript();
        float x, y, z;
        trans3D.getLaserPos(x, y, z);
        laserPos->setValue(x, y, z);
    }
    else
    {
        // we calculated something, so generate output objects
        createObjects();
    }

    return SUCCESS;
}

void Trans3D::createObjects()
{
    int xDim = 0, yDim = 0, zDim = 0;
    float *xc, *yc, *zc, *t, *q;
    coDoStructuredGrid *Grid; // output grid
    coDoFloat *T; // Temperature output
    coDoFloat *Q; // Q output
    trans3D.getGridSize(xDim, yDim, zDim);
    Grid = new coDoStructuredGrid(gridPort->getObjName(), xDim, yDim, zDim);
    T = new coDoFloat(TDataPort->getObjName(), xDim, yDim, zDim);
    Q = new coDoFloat(QDataPort->getObjName(), xDim, yDim, zDim);
    gridPort->setCurrentObject(Grid);
    TDataPort->setCurrentObject(T);
    QDataPort->setCurrentObject(Q);
    Grid->getAddresses(&xc, &yc, &zc);
    T->getAddress(&t);
    Q->getAddress(&q);
    int i, j, k;
    for (i = 0; i < xDim; i++)
    {
        for (j = 0; j < yDim; j++)
        {
            for (k = 0; k < zDim; k++)
            {
                trans3D.getValues(i, j, k, xc, yc, zc, t, q);
                xc++;
                yc++;
                zc++;
                t++;
                q++;
            }
        }
    }
    coFeedback feedback("Trans3D");
    feedback.addPara(laserPos);
    feedback.addPara(numVisStepsParam);
    feedback.addPara(radius);
    feedback.addPara(divergenz);
    feedback.addPara(wellenlaenge);
    feedback.addPara(strahlradius);
    feedback.addPara(fokuslage);
    feedback.apply(Grid);
}

/*
void
Trans3D::readFile()
{
    char line[LINE_SIZE];
    coDoStructuredGrid *Grid; 	  // output grid
    coDoFloat *T; 	  // Temperature output
    coDoFloat *Q; 	  // Q output
    xDim=yDim=zDim=0;
    float Raumtemperatur;
    float Verdampfungstemperatur;
while(!feof(fp))
{
fgets(line,LINE_SIZE,fp);
if(strncmp(line,"Raumtemperatur",14)==0)
{
sscanf(line+16,"%f",&Raumtemperatur);
break;
}
}
while(!feof(fp))
{
fgets(line,LINE_SIZE,fp);
if(strncmp(line,"Verdampfungstemperatur",22)==0)
{
sscanf(line+24,"%f",&Verdampfungstemperatur);
break;
}
}
while(!feof(fp))
{
fgets(line,LINE_SIZE,fp);
if(strncmp(line,"x-Gitterpunkte",14)==0)
{
sscanf(line+16,"%d",&xDim);
break;
}
}
while(!feof(fp))
{
fgets(line,LINE_SIZE,fp);
if(strncmp(line,"y-Gitterpunkte",14)==0)
{
sscanf(line+16,"%d",&yDim);
break;
}
}
while(!feof(fp))
{
fgets(line,LINE_SIZE,fp);
if(strncmp(line,"z-Gitterpunkte",14)==0)
{
sscanf(line+16,"%d",&zDim);
break;
}
}
xDim+=2;
yDim+=2;

Grid = new coDoStructuredGrid(gridPort->getObjName(),xDim,yDim,zDim);
T = new coDoFloat(TDataPort->getObjName(),xDim,yDim,zDim);
Q = new coDoFloat(QDataPort->getObjName(),xDim,yDim,zDim);
gridPort->setCurrentObject(Grid);
TDataPort->setCurrentObject(T);
QDataPort->setCurrentObject(Q);
float *xc,*yc,*zc,*t,*q;
Grid->getAddresses(&xc,&yc,&zc);
T->getAddress(&t);
Q->getAddress(&q);

while(!feof(fp))
{
fgets(line,LINE_SIZE,fp);
if(strncmp(line,"Volumengitter:",14)==0)
{
int i,j,k;//,c;
fgets(line,LINE_SIZE,fp); // header
for(i=0;i<xDim;i++)
{
for(j=0;j<yDim;j++)
{
for(k=0;k<zDim;k++)
{
if(fgets(line,LINE_SIZE,fp)==0)
{
sendError("Premature Endo of File");
i=xDim;
j=yDim;
k=zDim;
break;
}
//c=yDim*zDim*i+zDim*j+k;
sscanf(line,"%f %f %f %f %f",xc,yc,zc,t,q);
// Umrechnung von Kirchhoff nach Kelvin
*t = Raumtemperatur + Verdampfungstemperatur * *t;
xc++;yc++;zc++;t++;q++;
}
}
}

}
}

}*/
