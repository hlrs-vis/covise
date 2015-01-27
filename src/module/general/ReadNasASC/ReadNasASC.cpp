/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
**                                                           (C)1994 RUS  **
**                                                                        **
** Description:   COVISE ReadPlot3D application module                    **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                             (C) 1994                                   **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **   **                                                                        **
**                                                                        **
** Author:  Andreas Wierse                                                **
**                                                                        **
**                                                                        **
** Date:  17.06.99  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>
#include "ReadNasASC.h"

int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();

    return 0;
}

//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//
//
//..........................................................................
//
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void Application::compute(void *)
{
    char *filename, *Mesh, *Temp, *Mach, *Press, *Vel, *Poly;
    int i;
    float *x, *y, *z;
    float *xVelData, *yVelData, *zVelData;
    float *pressureData, *machData, *tempData;
    int *el, *cl, *tl;
    coDoUnstructuredGrid *grid;
    coDoPolygons *mesh;
    coDoVec3 *vel;
    coDoFloat *mach;
    coDoFloat *press;
    coDoFloat *temp;

    // read input parameters and data object name

    Covise::get_browser_param("path", &filename);
    Mesh = Covise::get_object_name("mesh");
    Poly = Covise::get_object_name("poly");
    Press = Covise::get_object_name("pressure");
    Temp = Covise::get_object_name("temp");
    Mach = Covise::get_object_name("mach");
    Vel = Covise::get_object_name("velocity");
    if (Mesh == NULL)
    {
        Covise::sendError("ERROR: object name not correct for 'mesh'");
        return;
    }
    if (Press == NULL || Mesh == NULL || Temp == NULL || Vel == NULL)
    {
        Covise::sendError("ERROR: object name not correct for 'data'");
        return;
    }

    if ((fp = Covise::fopen(filename, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s", filename);
        return;
    }

    read_length();
    if (numTetras)
    {

        grid = new coDoUnstructuredGrid(Mesh, numTetras, numTetras * 4,
                                        numVertices, 1);
        if (!grid->objectOk())
        {
            cerr << "problem creating object\n";
            return;
        }

        grid->getAddresses(&el, &cl, &x, &y, &z);

        grid->getTypeList(&tl);
        for (i = 0; i < numTetras; i++)
            tl[i] = TYPE_TETRAHEDER;

        read_vertices(x, y, z);

        read_tetras(el, cl);
    }
    else if (numTris)
    {
        mesh = new coDoPolygons(Poly, numVertices, numTris * 3, numTris);
        if (!mesh->objectOk())
        {
            cerr << "problem creating object\n";
            return;
        }

        mesh->getAddresses(&x, &y, &z, &cl, &el);
        read_vertices2(x, y, z);
        read_tris(el, cl);
    }
    if (numTetras)
    {
        press = new coDoFloat(Press, numVertices);
        if (!press->objectOk())
        {
            cerr << "problem creating object\n";
            return;
        }

        press->getAddress(&pressureData);

        read_scalar(pressureData);

        vel = new coDoVec3(Vel, numVertices);
        if (!vel->objectOk())
        {
            cerr << "problem creating object\n";
            return;
        }

        vel->getAddresses(&xVelData, &yVelData, &zVelData);

        read_vector(xVelData, yVelData, zVelData);

        mach = new coDoFloat(Mach, numVertices);
        if (!mach->objectOk())
        {
            cerr << "problem creating object\n";
            return;
        }

        mach->getAddress(&machData);

        read_scalar(machData);

        temp = new coDoFloat(Temp, numVertices);
        if (!temp->objectOk())
        {
            cerr << "problem creating object\n";
            return;
        }

        temp->getAddress(&tempData);

        read_scalar(tempData);
    }
}

int Application::read_vertices2(float *x, float *y, float *z)
{
    int i;
    char line[500];

    for (i = 0; i < numVertices; i++)
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_1 failed in ReadNasASC.cpp");
        }

        sscanf(&line[40], "%f", &z[i]);
        line[40] = 0;
        sscanf(&line[32], "%f", &y[i]);
        line[32] = 0;
        sscanf(&line[24], "%f", &x[i]);
    }

    return i;
}
int Application::read_vertices(float *x, float *y, float *z)
{
    int i, j;
    int dummy, count;
    char line[500];

    for (i = 0; i < numVertices; i++)
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_1 failed in ReadNasASC.cpp");
        }
        j = 9;
        while (line[j] != '*' && line[j] != 0)
            j++;
        line[j] = 0;
        sscanf(&line[9], "%d %d %f %f", &count, &dummy, &x[i], &y[i]);
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_2 failed in ReadNasASC.cpp");
        }
        sscanf(&line[9], "%f", &z[i]);
        //		cerr << "["<<count<<"] "<< x[i] << ", " << y[i] << ", " << z[i] << endl;
    }

    return i;
}

int Application::read_tetras(int *el, int *cl)
{
    int i;
    int dummy, count;
    char line[500];

    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_3 failed in ReadNasASC.cpp");
    }
    for (i = 0; i < numTetras; i++)
    {
        el[i] = 4 * i;
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_4 failed in ReadNasASC.cpp");
        }
        sscanf(&line[7], "%d %d %d %d %d %d", &count, &dummy,
               &cl[4 * i + 1], &cl[4 * i + 0], &cl[4 * i + 2], &cl[4 * i + 3]);
        cl[4 * i + 0]--;
        cl[4 * i + 1]--;
        cl[4 * i + 2]--;
        cl[4 * i + 3]--;
        //		cerr << "["<<count<<"] "<< x[i] << ", " << y[i] << ", " << z[i] << endl;
    }

    return i;
}

int Application::read_tris(int *el, int *cl)
{
    int i;
    int dummy, count;
    char line[500];

    for (i = 0; i < numTris; i++)
    {
        el[i] = 3 * i;
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_4 failed in ReadNasASC.cpp");
        }
        sscanf(&line[7], "%d %d %d %d %d", &count, &dummy,
               &cl[3 * i + 0], &cl[3 * i + 1], &cl[3 * i + 2]);
        cl[3 * i + 0]--;
        cl[3 * i + 1]--;
        cl[3 * i + 2]--;
        //		cerr << "["<<count<<"] "<< x[i] << ", " << y[i] << ", " << z[i] << endl;
    }

    return i;
}

int Application::read_scalar(float *data)
{
    int i;
    int dummy, count;
    char line[500];

    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_5 failed in ReadNasASC.cpp");
    }
    Covise::sendInfo("Reading %s", &line[12]);
    for (i = 0; i < numVertices; i++)
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_6 failed in ReadNasASC.cpp");
        }
        sscanf(&line[7], "%d %d %f", &count, &dummy, &data[i]);
        //		cerr << "["<<count<<"] "<< x[i] << ", " << y[i] << ", " << z[i] << endl;
    }

    return i;
}

int Application::read_vector(float *x, float *y, float *z)
{
    int i;
    int dummy, count;
    char line[500];

    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_7 failed in ReadNasASC.cpp");
    }
    Covise::sendInfo("Reading %s", &line[12]);
    for (i = 0; i < numVertices; i++)
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_8 failed in ReadNasASC.cpp");
        }
        sscanf(&line[7], "%d %d %f", &count, &dummy, &x[i]);
    }

    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_9 failed in ReadNasASC.cpp");
    }
    Covise::sendInfo("Reading %s", &line[12]);
    for (i = 0; i < numVertices; i++)
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_10 failed in ReadNasASC.cpp");
        }
        sscanf(&line[7], "%d %d %f", &count, &dummy, &y[i]);
    }

    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_11 failed in ReadNasASC.cpp");
    }
    Covise::sendInfo("Reading %s", &line[12]);
    for (i = 0; i < numVertices; i++)
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_12 failed in ReadNasASC.cpp");
        }
        sscanf(&line[7], "%d %d %f", &count, &dummy, &z[i]);
    }

    return i;
}

int Application::read_length()
{
    char line[500];
    int i;

    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_13 failed in ReadNasASC.cpp");
    }
    if (strncmp(line, "$ File generated by InnovMetric", 30) == 0)
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "vertices failed in ReadNasASC.cpp");
        }
        //sscanf(line,"%d vertices, %d",
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "emptyLine failed in ReadNasASC.cpp");
        }
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "BEGIN BULK failed in ReadNasASC.cpp");
        }
    }
    else
    {
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_14 failed in ReadNasASC.cpp");
        }
        i = 0;
        while (line[i] != '=')
            i++; // skip white spaces
        Covise::sendInfo("Reading %s", &line[i + 2]);
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_15 failed in ReadNasASC.cpp");
        }
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_16 failed in ReadNasASC.cpp");
        }
        if (strncmp(line, "$ GRID POINTS", 13))
        {
            Covise::sendInfo("No Grid Points in Data Set");
            return 0;
        }
    }

    long start_grid = ftell(fp);

    numVertices = 0;
    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_17 failed in ReadNasASC.cpp");
    }
    while (strncmp(line, "GRID", 4) == 0)
    {
        numVertices++;
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_18 failed in ReadNasASC.cpp");
        }
        /*if (fgets(line,500,fp)==NULL)
      {
         fprintf(stderr,"fgets_19 failed in ReadNasASC.cpp");
      }*/
    }

    Covise::sendInfo("%d Gridpoints", numVertices);

    if (strncmp(line, "$ ELEMENTS", 10))
    {
        Covise::sendInfo("No $ Elements in Data Set");
    }

    numTetras = 0;
    if (fgets(line, 500, fp) == NULL)
    {
        fprintf(stderr, "fgets_20 failed in ReadNasASC.cpp");
    }
    while (strncmp(line, "CTETRA", 5) == 0)
    {
        numTetras++;
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_21 failed in ReadNasASC.cpp");
        }
    }

    numTris = 0;
    if (strncmp(line, "CTRIA3", 5) == 0)
        numTris++;
    while (strncmp(line, "CTRIA3", 6) == 0)
    {
        numTris++;
        if (fgets(line, 500, fp) == NULL)
        {
            fprintf(stderr, "fgets_22 failed in ReadNasASC.cpp");
        }
    }

    Covise::sendInfo("%d Elements %d Triangles", numTetras, numTris);

    fseek(fp, start_grid, SEEK_SET);

    return numVertices;
}

/*
int Application::read_file(coDistributedObject **set,char *objname)
{
coDoStructuredGrid *mesh;
float *x_coord,*y_coord,*z_coord;
int idim=1,jdim=1,kdim=1,i,j,k,n,tmpi=0;
char line[500];
char *color=NULL;
while(!feof(fp))
{
if (fgets(line,500,fp)!=NULL)
{
fprintf(stderr,"fgets_22 failed in ReadNasASC.cpp");
}
i=0;
while((line[i]==' ')||(line[i]=='\t'))
i++; // skip white spaces
if(strncasecmp(line+i,"zone",4)==0)
{
n=i+4;
while((line[n]!='=')&&(line[n]!='\0'))
n++; // skip to '='
sscanf(line+n+3,"%d",&tmpi);
if(tmpi>0)
color="brown";
n++;
while(line[n]!='\0')
{
while((line[n]!='=')&&(line[n]!='\0'))
n++; // skip to '='
if(line[n]=='\0')
break;
if((line[n-2]=='i')||(line[n-2]=='I'))
sscanf(line+n+1,"%d",&idim);
if((line[n-2]=='j')||(line[n-2]=='J'))
sscanf(line+n+1,"%d",&jdim);
if((line[n-2]=='k')||(line[n-2]=='K'))
sscanf(line+n+1,"%d",&kdim);
n++;
}
break;
}
else if(strncasecmp(line+i,"Title",5)==0)
{
Covise::sendInfo(line+i+5);
}
}
if(feof(fp))
return 0;
mesh = new coDoStructuredGrid(objname, idim, jdim, kdim);
if (mesh->objectOk())
{
if(color)
mesh->addAttribute("COLOR",color);
mesh->getAddresses(&x_coord,&y_coord,&z_coord);
for(k=0;k<kdim;k++)
{
for(j=0;j<jdim;j++)
{
for(i=0;i<idim;i++)
{
if (fgets(line,500,fp)!=NULL)
{
fprintf(stderr,"fgets_23 failed in ReadNasASC.cpp");
}
n=0;
while(line[n])
{
if(line[n]=='D')
line[n]='E';
n++;
}

sscanf(line,"%f %f %f",x_coord+(i*kdim*jdim+j*kdim+k),y_coord+(i*kdim*jdim+j*kdim+k),z_coord+(i*kdim*jdim+j*kdim+k));
}
}
}
i=0;
while(set[i])
i++;
set[i]=mesh;
set[i+1]=NULL;
return 1;
}
else
{
Covise::sendError("ERROR: creation of data object 'mesh' failed");
return 0;
}
}

*/
