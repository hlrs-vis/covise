/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for PATRAN Neutral and Results Files          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Reiner Beller                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  18.07.97  V1.0                                                  **
\**************************************************************************/

#include "ReadFIRE.h"
#include <string.h>

//macros
#define ERR0(cond, text, action)     \
    {                                \
        if (cond)                    \
        {                            \
            Covise::sendError(text); \
            {                        \
                action               \
            }                        \
        }                            \
    }

#define ERR1(cond, text, arg1, action) \
    {                                  \
        if (cond)                      \
        {                              \
            sprintf(buf, text, arg1);  \
            Covise::sendError(buf);    \
            {                          \
                action                 \
            }                          \
        }                              \
    }

#define ERR2(cond, text, arg1, arg2, action) \
    {                                        \
        if (cond)                            \
        {                                    \
            sprintf(buf, text, arg1, arg2);  \
            Covise::sendError(buf);          \
            {                                \
                action                       \
            }                                \
        }                                    \
    }

extern "C" {
void fire2covise(int *nelem, int *nconn, int *ncoord,
                 int **el, int **cl, float **xc, float **yc, float **zc,
                 float **u, float **v, float **w, float **sc,
                 char *filename, char *quantity, int timestep);
}

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
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

/*******************************
 *                             *
 *     D E S T R U C T O R     *
 *                             *
 *******************************/

Application::~Application()
{
}

void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void Application::compute(void *)
{

    // ======================== Input parameters ======================
    coDoUnstructuredGrid *grid;
    //coDoFloat *s_data;
    //coDoVec3 *v_data;
    char *grid_name;
    //char *scalar_name;
    //char *vector_name;
    char *fileset_path;
    char *quantity;
    int time_step, qi;
    int nelem, nconn, ncoord;
    int *el, *cl, i, *tl;
    float *xc, *yc, *zc;
    float *u, *v, *w;
    float *scalar;

    Covise::get_browser_param("file_name", &fileset_path);
    Covise::get_choice_param("quantity", &qi);
    Covise::get_scalar_param("timestep", &time_step);

    switch (qi)
    {
    case 1:
        quantity = "vel";
        break;
    case 2:
        quantity = "pressure";
        break;
    case 3:
        quantity = "temp";
        break;
    case 4:
        quantity = "dens";
        break;
    case 5:
        quantity = "vis";
        break;
    }

    fire2covise(&nelem, &nconn, &ncoord, &el, &cl, &xc, &yc, &zc,
                &u, &v, &w, &scalar,
                fileset_path, quantity, time_step);

    cerr << "nelem: " << nelem << endl;
    cerr << "nconn: " << nconn << endl;
    cerr << "ncoord: " << ncoord << endl;

    if (nelem > 1000000)
        return;

    tl = new int[nelem];
    for (i = 0; i < nelem; i++)
        tl[i] = TYPE_HEXAEDER;

    grid_name = Covise::get_object_name("mesh");
    grid = new coDoUnstructuredGrid(grid_name, nelem, nconn, ncoord,
                                    el, cl, xc, yc, zc, tl);

    delete grid;
}

/*
   if (gridFile) {

     char *Mesh = Covise::get_object_name("mesh");
     ERR0((Mesh==NULL),"Error getting name 'mesh'", return; )
     gridFile->eval_num_connections();

     coDoUnstructuredGrid *mesh =
          new coDoUnstructuredGrid(Mesh,gridFile->num_elements,
                        gridFile->num_connections,
                        gridFile->num_nodes, 1    );
ERR1((mesh==NULL),"Error allocating '%s'",Mesh,return; );
int *clPtr,*tlPtr,*elPtr;
float *xPtr,*yPtr,*zPtr;
mesh->getAddresses(&elPtr,&clPtr,&xPtr,&yPtr,&zPtr);
mesh->getTypeList(&tlPtr);

// Array of IDs
int *id;
char *Type = Covise::get_object_name("type");
ERR0((Type==NULL),"Error getting name 'type'", return; )
int size[2];
size[0] = gridFile->num_elements;
size[1] = 3;  // Element ID
// Property ID
// Component ID
coDoIntArr *type = new coDoIntArr(Type, 2, size);

ERR1((type==NULL),"Error allocating '%s'",Type, return; );
type->getAddress(&id);

// get the mesh

gridFile->getMesh(elPtr,clPtr,tlPtr,xPtr,yPtr,zPtr,
type->getAddress());
delete mesh;
delete type;
}

// ============================= DATA =============================
if (nodalFile) {

// check consistency
/ *
ERR2( (gridFile->num_nodes != nodalFile->nnodes),
"Mesh and Data file not consistent: Nodes %i vs. %i",
gridFile->num_nodes , nodalFile->nnodes,
return; )
* /
/ *     if (gridFile->num_nodes != nodalFile->nnodes)
{
sprintf(buf,"WARNING: Mesh and Data file not consistent: Nodes %i vs. %i",
gridFile->num_nodes , nodalFile->nnodes);
Covise::sendInfo(buf);
}

for (i=0; i<NDATA; i++)
if (fieldNo[i] > 1)
{

int field = fieldNo[i]-1;
// output field name
char objName[16];
sprintf(objName,"data%i",i+1);

char *Name = Covise::get_object_name(objName);
ERR1((Name==NULL),"Error getting name '%s'",objName,return;)

if (field==1) {   // Displacements -> Vector

// missing or additional data ?
int fix = gridFile->num_nodes - nodalFile->nnodes;

coDoVec3 *data
= new coDoVec3(Name,nodalFile->nnodes+fix);
ERR1((data==NULL),"Error allocating '%s'",objName,return; );

float *dx,*dy,*dz;
data->getAddresses(&dx, &dy, &dz);
ERR1 ( (nodalFile->getDataField(field,
gridFile->nodeMap,
dx, dy, dz, fix, gridFile->getMaxnode()) < 0) ,
"Cannot build  Displacements for Port Data%i",i, return;)

delete data;
}

if (field==2) {   // Average Nodal stress -> Scalar

// missing or additional data ?
int fix = gridFile->num_nodes - nodalFile->nnodes;

coDoFloat *data
= new coDoFloat(Name, nodalFile->nnodes+fix);
ERR1((data==NULL),"Error allocating '%s'",objName,return; )

float *S;
data->getAddress(&S);
ERR1 ( (nodalFile->getDataField(field,
gridFile->nodeMap,
colNo[i], S, fix, gridFile->getMaxnode()) < 0) ,
"Cannot read Nodal Stress for Port Data%i",i, return;)

delete data;
}
}
}

if (elemFile) {

// check consistency
/ *
ERR2( (gridFile->num_elements != elemFile->numlines),
"Mesh and Data file not consistent: Elements %i vs. %i",
gridFile->num_elements , elemFile->numlines,
return; )
* /
/ *     if (gridFile->num_elements != elemFile->numlines)
{
sprintf(buf,"WARNING: Mesh and Data file not consistent: Elements %i vs. %i",
gridFile->num_elements , elemFile->numlines);
Covise::sendInfo(buf);
}

for (i=0; i<NDATA; i++)
if (fieldNo[i] > 1)
{
int field = fieldNo[i]-1;
// output field name
char objName[16];
sprintf(objName,"data%i",i+1);

char *Name = Covise::get_object_name(objName);
ERR1((Name==NULL),"Error getting name '%s'",objName,return;)

if (field==3) {   // Element stress -> Scalar

// missing or additional data ?
int fix = gridFile->num_elements - elemFile->numlines;

coDoFloat *data
= new coDoFloat(Name, elemFile->numlines+fix);
ERR1((data==NULL),"Error allocating '%s'",objName,return; )

float *S;
data->getAddress(&S);
ERR1 ( (elemFile->getDataField(field,
gridFile->elemMap,
colNo[i], S, fix, gridFile->getMaxelem()) < 0),
"Cannot read Element Stress for Port Data%i",i, return;)

delete data;
}
}
}
}
*/
