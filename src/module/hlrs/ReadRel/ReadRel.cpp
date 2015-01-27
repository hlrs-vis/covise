/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 ************************************************************************/

/************************************************************************/
#include <appl/ApplInterface.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include "ReadRel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "reldata.hpp"
#include "Globals.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>

// Überschreiben der Deklaration von WriteText()
int WriteText(VERBOSE, const char *, ...);

// Instanz von GLOBALS erzeugen
GLOBALS Globals;

int main(int argc, char *argv[])
{

    application = new Application(argc, argv);
    application->run();

    return 0;
}

Application::Application(int argc, char *argv[])
{
    Covise::set_module_description("File reader for formatted Fluent(R) files V 0.99b");

    // module parameter filename as a browser
    Covise::add_port(PARIN, "Projectfile", "Browser", "filename");
    Covise::set_port_default("Projectfile", "~/covise/src/application/hlrs/READ_REL/test.nodes *.nodes");

    Covise::add_port(PARIN, "select_data_1", "Choice", "Select Output Data");
    Covise::set_port_default("select_data_1", "1 (none)");
    Covise::add_port(PARIN, "select_data_2", "Choice", "Select Output Data");
    Covise::set_port_default("select_data_2", "1 (none)");
    Covise::add_port(PARIN, "select_data_3", "Choice", "Select Output Data");
    Covise::set_port_default("select_data_3", "1 (none)");
    Covise::add_port(PARIN, "smoothType", "Choice", "Smoothing algorithm");
    Covise::set_port_default("smoothType", "1 None Quality Normal Grid");

    // define module output ports
    //Covise::add_port(OUTPUT_PORT,"grid","StructuredGrid|UnstructuredGrid","grid");
    Covise::add_port(OUTPUT_PORT, "polygons", "Polygons", "geometry polygons");

    Covise::add_port(OUTPUT_PORT, "normals", "Vec3", "normals");

    Covise::add_port(OUTPUT_PORT, "data1", "Float|Vec3", "data1");
    Covise::add_port(OUTPUT_PORT, "data2", "Float|Vec3", "data2");
    Covise::add_port(OUTPUT_PORT, "data3", "Float|Vec3", "data3");

    // covise setup
    Covise::init(argc, argv);

    // define callbacks
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::executeCallback, this);
    Covise::set_param_callback(Application::paramCallback, this);
}

Application::~Application()
{
    // ...
}

void Application::run()
{

    Covise::main_loop();
}

//..........................................................................

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::executeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

void Application::paramCallback(bool /*inMapLoading*/, void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->paramChange(callbackData);
}

void Application::paramChange(void *)
{
    const char *fileName = NULL;
    const char *paramname = Covise::get_reply_param_name();

    if ((strcmp("Projectfile", paramname) == 0) && (Covise::get_reply_browser(&fileName) > 0))
    {
        if (fileName == NULL)
        {
            Covise::sendError("ERROR: filename is NULL");
            return;
        }
    }
}

void Application::compute(void *)
{
    int i; //,n;
    float *x_c, *y_c, *z_c, *u, *v, *w;
    int *vl, *pl; //,*el,*tl;
    coDoPolygons *polygonObject; // output object
    char *polygonObjectName = Covise::get_object_name("polygons");
    char *normalName = Covise::get_object_name("normals");
    char *fileName;
    char realName[1000];

    /*Covise::get_choice_param("select_data_1", &dataSelection[0]);
   Covise::get_choice_param("select_data_2", &dataSelection[1]);
   Covise::get_choice_param("select_data_3", &dataSelection[2]);
   */

    // Instanz erzeugen
    RELDATA RelData;

    // Globals Variablen alle auf null;
    memset(&Globals, 0, sizeof(GLOBALS));

    sprintf(Globals.input, "test");
    sprintf(Globals.project, "output_pro");
    Globals.verbose = VER_MAX;
    Globals.smoothtyp = SM_NONE;
    Globals.smoothruns = 0;
    Globals.writeas = FMT_ANSYS;
    Globals.writeplus = WRT_NORMAL;
    Globals.breakdownto = 3;
    Globals.femtyp = 69; // SHELL69, fuer ANSYS
    Globals.optarc = 0;
    Globals.optsteps = 0;
    Globals.exslices = 0;
    Globals.exsize = 1.0;
    Globals.runs = 1;
    Globals.reindex = FALSE;

    Covise::get_browser_param("Projectfile", &fileName);
    Covise::getname(realName, fileName);
    char *c = strrchr(realName, '.');
    if (c)
    {
        *c = '\0';
    }

    RelData.ReadData(realName);
    RelData.BreakDownFaces(3);
    RelData.ReIndex(ELEM_ALL);
    RelData.CalculatePhongNormals();
    int st;
    Covise::get_choice_param("smoothType", &st);
    switch (st)
    {
    case 2:
        RelData.SurfaceSmooth(SM_QUALITY, 3);
        break;
    case 3:
        RelData.SurfaceSmooth(SM_NORMAL, 3);
        break;
    case 4:
        RelData.SurfaceSmooth(SM_GRID, 3);
        break;
    }
    coDoVec3 *NormalObject;
    NormalObject = new coDoVec3(normalName, RelData.anz_eknoten);
    NormalObject->getAddresses(&u, &v, &w);
    polygonObject = new coDoPolygons(polygonObjectName, RelData.anz_eknoten, RelData.anz_eflaechen * 3, RelData.anz_eflaechen);
    polygonObject->getAddresses(&x_c, &y_c, &z_c, &vl, &pl);
    polygonObject->addAttribute("vertexOrder", "2");
    for (i = 0; i < RelData.anz_eknoten; i++)
    {
        x_c[i] = (float)RelData.eknoten[i].data[X].value;
        y_c[i] = (float)RelData.eknoten[i].data[Y].value;
        z_c[i] = (float)RelData.eknoten[i].data[Z].value;
        u[i] = (float)RelData.eknoten[i].data[RelData.eknoten[i].d_anz - 3].value;
        v[i] = (float)RelData.eknoten[i].data[RelData.eknoten[i].d_anz - 2].value;
        w[i] = (float)RelData.eknoten[i].data[RelData.eknoten[i].d_anz - 1].value;
    }
    for (i = 0; i < RelData.anz_eflaechen; i++)
    {
        pl[i] = i * 3;
        int *nodes = RelData.GetNodes(&RelData.eflaeche[i]);
        vl[i * 3] = nodes[2] - 1;
        vl[i * 3 + 1] = nodes[1] - 1;
        vl[i * 3 + 2] = nodes[0] - 1;
        delete[] nodes;
    }
    delete polygonObject;
    delete NormalObject;
}

void Application::updateChoice()
{
    /*int i;
   const char **choices= new const char *[numVars+2];
   choices[0]="(none)";
   for(i=0;i<numVars;i++)
   {
       if(varIsFace[i])
       {
           if(varTypes[i] < 0)
               choices[i+1]=FluentVecFaceVarNames[-(varTypes[i])];
           else
               choices[i+1]=FluentFaceVarNames[varTypes[i]];
   }
   else
   {
   if(varTypes[i] < 0)
   choices[i+1]=FluentVecVarNames[-(varTypes[i])];
   else
   choices[i+1]=FluentVarNames[varTypes[i]];
   }
   }
   choices[numVars+1]=NULL;
   Covise::update_choice_param("select_data_1",numVars+1,(char **)choices,1);
   Covise::update_choice_param("select_data_2",numVars+1,(char **)choices,1);
   Covise::update_choice_param("select_data_3",numVars+1,(char **)choices,1);
   delete[] choices;*/
}

void Application::quit(void *)
{
}
