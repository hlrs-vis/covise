/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for VECTIS Files                              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Wierse                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  30.07.98  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadVectisTri.h"
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

void
Application::paramCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->paramChange(callbackData);
}

/*********************************
 *                               *
 *     C O N S T R U C T O R     *
 *                               *
 *********************************/

Application::Application(int argc, char *argv[])
{
    vect_file = 0L;
    file_name = 0L;

    Covise::set_module_description("Read VECTIS Triangle Files");

    // File Name
    Covise::add_port(PARIN, "file_name", "Browser", "File path");
    Covise::set_port_default("file_name", "./*.*");
    Covise::set_port_immediate("file_name", 1);

    // Output
    Covise::add_port(OUTPUT_PORT, "surface", "DO_Polygon", "Patch Output");

    // Do the setup
    Covise::init(argc, argv);
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);
    Covise::set_param_callback(Application::paramCallback, this);

    // Set internal object pointers to Files and Filenames
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

void Application::paramChange(void *)
{

    const char *tmp;
    char *pname, *new_file_name;

    // get watchdir parameter
    pname = Covise::get_reply_param_name();

    if (strcmp("file_name", pname) == 0)
    {
        Covise::get_reply_browser(&tmp);

        new_file_name = (char *)new char[strlen(tmp) + 1];
        strcpy(new_file_name, tmp);

        if (new_file_name != NULL)
        {
            if (file_name == 0 || strcmp(file_name, new_file_name) != 0)
            {
                delete file_name;
                file_name = new_file_name;
            }
        }
        else
        {
            Covise::sendError("ERROR:file_name is NULL");
        }
    }
}

void Application::compute(void *)
{

    // ======================== Input parameters ======================

    char *new_file_name, *tmp_name, buf[256];
    char *poly_out;
    int i;
    float *x, *y, *z;
    int *vl, *pl, tmpi;
    coDoPolygons *polygons_out;
    FILE *vect_file;

    Covise::get_browser_param("file_name", &tmp_name);
    new_file_name = (char *)new char[strlen(tmp_name) + 1];
    strcpy(new_file_name, tmp_name);
    poly_out = Covise::get_object_name("surface");

    if (new_file_name != NULL)
    {
        if (file_name == 0L || strcmp(file_name, new_file_name) != 0)
        {
            delete file_name;
            file_name = new_file_name;
        }
    }
    else
    {
        Covise::sendError("ERROR:file_name is NULL");
    }

    vect_file = Covise::fopen(file_name, "r");
    ERR1(vect_file == NULL, "fopen for %s failed", file_name, return;);

    //	fscanf(vect_file,"%s",buf);
    //	cout<<buf<<endl;
    while ((tmpi = fgetc(vect_file)) != '\n')
        cout << tmpi << endl;
    while ((tmpi = fgetc(vect_file)) != '\n')
        cout << tmpi << endl;
    fscanf(vect_file, "%d", &num_points);
    cout << "number of points: " << num_points << endl;
    x = new float[num_points];
    y = new float[num_points];
    z = new float[num_points];
    for (i = 0; i < num_points; i++)
        fscanf(vect_file, "%f %f %f", &x[i], &y[i], &z[i]);

    fscanf(vect_file, "%d", &num_triangles);
    cout << "number of triangles: " << num_triangles << endl;
    vl = new int[num_triangles * 3];
    pl = new int[num_triangles];
    for (i = 0; i < num_triangles; i++)
    {
        fscanf(vect_file, "%d", &vl[3 * i + 0]);
        vl[3 * i + 0]--;
        fscanf(vect_file, "%d", &vl[3 * i + 1]);
        vl[3 * i + 1]--;
        fscanf(vect_file, "%d", &vl[3 * i + 2]);
        vl[3 * i + 2]--;
        pl[i] = 3 * i;
    }

    polygons_out = new coDoPolygons(poly_out, num_points, x, y, z,
                                    num_triangles * 3, vl, num_triangles, pl);

    if (polygons_out->objectOk())
    {
        polygons_out->addAttribute("MATERIAL", "metal metal.30");
        polygons_out->addAttribute("vertexOrder", "1");
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] vl;
        delete[] pl;
        delete polygons_out;
    }
    else
    {
        ERR0(1, "new coDoPolygons failed", return;);
    }
}
