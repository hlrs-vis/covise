/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Fake read for Performer Models         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  12.10.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "PerformerScene.h"
int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();

    return 0;
}

Application::Application(int argc, char *argv[])
    : d_title(NULL)
{
    point = NULL;
    pointName = NULL;
    Covise::set_module_description("Fake read for Performer Models");
    Covise::add_port(OUTPUT_PORT, "model", "Points", "Model");
    Covise::add_port(PARIN, "modelPath", "Browser", "modelPath");
    Covise::set_port_default("modelPath", "data/nofile.i");
    Covise::add_port(PARIN, "modelPath___filter", "BrowserFilter", "modelPath");
    Covise::set_port_default("modelPath___filter", "modelPath *.wrl;*.vrml;*.WRL;*.osg;*.iv;*.obj;*.stl;*.3ds;*");
    // 0.0 keep Scale <0view all > 0 set scale to s
    Covise::add_port(PARIN, "scale", "FloatScalar", "Scale factor");
    Covise::set_port_default("scale", "-1.0");
    Covise::add_port(PARIN, "backface", "Boolean", "Backface Culling");
    Covise::set_port_default("backface", "FALSE");
    Covise::add_port(PARIN, "orientation_iv", "Boolean", "Orientation of iv models like in Inventor Renderer");
    Covise::set_port_default("orientation_iv", "FALSE");
    Covise::add_port(PARIN, "convert_xforms_iv", "Boolean", "create Performer DCS nodes");
    Covise::set_port_default("convert_xforms_iv", "FALSE");

    Covise::init(argc, argv);
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);
    Covise::set_param_callback(Application::paramCallback, this);
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

void Application::paramCallback(bool inMapLoading, void *userData, void *callbackData)
{
    (void)inMapLoading;
    (void)callbackData;

    Application *thisApp = (Application *)userData;
    const char *paramname = Covise::get_reply_param_name();

    // title of module has changed
    if (0 == strcmp(paramname, "SetModuleTitle"))
    {
        const char *title;
        Covise::get_reply_string(&title);
        thisApp->setTitle(title);
    }
}

void Application::setTitle(const char *title)
{
    delete[] d_title;
    d_title = strcpy(new char[strlen(title) + 1], title);
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
    //
    // ...... do work here ........
    //

    float scale;
    char buf[2000], *b;
    int backface;
    int orientation_iv;
    int convertXforms;
    char *fname;

    Covise::get_browser_param("modelPath", &fname);
    Covise::getname(modelPath, fname); // translate relative pathes
    if (/*!modelPath || */ !*modelPath)
    {
        Covise::sendInfo("Covise::getname failed for filename [%s]\n", fname);
        strcpy(modelPath, fname);
    }

    Covise::get_scalar_param("scale", &scale);
    Covise::get_boolean_param("backface", &backface);
    Covise::get_boolean_param("orientation_iv", &orientation_iv);
    Covise::get_boolean_param("convert_xforms_iv", &convertXforms);

    pointName = Covise::get_object_name("model");
    float c = 0.f;
    point = new coDoPoints(pointName, 1, &c, &c, &c);
    if (point->objectOk())
    {
        b = strrchr(modelPath, '/');
        if (b)
        {
            *b = '\0';
            point->addAttribute("MODEL_PATH", modelPath);
            b++;
            Covise::sendInfo("Setting modelPath to [%s]\n", modelPath);
        }
        else
        {
            point->addAttribute("MODEL_PATH", "./");
            b = modelPath;
            Covise::sendInfo("Setting modelPath to [./]\n");
        }
        point->addAttribute("MODEL_FILE", b);
        Covise::sendInfo("Setting modelFile to [%s]\n", b);

        point->addAttribute("COLOR", "black");

        if (backface)
            point->addAttribute("BACKFACE", "ON");
        else
            point->addAttribute("BACKFACE", "OFF");

        if (orientation_iv)
            point->addAttribute("PFIV_CONVERT_ORIENTATION", "ON");
        else
            point->addAttribute("PFIV_CONVERT_ORIENTATION", "OFF");

        if (convertXforms)
            point->addAttribute("PFIV_CONVERT_XFORMS", "ON");
        else
            point->addAttribute("PFIV_CONVERT_XFORMS", "OFF");

        if (scale < 0.0)
        {
            point->addAttribute("SCALE", "viewAll");
        }
        else if (scale == 0.0)
        {
            point->addAttribute("SCALE", "keep");
        }
        else
        {
            sprintf(buf, "%f", scale);
            point->addAttribute("SCALE", buf);
        }
        point->addAttribute("OBJECTNAME", this->getTitle());
    }
    else
    {
        Covise::sendError("ERROR: Could not create Point Object");
        return;
    }
    delete point;
}
