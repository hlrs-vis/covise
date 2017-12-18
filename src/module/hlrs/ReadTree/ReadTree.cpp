/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:   COVISE ReadTree application module                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author: D.Rantzau, U. Woessner                                         **
 **                                                                        **
 **                                                                        **
 ** Date:  14.02.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadTree.h"

#include <util/coviseCompat.h>
#include <sys/stat.h>

//
// static stub callback functions calling the real class
// member functions
//

int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();

    return 0;
}

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
    //
    // ...... do work here ........
    //

    // read input parameters and data object name
    char *adress;
    size_t length;
    int doColor = true;
    int ignoreLength = false;

    err_status = Covise::get_browser_param("path", &Path);
    IvDescr = Covise::get_object_name("descr");
    err_status = Covise::get_boolean_param("colors", &doColor);
    err_status = Covise::get_boolean_param("length", &ignoreLength);
    descr = NULL;
    count++;

    char *ivbuf = readTree(Path, ignoreLength, (doColor != 0));

    if (ivbuf == NULL)
    {
        Covise::sendError("ERROR: Can't access file :%s", Path);
        return;
    }
    length = strlen(ivbuf);

    if (IvDescr != NULL)
    {
        descr = new coDoText(IvDescr, (int)length);
        if (descr->objectOk())
        {
            descr->getAddress(&adress);
            memcpy(adress, ivbuf, length);
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'Iv-File'");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: creation of data object 'descr' failed");
        return;
    }

    delete descr;
}
