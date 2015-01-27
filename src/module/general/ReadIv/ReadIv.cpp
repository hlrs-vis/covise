/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:   COVISE ReadIv application module                        **
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
#include "ReadIv.h"

#include <util/coviseCompat.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

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
    int ofp;
    char *address;
    int length;
    struct stat My_stat_buf;

    err_status = Covise::get_browser_param("path", &Path);
    IvDescr = Covise::get_object_name("descr");

    count++;

#ifdef _WIN32
    if ((ofp = Covise::open(Path, _O_RDONLY)) == -1)
#else
    if ((ofp = Covise::open(Path, O_RDONLY)) == -1)
#endif
    {
        Covise::sendError("ERROR: Can't open file >> ");
        return;
    }

    if (fstat(ofp, &My_stat_buf) < 0)
    {
        Covise::sendError("ERROR: Can't access file :");
        return;
    }
    length = (int)My_stat_buf.st_size;

    if (IvDescr != NULL)
    {
        descr = new coDoText(IvDescr, length);
        if (descr->objectOk())
        {
            descr->getAddress(&address);
            int n = read(ofp, address, length);
            if (n != length)
            {
                fprintf(stderr, "ReadIv: compute(): short read\n");
            }
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
    close(ofp);
}
