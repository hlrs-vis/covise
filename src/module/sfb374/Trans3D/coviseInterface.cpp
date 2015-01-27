/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "error.h"
#include "Trans3D.h"
#include "coviseInterface.h"

coviseInterface::coviseInterface()
{
}

coviseInterface::~coviseInterface()
{
}

void coviseInterface::run(int argc, char **argv)
{
    Trans3D *application = new Trans3D();

    application->start(argc, argv);
}

void coviseInterface::error(const char *buf)
{
    Covise::sendError(buf);
}

void coviseInterface::warning(const char *buf)
{
    char *b = new char[strlen(buf) + 20];
    strcpy(b, "WARNING: ");
    strcat(b, buf);
    Covise::sendInfo(b);
    delete[] b;
}

void coviseInterface::info(const char *buf)
{
    Covise::sendInfo(buf);
}

/*int  coviseInterface::checkForMessages()
{
    return Covise::check_and_handle_event();
}*/

coviseInterface covise;
