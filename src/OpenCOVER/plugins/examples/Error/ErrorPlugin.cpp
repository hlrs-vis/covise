/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 ZAIK  **
 **                                                                          **
 ** Description: Error Plugin (use HUD to show COVISE errors)                **
 **                                                                          **
 **                                                                          **
 ** Author: Martin Aumueller <aumueller@uni-koeln.de>                        **
 **                                                                          **
\****************************************************************************/

#include "ErrorPlugin.h"
#include <cover/OpenCOVER.h>
#include <cover/coHud.h>
#include <util/unixcompat.h>

ErrorPlugin::ErrorPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

ErrorPlugin::~ErrorPlugin()
{
}

void ErrorPlugin::coviseError(const char *message)
{
    char *msg = strdup(message);

    const char *kind = strsep(&msg, "\n");
    const char *module = strsep(&msg, "\n");
    const char *instance = strsep(&msg, "\n");
    const char *host = strsep(&msg, "\n");
    const char *text = strsep(&msg, "\n");

    if (text && !strcmp(kind, "ERROR"))
    {
        fprintf(stderr, "COVISE Error from %s_%s on %s: %s,\n", module, instance, host, text);
        OpenCOVER::instance()->hud->setText1("COVISE Error");
        std::string who = std::string(module) + "_" + instance + " on " + host;
        OpenCOVER::instance()->hud->setText2(who.c_str());
        OpenCOVER::instance()->hud->setText3(text);
        OpenCOVER::instance()->hud->show();
        OpenCOVER::instance()->hud->redraw();
        OpenCOVER::instance()->hud->hideLater(3.f);
    }
    free(msg);
}

COVERPLUGIN(ErrorPlugin)
