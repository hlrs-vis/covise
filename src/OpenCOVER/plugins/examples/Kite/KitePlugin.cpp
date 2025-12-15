/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KitePlugin.h"
#include <cover/coVRPluginSupport.h>

using namespace opencover;

KitePlugin::KitePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

KitePlugin::~KitePlugin()
{
}

bool KitePlugin::init()
{
    return true;
}

COVERPLUGIN(KitePlugin)
