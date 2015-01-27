/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covise_version.h"
#include <covise_version_autogen.h>
#include <stdio.h>
#include <string.h>

using namespace covise;

static char versionString[1024];
static char longVersionString[1024];

// get the short version string, e.g. "5.2_snap"
// MUST start with Major.Minor, otherwise no FLEXLM licenses
const char *CoviseVersion::shortVersion()
{
    if (strcmp(COVISE_VERSION_HASH, ""))
    {
        sprintf(versionString, "%d.%d-%s",
                COVISE_VERSION_YEAR, COVISE_VERSION_MONTH, COVISE_VERSION_HASH);
    }
    else
    {
        sprintf(versionString, "%d.%d.%d%s",
                COVISE_VERSION_MAJOR, COVISE_VERSION_MINOR, COVISE_VERSION_PATCH,
                COVISE_VERSION_REVISION);
    }
    return versionString;
}

// get the long version string, e.g. "VirCinity Development - July 2001"
const char *CoviseVersion::longVersion()
{
    if (strcmp(COVISE_VERSION_HASH, ""))
    {
        sprintf(longVersionString, "%d.%d-%s (from %s on %s)",
                COVISE_VERSION_YEAR, COVISE_VERSION_MONTH, COVISE_VERSION_HASH,
                COVISE_VERSION_DATE, COVISE_VERSION_ARCH);
    }
    else
    {
        sprintf(longVersionString, "%d.%d.%d (%s from %s on %s)",
                COVISE_VERSION_MAJOR, COVISE_VERSION_MINOR, COVISE_VERSION_PATCH,
                COVISE_VERSION_REVISION, COVISE_VERSION_DATE, COVISE_VERSION_ARCH);
    }
    return longVersionString;
}
