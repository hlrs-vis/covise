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
    snprintf(versionString, sizeof(versionString), "%d.%d-%s", COVISE_VERSION_YEAR, COVISE_VERSION_MONTH,
             COVISE_VERSION_HASH);
    return versionString;
}

// get the long version string, e.g. "VirCinity Development - July 2001"
const char *CoviseVersion::longVersion()
{
    snprintf(longVersionString, sizeof(longVersionString), "%d.%d-%s (from %s on %s)", COVISE_VERSION_YEAR,
             COVISE_VERSION_MONTH, COVISE_VERSION_HASH, COVISE_VERSION_DATE, ARCHSUFFIX);
    return longVersionString;
}

const char *CoviseVersion::hash()
{
    return COVISE_VERSION_HASH;
}

const char *CoviseVersion::compileDate()
{
    return COVISE_VERSION_DATE;
}

const char *CoviseVersion::arch()
{
    return ARCHSUFFIX;
}

int CoviseVersion::year()
{
    return COVISE_VERSION_YEAR;
}

int CoviseVersion::month()
{
    return COVISE_VERSION_MONTH;
}
