/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_VERSION_H
#define COVISE_VERSION_H

#include <util/coTypes.h>

#define NET_FILE_VERERSION 632

namespace covise
{

class COVISEEXPORT CoviseVersion
{
public:
    // get the short version string, e.g. "VIR_SNAP-2001-01-F"
    static const char *shortVersion();

    // get the long version string, e.g. "Vircinity Development - January 2001"
    static const char *longVersion();
};
}
#endif
