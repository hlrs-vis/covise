/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_VERSION_H
#define COVISE_VERSION_H

#include <util/coTypes.h>

// update, when .net file format changes
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

    //! get the hash of the latest git commit
    static const char *hash();

    //! get the compilation date
    static const char *compileDate();

    //! get the ARCHSUFFIX
    static const char *arch();

    //! get the year of the last git commit
    static int year();

    //! get the month of the last git commit
    static int month();
};
}
#endif
