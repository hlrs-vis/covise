/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BLANK_CONV_H
#define CO_BLANK_CONV_H

#include <util/coTypes.h>

// 16.09.99

/**
 * Class to supply utility functions mapping blanks to char(255)
 *
 */
namespace covise
{

class APIEXPORT coBlankConv
{

public:
    // Allocate a new char [] of same size and replace all blanks by char(255)
    // convert empty string to single char(1) to prevent bugs
    static char *all(const char *inString);

    // dito, but convert only blanks inside '...' apostrophies
    static char *escaped(const char *inString);
};
}
#endif
