/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coBlankConv.h"
#include <string.h>

using namespace covise;

// ---------------------- Utility functions -----------------------------

// Allocate a new char [] of same size and replace all blanks by char(255)
char *coBlankConv::all(const char *inString)
{
    // make sure this is a (char*) for old Covise bugs
    // prevent other bugs by sending a single \001 for empty strings

    if (!inString)
        return NULL;

    int length = int(strlen(inString));
    if (length == 0)
    {
        static const char empty[2] = { 1, 0 };
        return strcpy(new char[2], empty);
    }
    char *outString = new char[1 + length];

    // copy, but change all blanks in '...' parts by \177 (char 255)
    char *oPtr = outString;
    const char *vPtr = inString;
    while (*vPtr)
    {
        if (*vPtr == ' ')
            *oPtr = '\177';
        else
            *oPtr = *vPtr;
        vPtr++;
        oPtr++;
    }
    *oPtr = '\0';

    return outString;
}

// dito, but convert only blanks inside '...' apostrophies
char *coBlankConv::escaped(const char *inString)
{
    // make sure this is a (char*) for old Covise bugs
    char *outString = new char[strlen(inString) + 1];

    // copy, but change all blanks in '...' parts by \177 (char 255)
    char *oPtr = outString;
    const char *vPtr = inString;
    int masked = 0;
    while (*vPtr)
    {
        switch (*vPtr)
        {
        case '\'':
            masked = !masked;
            break;

        case ' ':
            if (masked)
                *oPtr = '\177';
            else
                *oPtr = ' ';
            oPtr++;
            break;

        default:
            *oPtr = *vPtr;
            oPtr++;
            break;
        }
        vPtr++;
    }
    *oPtr = '\0';

    return outString;
}
