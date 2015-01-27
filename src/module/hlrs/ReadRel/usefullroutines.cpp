/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "usefullroutines.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#ifndef _WIN32
int strnicmp(const char *string1, const char *string2, int n)
{
    int len;
    char str1[1024], str2[1024];
    int x;

    // lokale Kopien anfertigen
    sprintf(str1, "%s", string1);
    sprintf(str2, "%s", string2);

    // untersuchte Länge ist n oder die kleinste Stringlänge, falls n größer als diese ist
    len = MIN(MIN(strlen(str1), strlen(str2)), n);

    // jetzt erst mal alle auf gleichen Fall (Groß/kleinschreibung)
    for (x = 0; x < len; ++x)
    {
        str1[x] = toupper(str1[x]);
        str2[x] = toupper(str2[x]);
    }

    // und normales Stringcompare zurück
    return (strncmp(str1, str2, len));
}

int stricmp(const char *string1, const char *string2)
{
    int len;
    char str1[1024], str2[1024];
    int x;

    // lokale Kopien anfertigen
    sprintf(str1, "%s", string1);
    sprintf(str2, "%s", string2);

    // untersuchte Länge ist die kleinste Stringlänge
    len = MIN(strlen(str1), strlen(str2));

    // jetzt erst mal alle auf gleichen Fall (Groß/kleinschreibung)
    for (x = 0; x < len; ++x)
    {
        str1[x] = toupper(str1[x]);
        str2[x] = toupper(str2[x]);
    }

    // und normales Stringcompare zurück
    // wenn die Strings unterschiedlich lang sind, liefert strcmp
    // sowieso ungleich.
    return (strcmp(str1, str2));
}
#endif
// Entdeckt, ob Zeichen zu einer ASCII-Float-Zahl gehört
// Float Format z.B. -96.345E+12
int isfloat(char c)
{
    if (isdigit(c))
        return (1);
    if (c == 'e' || c == 'E')
        return (1);
    if (c == '-' || c == '+' || c == '.')
        return (1);
    return (0);
}
