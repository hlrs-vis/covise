/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __STAR_FILE_H_
#define __STAR_FILE_H_
#include <util/ChoiceList.h>

#include "util/coTypes.h"

namespace covise
{

class STAREXPORT StarFile
{
private:
    StarFile(const StarFile &);
    StarFile &operator=(const StarFile &);

public:
    enum // 1
    {
        VELOCITY = 1,
        VMAG, // 2
        U,
        V,
        W, // 3 4 5
        PRESSURE, // 6
        TE,
        ED, // 7 8
        TVIS,
        TEMPERATURE,
        DENSITY, // 9 10 11
        LAMVIS,
        CP,
        COND, // 12 13 14
        DROP_COORD,
        DROP_VEL, // 15 16
        DROP_DENS,
        DROP_DIAM, // 17 18
        DROP_TEMP,
        DROP_NO, // 19 20
        DROP_MASS, // 21
        SCALAR // 22
    };

    StarFile(){};
    virtual ~StarFile(){};

    virtual ChoiceList *get_choice(const char **, int) const = 0;

    char *secure_strdup(const char *string);
    char *secure_strcat(char *s1, const char *s2);
    char *secure_strcpy(char *s1, const char *s2);
    char *secure_strncpy(char *s1, const char *s2, int n);
    int secure_strcmp(const char *s1, const char *s2);
};

class STAREXPORT StarModelFile
{
public:
    struct CellTabEntry
    {
        int vertex[8];
        int ictID;
    };

    struct SammTabEntry
    {
        int vertex[12];
        int ictID;
    };

    struct BounTabEntry
    {
        int vertex[4];
        int region, patch;
    };

    struct VertexTabEntry
    {
        float coord[3];
    };

    struct CellTypeEntry
    {
        int ctype, colorIdx, poroIdx, matIdx, spnIdx, grpIdx, dummy[4];
    };

    struct RegionSize
    {
        int numPoly, numTria;
    };
};
}
#endif
