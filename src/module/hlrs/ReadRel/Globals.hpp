/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GLOBALS__
#define __GLOBALS__

struct GLOBALS
{
    int smoothtyp;
    int smoothruns;
    unsigned int writeas;
    unsigned int writeplus;
    unsigned int femtyp;
    int exslices;
    double exsize; // Extrude
    double optarc;
    int optsteps;
    int breakdownto;
    int verbose;
    int runs;
    unsigned int reindex;
    char project[256];
    char input[256];
};

extern struct GLOBALS Globals;

enum WRITEPLUS
{
    WRT_NORMAL,
    WRT_SMOOTH,
    ANZ_WRITEPLUS
};

#endif
