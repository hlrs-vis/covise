/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ext2SV_h_
#define _ext2SV_h_

#define CELL 67 /* Numeric value of ASCII 'C' */
#define NODE 78 /* Numeric value of ASCII 'N' */
#define XYGRID 1 /* Type of values is 1 layer node values */
#define ELEVGRID 2 /* Type of values is node values for the whole model */
#define ATTRGRID 3 /* Type of values is cell center for the whole model */
#define ACTIVE 4 /* Type of values is cell center for the whole model */

#include <inttypes.h>

struct S_simDATE
{
    uint32_t day; /* Numeric day */
    uint32_t month; /* Numeric month */
    uint32_t year; /* Numeric year */
};

struct S_simHDR
{
    char title[120]; /* Title */
    struct S_simDATE date; /* Data structure */
    char sim_name[15]; /* name of the simulator */
    uint32_t version; /* version number of EXTRACT */
    uint32_t nR; /* Number of rows */
    uint32_t nC; /* Number of columns */
    uint32_t nL; /* Number of layers */
    uint32_t nTS; /* Number of Time steps */
    uint32_t flag; /* UNUSED (Reserved for future use) */
    uint32_t nIprop; /* Number of time independent properties */
    uint32_t nTprop; /* Number of time dependent properties */
    float xOFF; /* X bottom left corner coordinate value */
    float yOFF; /* Y bottom left corner coordinate value */
    float aROT; /* angle of rotation for the model */
    char TS_units[10]; /* Time step unit either YEARS or DAYS */
    float TS_dates[2000]; /* array holding the date for each time step */
};

struct S_simDATA
{
    char title[120]; /* Title */
    uint32_t type; /* type of grid */
    uint32_t nR; /* Number of rows */
    uint32_t nC; /* Number of columns */
    uint32_t nL; /* Number of layers */
    uint32_t nTS; /* Number of Time steps */
    float min; /* data minimum value */
    float max; /* data maximum value */
    uint32_t dataref; /* Data reference CELL = 67, NODE = 78 */
};

typedef struct S_simDATE simDATE;
typedef struct S_simHDR simHDR;
typedef struct S_simDATA simDATA;
#endif
