/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ====================================================================
// Binärer Header für alle ANSYS Ergebnisfiles
// Quelle: Guide to interacing with ANSYS Ver. 5.6
// alle integer haben die Länge 32 bit.
// alle chars haben die Länge 8 bit.
// Länge über alles: 100 x integer = 400 Bytes
// 17.10.2001 Björn Sander
// ====================================================================
#ifndef __BIN_HEADER_HPP__
#define __BIN_HEADER_HPP__

#include <util/coviseCompat.h>

struct BINHEADER
{
    int filenum; // File Number
    int format; // File Format: 0=internal, 1=external
    int time; // Time in compact(?) form
    int date; // Date in compact(?) form
    //  int res1[4];          // ??
    int unit; // 0=user def, 1=SI, 2=CSG, 3=feet, 4=inches
    int version; // Version of ANSYS, calculating this result
    int ansysdate; // Release date of ANSYS calculating this result
    char machine[12]; // Computer Identifier calculating this result
    char jobname[8]; // any questions?
    char product[8]; // Product name of ANSYS
    char label[4]; // ANSYS special version label string
    char user[12]; // user name
    char machine2[12]; // again, machine identifier (?)
    int recordsize; // System record size
    int maxfilelen; // the maximum file length
    int maxrecnum; // The maximun rcord number
    int cpus; // number of processors used for this task
    char title[80]; // main title of analysis
    char subtitle[80]; // subtitle of analysis

    // Initialisierung:
    BINHEADER(void)
    {
        memset(this, 0, sizeof(BINHEADER));
    }
};
#endif
