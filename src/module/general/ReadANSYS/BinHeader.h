/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ====================================================================
// Binaerer Header fuer alle ANSYS Ergebnisfiles
// Quelle: Guide to interacting with ANSYS Ver. 5.6
// alle integer haben die Laenge 32 bit.
// alle chars haben die Laenge 8 bit.
// Laenge ueber alles: 100 x integer = 400 Bytes
// 17.10.2001 Bjoern Sander
// ====================================================================
#ifndef __BIN_HEADER_HPP__
#define __BIN_HEADER_HPP__

#include <util/coviseCompat.h>

// 27.04.2011 Eduardo Aguilar
// The Standard Header for ANSYS Binary Files
// Programmer's Manual for Mechanical APDL Release 12.1 (November 2009)
// Each of the ANSYS program's binary files contains a standard, 100-integer file header that
// describes the file contents.  The header contains the items listed below, always in the following order:

struct BinHeader
{
    int filenum_; // Item 1 - File number: 12=standard header
    // int format_;                               // File Format: 0=internal, 1=external (old)
    int format_; // Item 2 - File format: 1=small, -1=large
    int time_; // Item 3 - Time in compact form
    int date_; // Item 4 - Date in compact form
    int unit_; // Item 5 - Units of measurement:  0=user defined, 1=SI, 2=CSG, 3=feet, 4=inches, 5=MKS, 6=MPA, 7=microMKS
    int version_; // Item 10 - ANSYS release level in integer form (version of ANSYS calculating this result)
    int ansysdate_; // Item 11 - Release date of ANSYS calculating this result
    char machine_[13]; // Items 12-14 - Machine identifier in integer form calculating this result
    char jobname_[9]; // Items 15-16 - Jobname in integer form
    char product_[9]; // Items 17-18 - Product name of ANSYS
    char label_[5]; // Item 19 - ANSYS special version label string
    char user_[13]; // Items 20-22 - User name in integer form
    char machine2_[13]; // Items 23-25 - Machine identifier in integer form
    int recordsize_; // Item 26 - System record size
    int maxfilelen_; // Item 27 - Maximum file length
    int maxrecnum_; // Item 28 - Maximun record number
    int cpus_; // Item 29 - Number of processors used for this task
    // Items 31-38 - Jobname
    char title_[81]; // Items 41-60 - Main title of analysis in integer form
    char subtitle_[81]; // Items 61-80 - First subtitle in integer form
    // Item 95 - Split point of the file
    // Items 97-98 - LONGINT of the maximum file length

    // Initialisierung:
    BinHeader(void)
    {
        memset(this, 0, sizeof(BinHeader));
    }
};
#endif
