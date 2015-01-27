/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SAMM_CONV_H_
#define __SAMM_CONV_H_

#include <util/coTypes.h>

// 25.10.00
#include "StarFile.h"
/**
 * Class
 *
 */

namespace covise
{

class SammConv
{
public:
    // Struct: conversion SAMM -> parts + Proc to create conversions
    enum
    {
        MAXPARTS = 8
    };
    struct ConvertSAMM
    {
        signed char numParts;
        char conv[MAXPARTS][8];
        char type;
    };

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    SammConv(const SammConv &);

    /// Assignment operator: NOT  IMPLEMENTED
    SammConv &operator=(const SammConv &);

    // Table convSAMM[case] :  2^8 = 256 cases -> each Corner cut
    ConvertSAMM *d_convSAMM;

    // 'Master' Cells
    static const ConvertSAMM s_samm0, s_samm1, s_samm2, s_samm2a,
        s_samm3, s_samm8dummy,
        s_samm4, s_samm5, s_samm8[4096];

    void createSammConv(const ConvertSAMM &base, const char *code,
                        int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7);

    // create this ratio of 'spare' entries for degenerated cells
    static const float DEGEN_RATIO;

    enum
    {
        UNUSED = 5
    };

public:
    /// Destructor : virtual in case we derive objects
    virtual ~SammConv();

    /// Default constructor: create conversion tables
    SammConv();

    // convert cell-table + samm-table -> new cell-table
    // return newCell->oldCell table

    int *convertSamm(StarModelFile::CellTabEntry *&cellTab,
                     StarModelFile::SammTabEntry *sammTab,
                     StarModelFile::CellTypeEntry *cellType,
                     int &numElem, int mxtb,
                     void (*dumpFunct)(const char *));
};
}
#endif
