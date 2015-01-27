/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ILL_CONV_H_
#define __ILL_CONV_H_

#include <util/coTypes.h>

// 25.10.00
#include "StarFile.h"
/**
 * Class
 *
 */
namespace covise
{

class STAREXPORT IllConv
{
public:
    // Struct: conversion ILL -> parts + Proc to create conversions
    enum
    {
        MAXPARTS = 2
    };
    struct ConvertILL
    {
        signed short numParts;
        char conv[MAXPARTS][8];
    };

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    IllConv(const IllConv &);

    /// Assignment operator: NOT  IMPLEMENTED
    IllConv &operator=(const IllConv &);

    // Table convILL[case] :  2^12 = 4096 cases -> each Edge cut
    static const ConvertILL s_convILL[4096];

    void error();

    enum // never 7 adjacent edges away witfout fourth
    {
        UNUSED = 0x577
    };

public:
    /// Destructor : virtual in case we derive objects
    virtual ~IllConv();

    /// Default constructor: create conversion tables
    IllConv();

    // convert cell-table + ill-table -> new cell-table
    // transform convertMap if given, create new otherwise

    void convertIll(StarModelFile::CellTabEntry *&cellTab,
                    StarModelFile::CellTypeEntry *cellType,
                    int *&convertMap,
                    int &numElem, int mxtb,
                    void (*dumpFunct)(const char *));
};
}
#endif
