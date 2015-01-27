/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
/*
//============================================================
// VirCinity IT Consulting (C)
// GeomLexer.l
//
// Header for lexograpic analysis for Ensight geometry & data files
//
//
// Creation date: 20.05.02
//============================================================
*/

#ifndef flexer_H
#define flexer_H

// include standard Flex header file, but use First as prefix, not yy.
#ifndef yyFlexLexer
#define yyFlexLexer GeomFlexLexer
#include <FlexLexer.h>
#undef yyFlexLexer
#endif
//
// switch off register
//
#define register

// include the parser for types and identifiers
#include "GeomParser.h"

// define new class
class GeomLexer : public GeomFlexLexer
{
public:
    // new constructor
    GeomLexer(istream *pDesc)
        : GeomFlexLexer(pDesc)
    {
    }

    // new scanner method
    int scan(GeomTokenType *token);
};
#endif
