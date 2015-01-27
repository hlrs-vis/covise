/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
/*
//============================================================
// VirCinity IT Consulting (C)
// CaseLexer.l
//
// Header for lexographic analysis for Ensight geometry & data files
//
//
// Creation date: 20.05.02
//============================================================
*/

#ifndef flexer_H
#define flexer_H
using namespace std;
// include standard Flex header file, but use First as prefix, not yy.
#ifndef yyFlexLexer
#define yyFlexLexer CaseFlexLexer
#include <FlexLexer.h>
#undef yyFlexLexer
#endif
//
// switch off register
//
#define register

// include the parser for types and identifiers
#include "CaseParser.hpp"

// define new class
class CaseLexer : public CaseFlexLexer
{
public:
    // new constructor
    CaseLexer(istream *pDesc)
        : CaseFlexLexer(pDesc)
    {
    }
    virtual void LexerError(const char msg[]);
    // new scanner method
    int scan(MyTokenType *token);
};
#endif
