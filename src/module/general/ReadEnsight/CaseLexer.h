#ifndef VISTLE_READENSIGHT_CASELEXER_H
#define VISTLE_READENSIGHT_CASELEXER_H

/*
//============================================================
// VirCinity IT Consulting (C)
// CaseLexer.l
//
// Header for lexographic analysis for EnSight geometry & data files
//
//
// Creation date: 20.05.02
//============================================================
*/

// include standard Flex header file, but use First as prefix, not yy.
#ifndef yyFlexLexer
#define yyFlexLexer CaseFlexLexer
#include <FlexLexer.h>
#undef yyFlexLexer
#endif

// include the parser for types and identifiers
#include "CaseParser.h"

// define new class
class CaseLexer: public CaseFlexLexer {
public:
    // new constructor
    CaseLexer(std::istream *pDesc);
    virtual ~CaseLexer();
    // new scanner method
    int scan(CaseTokenType *token, CaseParserDriver &driver);
};
#endif
