/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __PARSER_H__
#define __PARSER_H__

#include "SCANNER.H"
#include "ELIST.H"
#include "hlerror.h"

/*------------------------------------------------------------*/
/*  Definition der Klasse parser                              */
/*------------------------------------------------------------*/

class HlParser
{

private:
    HlSymbolTable *mSymTab;
    HlScanner mLexer;
    HlSymEntry *mLookahead;
    HlError *mError;

    HlBaseList *assignment();
    HlBaseList *comparisation();
    HlBaseList *expression();
    HlBaseList *term();
    HlBaseList *power();
    HlBaseList *factor();
    void match(types t);
    void match(HlSymEntry *t);

    HlBaseList *registerNew(HlBaseList *b);
    void registerDelete();

public:
    HlParser();
    ~HlParser();
    void setSymTab(HlSymbolTable *symtab);
    void setError(HlError *error);
    void fillSymTab();
    HlBaseList *parseString(const string &s);
};

inline void HlParser::setSymTab(HlSymbolTable *symtab)
{
    mSymTab = symtab;
    mLexer.setSymTab(mSymTab);
}

inline void HlParser::setError(HlError *error)
{
    mError = error;
}

#endif // __PARSER_H__
