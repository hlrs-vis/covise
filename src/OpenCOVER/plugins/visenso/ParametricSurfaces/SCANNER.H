/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SCANNER_H__
#define __SCANNER_H__

#include "SYMBOL.H"

/*------------------------------------------------------------*/
/*  Definition der Klasse HlScanner                           */
/*------------------------------------------------------------*/

class HlScanner
{

public:
    HlScanner();
    ~HlScanner();
    HlSymEntry *nextToken();
    void setInput(const string &s);
    void setSymTab(HlSymbolTable *symtab);

private:
    int nextChar();
    void backChar(int c);
    bool isStartSymbol(HlSymEntry *s);
    HlSymEntry *catchNumber(unsigned int c);
    HlSymEntry *catchName(unsigned int c);
    HlSymEntry *catchOp(unsigned int c);

    HlSymbolTable *mSymTab;
    string mInputbuffer;
    unsigned mInputPtr;
    HlSymEntry *mLastToken;
};

/*------------------------------------------------------------*/
/*  Inlinefunktionen der Klasse HlScanner                     */
/*------------------------------------------------------------*/

inline void HlScanner::setSymTab(HlSymbolTable *symtab)
{
    mSymTab = symtab;
}

inline HlScanner::HlScanner()
    : mSymTab(NULL)
    , mInputbuffer()
    , mInputPtr(0)
    , mLastToken(NULL)
{
}

inline HlScanner::~HlScanner()
{
}

inline int HlScanner::nextChar()
{
    if (mInputPtr >= mInputbuffer.length())
        return EOS;
    return mInputbuffer[mInputPtr++];
}

inline void HlScanner::backChar(int c)
{
    mInputbuffer[--mInputPtr] = c;
}

inline void HlScanner::setInput(const string &s)
{
    mInputbuffer = s;
    mInputPtr = 0;
    mLastToken = NULL;
}

inline bool HlScanner::isStartSymbol(HlSymEntry *s)
{
    if (NULL == s)
        return true;
    if (s->tokenIn(RIGHTB, ASSIGN, KOMMA))
        return true;
    return false;
}

#endif // __SCANNER_H__
