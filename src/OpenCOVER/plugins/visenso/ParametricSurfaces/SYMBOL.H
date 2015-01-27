/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SYMBOL_H__
#define __SYMBOL_H__

#include <string>
#include <vector>
using std::string;
using std::vector;

enum types
{
    EOS = 0,
    _BEGIN_TOKEN_,
    NOTOKEN,
    RIGHTB,
    LEFTB,
    KOMMA,
    _END_TOKEN_,
    _BEGIN_MATHSYMBOL_,
    _BEGIN_ATOM_,
    NUMBER,
    IDENT,
    _END_ATOM_,
    _BEGIN_BINOP_,
    LESS,
    GREATER,
    LESSEQUAL,
    GREATEREQUAL,
    EQUAL,
    ASSIGN,
    PLUS,
    MINUS,
    TIMES,
    DIV,
    POWER,
    OP,
    DIFF,
    _END_BINOP_,
    _BEGIN_UNOP_,
    UMIN,
    ERF,
    FRESNELSIN,
    FRESNELCOS,
    SQRT,
    EXP,
    LOG,
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    DIFFOP,
    ABS,
    SIGN,
    SINC,
    SI,
    ROMBERG,
    DEFINT,
    IFF,
    _END_UNOP_,
    _BEGIN_UNKOWN_,
    UNKNOWN1,
    UNKNOWN2,
    _END_UNKNOWN_,
    _END_MATHSYMBOL_,
    _BEGIN_TESTFUNCTIONS_,
    SELF,
    FREE,
    KOMM,
    ASSO,
    DIST,
    SIMPLIFY,
    VAL,
    POST,
    COMP,
    _END_TESTFUNCTIONS_
};

enum propertys
{
    NOPROP = 0x00,
    ASSOZIATIV = 0x01,
    KOMMUTATIV = 0x02,
    POSTFIX = 0x04,
    INFIX = 0x08,
    PREFIX = 0x10,
    BRACKET = 0x20,
    NARGS = 0x40
};

class HlExprList;
class HlSymEntry;
class symtable;

typedef vector<HlSymEntry *> symentryvector;

/*-------------------------------------------------------------*/
/* symentry: Struktur eines Symboltabelleneintrags        */
/*-------------------------------------------------------------*/

class HlSymEntry
{

public:
    HlSymEntry();
    HlSymEntry(types tok, double val, const string &lex);
    HlSymEntry(types tok, double val, const string &lex, const string &lname);

    double getVal();
    void setVal(double val);
    types getToken();
    void setToken(types tok);
    bool tokenIn(types);
    bool tokenIn(types, types);
    bool tokenIn(types, types, types);
    bool tokenIn(types, types, types, types);
    bool tokenIn(types, types, types, types, types);
    const string &getLexem();
    void setLexem(const string &s);
    const string &getLexemName();
    void setLexemName(const string &s);
    void setExprPtr(HlExprList *e);
    HlExprList *getExprPtr();
    double *getValuePtr();
    void setArgn(int argn);
    unsigned getArgn();
    void setProp(unsigned prop);
    bool getProp(unsigned prop);

private:
    string mLexemPtr; /* Lexem Pointer      */
    string mLexemName; /* Lexem Name         */
    types mToken; /* zugeordnetes Token */
    HlExprList *mExprPtr; /* Zeiger auf Elist   */
    double mValue; /* Wert               */
    unsigned mArgn; /* Anzahl Argumente   */
    unsigned mProp; /* Eigenschaften      */
};

/*------------------------------------------------------------------*/
/* class symboltable: Struktur der Symboltabelle                    */
/*------------------------------------------------------------------*/

class HlSymbolTable
{
public:
    HlSymbolTable();
    ~HlSymbolTable();
    HlSymEntry *lookup(const string &s);
    HlSymEntry *lookup(types t);
    HlSymEntry *insert(const string &lexem, types t);
    HlSymEntry *insert(const string &lexem, types t, const string &lexemname);
    HlSymEntry *insertFunction(const string &lexem, types t, int argn);
    HlSymEntry *insertInfix(const string &lexem, types t, const string &lexemname, int asso, int komm, int argn);

private:
    symentryvector mSymTable;
};

/*---------------------------------------------------------*/
/*   Inlinefunktionen fuer Symtableentry                   */
/*---------------------------------------------------------*/

inline HlSymEntry::HlSymEntry()
    : mLexemPtr()
    , mLexemName()
    , mToken(NUMBER)
    , mExprPtr(NULL)
    , mValue(0)
    , mArgn(0)
    , mProp(0)
{
}

inline HlSymEntry::HlSymEntry(types tok, double val, const string &lex)
    : mLexemPtr(lex)
    , mLexemName(lex)
    , mToken(tok)
    , mExprPtr(NULL)
    , mValue(val)
    , mArgn(0)
    , mProp(0)
{
}

inline HlSymEntry::HlSymEntry(types tok, double val, const string &lex, const string &lname)
    : mLexemPtr(lex)
    , mLexemName(lname)
    , mToken(tok)
    , mExprPtr(NULL)
    , mValue(val)
    , mArgn(0)
    , mProp(0)
{
}

inline double HlSymEntry::getVal()
{
    return mValue;
}

inline void HlSymEntry::setVal(double val)
{
    mValue = val;
}

inline void HlSymEntry::setToken(types t)
{
    mToken = t;
}

inline types HlSymEntry::getToken()
{
    return mToken;
}

inline bool HlSymEntry::tokenIn(types t1)
{
    return t1 == mToken;
}

inline bool HlSymEntry::tokenIn(types t1, types t2)
{
    return (t1 == mToken) || (t2 == mToken);
}

inline bool HlSymEntry::tokenIn(types t1, types t2, types t3)
{
    return (t1 == mToken) || (t2 == mToken) || (t3 == mToken);
}

inline bool HlSymEntry::tokenIn(types t1, types t2, types t3, types t4)
{
    return (t1 == mToken) || (t2 == mToken) || (t3 == mToken) || (t4 == mToken);
}

inline bool HlSymEntry::tokenIn(types t1, types t2, types t3, types t4, types t5)
{
    return (t1 == mToken) || (t2 == mToken) || (t3 == mToken) || (t4 == mToken) || (t5 == mToken);
}

inline const string &HlSymEntry::getLexem()
{
    return mLexemPtr;
}

inline void HlSymEntry::setLexem(const string &s)
{
    mLexemPtr = s;
}

inline const string &HlSymEntry::getLexemName()
{
    return mLexemName;
}

inline void HlSymEntry::setLexemName(const string &s)
{
    mLexemName = s;
}

inline void HlSymEntry::setExprPtr(HlExprList *e)
{
    mExprPtr = e;
}

inline HlExprList *HlSymEntry::getExprPtr()
{
    return mExprPtr;
}

inline double *HlSymEntry::getValuePtr()
{
    return &mValue;
}

inline void HlSymEntry::setArgn(int argn)
{
    mArgn = argn;
}

inline unsigned HlSymEntry::getArgn()
{
    return mArgn;
}

inline void HlSymEntry::setProp(unsigned prop)
{
    mProp |= prop;
}

inline bool HlSymEntry::getProp(unsigned prop)
{
    return (mProp & prop) ? true : false;
}

/*---------------------------------------------------------*/
/*   Inlinefunktionen fuer Symtable                        */
/*---------------------------------------------------------*/

inline HlSymbolTable::HlSymbolTable()
{
}

inline HlSymbolTable::~HlSymbolTable()
{
    for (unsigned i = 0; i < mSymTable.size(); i++)
        delete mSymTable[i];
}

inline HlSymEntry *HlSymbolTable::lookup(const string &s)
{
    for (unsigned i = 0; i < mSymTable.size(); i++)
    {
        if (mSymTable[i]->getLexem() == s)
            return mSymTable[i];
    }
    return NULL;
}

inline HlSymEntry *HlSymbolTable::lookup(types t)
{
    for (unsigned i = 0; i < mSymTable.size(); i++)
    {
        if (mSymTable[i]->getToken() == t)
            return mSymTable[i];
    }

    return NULL;
}

inline HlSymEntry *HlSymbolTable::insert(const string &lexem, types token)
{
    mSymTable.push_back(new HlSymEntry(token, 0., lexem, lexem));

    return mSymTable.back();
}

inline HlSymEntry *HlSymbolTable::insert(const string &lexem, types token, const string &lexemname)
{
    HlSymEntry *h = insert(lexem, token);

    h->setLexemName(lexemname);

    return h;
}

inline HlSymEntry *HlSymbolTable::insertFunction(const string &lexem, types t, int argn)
{
    HlSymEntry *h = insert(lexem, t);
    h->setProp(PREFIX);
    h->setProp(BRACKET);
    h->setArgn(argn);
    return h;
}

inline HlSymEntry *HlSymbolTable::insertInfix(const string &lexem, types t, const string &lexemname, int asso, int komm, int argn)
{
    HlSymEntry *h = insert(lexem, t, lexemname);
    h->setProp(INFIX);
    h->setArgn(argn);
    if (asso)
        h->setProp(ASSOZIATIV);
    if (komm)
        h->setProp(KOMMUTATIV);
    return h;
}

/*---------------------------------------------------------*/

#endif // __SYMBOL_H__
