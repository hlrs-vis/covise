/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __HlCAS_H__
#define __HlCAS_H__

#include <string>
using std::string;

#include "ELIST.H"
#include "SYMBOL.H"
#include "parser.h"
#include "hlerror.h"

class HlCAS
{

private:
    HlSymbolTable mSymbolTable;
    HlParser mParser;

public:
    HlError mError;

    HlCAS();
    void init();

    void setSymTab();
    HlExprList *parseString(const string &s);
    bool evalString(const string &s);
    HlExprList *diffTo(HlExprList *e, const string &dv);
    double evalf(HlExprList *e);
    double evalf(HlExprList *e, double *p, double v);
    double evalf(HlExprList *e, double *px, double vx, double *py, double vy);
    bool depend(HlExprList *e, const string &expr);

    HlSymEntry *lookup(const string &lexem);
    HlSymEntry *insert(const string &lexem);

    double *getValPtr(const string &varname);
    const string &getErrorMessage();
};

extern HlCAS HLCAS;

inline double HlCAS::evalf(HlExprList *e)
{
    setSymTab();
    return e->evalF();
}

inline double HlCAS::evalf(HlExprList *e, double *p, double d)
{
    if (e == NULL)
        return 0.0;
    if (!e->ok())
        return 0.0;
    *p = d;

    return evalf(e);
}

inline double HlCAS::evalf(HlExprList *e, double *px, double vx, double *py, double vy)
{
    if (e == NULL)
        return 0.0;
    if (!e->ok())
        return 0.0;
    *px = vx;
    *py = vy;

    return evalf(e);
}

inline double *HlCAS::getValPtr(const string &varname)
{
    return lookup(varname)->getValuePtr();
}

inline const string &HlCAS::getErrorMessage()
{
    return mError.getDescription();
}

inline HlCAS::HlCAS()
{
    init();
}

inline void HlCAS::init()
{
    setSymTab();
    mParser.setError(&mError);
    mParser.fillSymTab();
}

inline void HlCAS::setSymTab()
{
    HlBaseList::setSymTab(&mSymbolTable);
    mParser.setSymTab(&mSymbolTable);
}

inline HlExprList *HlCAS::parseString(const string &s)
{
    setSymTab();
    HlExprList *e = HlExprList::toElist(mParser.parseString(s));
    return e->eval()->simplify();
}

inline bool HlCAS::evalString(const string &s)
{
    HlExprList *h = parseString(s);
    bool OK = h->ok();
    delete h;
    return OK;
}

inline HlExprList *HlCAS::diffTo(HlExprList *e, const string &dv)
{

    if (NULL == e)
        return NULL;

    setSymTab();

    HlSymEntry *h = mSymbolTable.lookup(dv);

    if (h)
    {
        HlExprList *q = HlExprList::N(h);
        if (q)
        {
            return e->diff_all(q)->simplify();
            delete q;
        }
        else
            return NULL;
    }

    return HlExprList::N(0.0);
}

inline bool HlCAS::depend(HlExprList *e, const string &expr)
{
    if (NULL == e)
        return false;

    setSymTab();

    HlSymEntry *h = mSymbolTable.lookup(expr);

    if (h)
    {
        HlExprList *q = HlExprList::N(h);
        if (q)
        {
            bool erg = e->Depend(q);
            delete q;
            return erg;
        }
        else
            return false;
    }

    return false;
}

inline HlSymEntry *HlCAS::lookup(const string &lexem)
{
    setSymTab();
    return mSymbolTable.lookup(lexem);
}

inline HlSymEntry *HlCAS::insert(const string &lexem)
{
    setSymTab();
    return mSymbolTable.insert(lexem, IDENT);
}

#endif
