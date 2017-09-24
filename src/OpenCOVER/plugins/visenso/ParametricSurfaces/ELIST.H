/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __EXPRLIST_H__
#define __EXPRLIST_H__

#include "BASELIST.H"

class HlExprList : private HlBaseList
{

private:
    HlExprList *arg(size_t i);
    void setarg(size_t i, HlExprList *e);
    HlExprList *First();
    HlExprList *Second();
    HlExprList *Third();

    // C steht fuer Create
    HlExprList *C(HlExprList *op, HlExprList *a1);
    HlExprList *C(HlExprList *op, HlExprList *a1, HlExprList *a2);
    HlExprList *C(HlExprList *op, HlExprList *a1, HlExprList *a2, HlExprList *a3);
    HlExprList *C(HlExprList *op, HlExprList *a1, HlExprList *a2, HlExprList *a3, HlExprList *a4);

    // Tester
    bool MathSymbolQ();
    bool NumberQ();
    bool IntegerQ();
    bool EvenQ();
    bool OddQ();
    bool SameQ(HlExprList *);
    bool UnsameQ(HlExprList *);
    bool LessThen(HlExprList *);

    // Algebra
    HlExprList *Flat();
    HlExprList *Order();
    HlExprList *ExpandTimes();
    HlExprList *Distribute(types, HlExprList *);
    HlExprList *Distribute(types, HlExprList *, HlExprList *);

    size_t compare(HlExprList *);

    int is_times_of(HlExprList *);
    int is_power_of(HlExprList *);

    // Differenzieren
    HlExprList *diff();
    HlExprList *diff_plus(HlExprList *);
    HlExprList *diff_times(HlExprList *);
    HlExprList *diff_power(HlExprList *);
    HlExprList *diff_sqrt(HlExprList *dv);
    HlExprList *diff_sin(HlExprList *);
    HlExprList *diff_cos(HlExprList *);
    HlExprList *diff_tan(HlExprList *);
    HlExprList *diff_asin(HlExprList *);
    HlExprList *diff_acos(HlExprList *);
    HlExprList *diff_atan(HlExprList *);
    HlExprList *diff_sinh(HlExprList *);
    HlExprList *diff_cosh(HlExprList *);
    HlExprList *diff_tanh(HlExprList *);
    HlExprList *diff_asinh(HlExprList *);
    HlExprList *diff_acosh(HlExprList *);
    HlExprList *diff_atanh(HlExprList *);
    HlExprList *diff_log(HlExprList *);
    HlExprList *diff_exp(HlExprList *);
    HlExprList *diff_fresnelsin(HlExprList *);
    HlExprList *diff_fresnelcos(HlExprList *);
    HlExprList *diff_erf(HlExprList *);
    HlExprList *diff_sinc(HlExprList *);
    HlExprList *diff_si(HlExprList *);
    HlExprList *diff_betrag();
    HlExprList *diff_signum();
    HlExprList *diff_unknown(HlExprList *);
    HlExprList *diff_iff(HlExprList *);

    HlExprList *simp();
    HlExprList *simp_plus();
    HlExprList *simp_times();
    HlExprList *simp_power();
    HlExprList *simp_sqrt();
    HlExprList *simp_sin();
    HlExprList *simp_cos();
    HlExprList *simp_tan();
    HlExprList *simp_betrag();
    HlExprList *simp_signum();

    HlExprList *assign(HlExprList *e);

    // Methoden um Textausgaben zu erzeugen
    void infix(); // Infixnotation der internen Darstellung
    void postfix(); // Postfixnotation der internen Darstellung
    int need_bracket(HlExprList *e);

public:
    HlExprList();
    ~HlExprList();

    // N steht fuer Neu
    static HlExprList *N(HlExprList *);
    static HlExprList *N(types);
    static HlExprList *N(double);
    static HlExprList *N(HlSymEntry *);

    HlExprList *eval();
    HlExprList *simplify();

    static HlExprList *toElist(HlBaseList *b);

    bool Depend(HlExprList *);
    bool ok();

    HlExprList *diff_all(HlExprList *);

    const string &fktstr();

    double evalF();
    int getEvalError();

    double Romberg(double *var, double a, double b, int N);
    double RiemannSumme(double *var, double a, double b, int N);
};

/*-----------------------------------------------------------------*/
/*        Inlinefunktionen fuer HlExprList                              */
/*-----------------------------------------------------------------*/

inline HlExprList::HlExprList() // Default Konstruktor
{
}

inline HlExprList::~HlExprList()
{
}

inline HlExprList *HlExprList::toElist(HlBaseList *b)
{
    return (HlExprList *)b;
}

inline bool HlExprList::ok()
{
    //	if( ELISTERROR.get_error() ) return false;

    return getsym() ? true : false;
}

inline int HlExprList::getEvalError()
{
    //	return errno;
    return 1;
}

inline HlExprList *HlExprList::arg(size_t i)
{
    return (HlExprList *)HlBaseList::arg(i);
}

inline void HlExprList::setarg(size_t i, HlExprList *e)
{
    HlBaseList::setarg(i, e);
}

inline HlExprList *HlExprList::First()
{
    return arg(0);
}

inline HlExprList *HlExprList::Second()
{
    return arg(1);
}

inline HlExprList *HlExprList::Third()
{
    return arg(2);
}

/*--------------------------------------------------------------------*/

inline HlExprList *HlExprList::N(HlExprList *e)
{
    return (HlExprList *)e->copy();
}

inline HlExprList *HlExprList::N(types t)
{
    return (HlExprList *)new HlBaseList(t);
}

inline HlExprList *HlExprList::N(double v)
{
    return (HlExprList *)new HlBaseList(v);
}

inline HlExprList *HlExprList::N(HlSymEntry *sym)
{
    return (HlExprList *)new HlBaseList(sym);
}

/*--------------------------------------------------------------------*/

inline HlExprList *HlExprList::C(HlExprList *op, HlExprList *a1)
{
    op->HlBaseList::apparg(a1);
    return op;
}

inline HlExprList *HlExprList::C(HlExprList *op, HlExprList *a1, HlExprList *a2)
{
    op->apparg(a1);
    op->apparg(a2);
    return op;
}

inline HlExprList *HlExprList::C(HlExprList *op, HlExprList *a1, HlExprList *a2, HlExprList *a3)
{
    op->apparg(a1);
    op->apparg(a2);
    op->apparg(a3);
    return op;
}

inline HlExprList *HlExprList::C(HlExprList *op, HlExprList *a1, HlExprList *a2, HlExprList *a3, HlExprList *a4)
{
    op->apparg(a1);
    op->apparg(a2);
    op->apparg(a3);
    op->apparg(a4);
    return op;
}

/*--------------------------------------------------------------------*/
/*             Tester                                                 */
/*--------------------------------------------------------------------*/

inline bool HlExprList::MathSymbolQ()
{
    return (typeOfHead() > _BEGIN_MATHSYMBOL_) && (typeOfHead() < _END_MATHSYMBOL_);
}

inline bool HlExprList::NumberQ()
{
    return Is(NUMBER);
}

inline bool HlExprList::IntegerQ()
{
    return NumberQ() ? (double)((int)getDoubleVal()) == getDoubleVal() ? true : false : false;
}

inline bool HlExprList::EvenQ()
{
    return IntegerQ() ? (int)getDoubleVal() % 2 == 0 ? true : false : false;
}

inline bool HlExprList::OddQ()
{
    return IntegerQ() ? (int)getDoubleVal() % 2 == 1 ? true : false : false;
}

inline bool HlExprList::SameQ(HlExprList *e)
{
    return (compare(e) == 0);
}

inline bool HlExprList::UnsameQ(HlExprList *e)
{
    return (compare(e) != 0);
}

inline bool HlExprList::LessThen(HlExprList *e)
{
    return (compare(e) < 0);
}

/*--------------------------------------------------------------------*/

#endif // { __EXPRLIST_H__ }
