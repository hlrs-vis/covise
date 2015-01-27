/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ELIST.H"

static const double PI = 3.14159265358979323846;

HlExprList *HlExprList::diff()
{
    return First()->simplify()->diff_all(Second())->simplify();
}

HlExprList *HlExprList::diff_all(HlExprList *dv)
{
    if (Depend(dv))
    {

        HlExprList *a;

        switch (typeOfHead())
        {

        case IDENT:
            return N(1.0);
        case PLUS:
            return diff_plus(dv);
        case TIMES:
            return diff_times(dv);
        case POWER:
            return diff_power(dv);

        case LESS:
            return N(this);
        case GREATER:
            return N(this);
        case LESSEQUAL:
            return N(this);
        case GREATEREQUAL:
            return N(this);
        case EQUAL:
            return N(this);

        case IFF:
            return diff_iff(dv);

        case SQRT:
            a = diff_sqrt(First());
            break;
        case EXP:
            a = diff_exp(First());
            break;
        case LOG:
            a = diff_log(First());
            break;
        case SIN:
            a = diff_sin(First());
            break;
        case COS:
            a = diff_cos(First());
            break;
        case TAN:
            a = diff_tan(First());
            break;
        case ASIN:
            a = diff_asin(First());
            break;
        case ACOS:
            a = diff_acos(First());
            break;
        case ATAN:
            a = diff_atan(First());
            break;
        case SINH:
            a = diff_sinh(First());
            break;
        case COSH:
            a = diff_cosh(First());
            break;
        case TANH:
            a = diff_tanh(First());
            break;
        case ASINH:
            a = diff_asinh(First());
            break;
        case ACOSH:
            a = diff_acosh(First());
            break;
        case ATANH:
            a = diff_atanh(First());
            break;
        case FRESNELSIN:
            a = diff_fresnelsin(First());
            break;
        case FRESNELCOS:
            a = diff_fresnelcos(First());
            break;
        case ERF:
            a = diff_erf(First());
            break;
        case SINC:
            a = diff_sinc(First());
            break;
        case SI:
            a = diff_si(First());
            break;
        case ABS:
            a = diff_betrag();
            break;
        case SIGN:
            a = diff_signum();
            break;

        default:
            return diff_unknown(dv);
        }

        // u'(x) -> u'(x)
        if (First()->Is(IDENT))
        {
            return a;
        }

        // u(v(x))' -> v'(x) * u'(v(x))
        else
        {
            return C(N(TIMES), First()->diff_all(dv), a);
        }
    }

    // Nicht abhaengig von der Variablen nach der differnziert wird -> 0
    return N(0.0);
}

HlExprList *HlExprList::diff_unknown(HlExprList *dv)
{
    switch (First()->typeOfHead())
    {

    case IDENT:
        return C(N(DIFF),
                 N(this),
                 N(dv));

    default:
        return C(N(TIMES),
                 First()->diff_all(dv),
                 C(N(OP),
                   C(N(DIFFOP),
                     N(getsym())),
                   N(First())));
    }
}

// (a(x)+b(x)+c(x))' -> a'(x) + b'(x) + c'(x)
HlExprList *HlExprList::diff_plus(HlExprList *dv)
{
    HlExprList *q = N(PLUS);

    for (int i = 0, n = Length(); i < n; i++)
    {
        if (arg(i)->Depend(dv))
            q->apparg(arg(i)->diff_all(dv));
    }

    return q;
}

// (a(x)*b(x)*c(x)) -> a'(x)*b(x)*c(x) + a(x)*b'(x)*c(x) + a(x)*b(x)*c'(x)
HlExprList *HlExprList::diff_times(HlExprList *dv)
{
    HlExprList *q = N(PLUS);
    HlExprList *a;

    for (int i = 0, n = Length(); i < n; i++)
    {
        if (arg(i)->Depend(dv))
        {
            a = (HlExprList *)q->apparg(TIMES);
            for (int k = 0; k < n; k++)
            {
                if (k == i)
                    a->apparg(arg(k)->diff_all(dv));
                else
                    a->apparg(N(arg(k)));
            }
        }
    }
    return q;
}

HlExprList *HlExprList::diff_power(HlExprList *dv)
{

    // (u^r)'
    if (Second()->NumberQ())
    {

        // (u^0)' -> 0
        if (Second()->Is(0.0))
        {
            return N(0.0);
        }

        // (u^1)' -> u'
        if (Second()->Is(1.0))
        {
            return First()->diff_all(dv);
        }

        // (x^r)' = r * x^(r-1)
        if (First()->Is(IDENT))
        {
            if (Second()->Is(2.0))
            {
                return C(N(TIMES),
                         N(Second()->getDoubleVal()),
                         N(First()));
            }
            else
            {
                return C(N(TIMES),
                         N(Second()->getDoubleVal()),
                         C(N(POWER),
                           N(First()),
                           N(Second()->getDoubleVal() - 1)));
            }
        }

        // (u^r)' = r * u' * u^(r-1)
        else
        {
            return C(N(TIMES),
                     N(Second()->getDoubleVal()),
                     First()->diff_all(dv),
                     C(N(POWER),
                       N(First()),
                       N(Second()->getDoubleVal() - 1)));
        }
    }

    // Wenn nichts mehr hilft (u(x)^v(x))' -> (exp(v(x)*log(u(x))))'
    return C(N(EXP),
             C(N(TIMES),
               N(Second()),
               C(N(LOG),
                 N(First()))))->diff_all(dv);
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_sqrt(HlExprList *dv)
{
    return C(N(TIMES),
             N(0.5),
             C(N(POWER),
               N(dv),
               N(-0.5)));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_exp(HlExprList *dv)
{
    return C(N(EXP),
             N(dv));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_log(HlExprList *dv)
{
    return C(N(POWER),
             N(dv),
             N(-1.0));
}

/*-----------------------------------------------------------------*/

// sin(x)' -> cos(x)
HlExprList *HlExprList::diff_sin(HlExprList *dv)
{
    return C(N(COS), N(dv));
}

/*-----------------------------------------------------------------*/

// cos(x)' -> -sin(x)
HlExprList *HlExprList::diff_cos(HlExprList *dv)
{
    return C(N(TIMES), N(-1.0), C(N(SIN), N(dv)));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_tan(HlExprList *dv)
{
    return C(N(PLUS),
             N(1.0),
             C(N(POWER),
               C(N(TAN),
                 N(dv)),
               N(2.0)));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_asin(HlExprList *dv)
{
    return C(N(POWER),
             C(N(PLUS),
               N(1.0),
               C(N(TIMES),
                 N(-1.0),
                 C(N(POWER),
                   N(dv),
                   N(2.0)))),
             N(-0.5));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_acos(HlExprList *dv)
{
    return C(N(TIMES),
             N(-1.0),
             diff_asin(dv));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_atan(HlExprList *dv)
{
    return C(N(POWER),
             C(N(PLUS),
               N(1.0),
               C(N(POWER),
                 N(dv),
                 N(2.0))),
             N(-1.0));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_sinh(HlExprList *dv)
{
    return C(N(COSH),
             N(dv));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_cosh(HlExprList *dv)
{
    return C(N(SINH),
             N(dv));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_tanh(HlExprList *dv)
{
    return C(N(PLUS),
             N(1.0),
             C(N(TIMES),
               N(-1.0),
               C(N(POWER),
                 C(N(TANH),
                   N(dv)),
                 N(2.0))));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_asinh(HlExprList *dv)
{
    return C(N(POWER),
             C(N(PLUS),
               N(1.0),
               C(N(POWER),
                 N(dv),
                 N(2.0))),
             N(-0.5));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_acosh(HlExprList *dv)
{
    return C(N(POWER),
             C(N(PLUS),
               N(-1.0),
               C(N(POWER),
                 N(dv),
                 N(2.0))),
             N(-0.5));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_atanh(HlExprList *dv)
{
    return C(N(POWER),
             C(N(PLUS),
               N(1.0),
               C(N(TIMES),
                 N(-1.0),
                 C(N(POWER),
                   N(dv),
                   N(2.0)))),
             N(-1.0));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_erf(HlExprList *dv)
{
    return C(N(TIMES),
             C(N(POWER),
               C(N(SQRT),
                 C(N(TIMES),
                   N(2.0),
                   N(PI))),
               N(-1.0)),
             C(N(EXP),
               C(N(TIMES),
                 N(-1.0),
                 C(N(POWER),
                   N(dv),
                   N(2)))));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_fresnelsin(HlExprList *dv)
{
    return C(N(SIN),
             C(N(TIMES),
               N(0.5),
               N(PI),
               C(N(POWER),
                 N(dv),
                 N(2.0))));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_fresnelcos(HlExprList *dv)
{
    return C(N(COS),
             C(N(TIMES),
               N(0.5),
               N(PI),
               C(N(POWER),
                 N(dv),
                 N(2.0))));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_sinc(HlExprList *dv)
{
    return C(N(PLUS),
             C(N(TIMES),
               C(N(COS),
                 N(dv)),
               C(N(POWER),
                 N(dv),
                 N(-1))),
             C(N(TIMES),
               C(N(SIN),
                 N(dv)),
               C(N(POWER),
                 N(dv),
                 N(-2)),
               N(-1))

                 );
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_si(HlExprList *dv)
{
    return C(N(SINC),
             N(dv));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_betrag()
{
    return C(N(ABS),
             N(First()),
             N(Second()->evalF() + 1));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_signum()
{
    return C(N(SIGN),
             N(First()),
             N(Second()->evalF() + 1));
}

/*-----------------------------------------------------------------*/

HlExprList *HlExprList::diff_iff(HlExprList *dv)
{
    return C(N(IFF),
             N(First()),
             N(Second()->diff_all(dv)),
             N(Third()->diff_all(dv)));
}
