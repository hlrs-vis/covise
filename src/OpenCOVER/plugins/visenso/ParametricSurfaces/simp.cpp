/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <math.h>
#include <stdlib.h>
#include "numeric.h"
#include "ELIST.H"

static int changed = false;
static int eval_functions = false;

HlExprList *HlExprList::simplify()
{
    HlExprList *q = this;
    int n = 0;

    eval_functions = false;

    do
    {
        changed = false;
        q = q->Flat()->ExpandTimes()->Order()->simp();

    } while (changed && ++n < 10);

    return q;
}

HlExprList *HlExprList::simp()
{
    for (int i = 0; i < Length(); i++)
        setarg(i, arg(i)->simp());

    switch (typeOfHead())
    {

    case PLUS:
        return simp_plus();
    case TIMES:
        return simp_times();
    case POWER:
        return simp_power();
    case SQRT:
        return simp_sqrt();
    case SIN:
        return simp_sin();
    case COS:
        return simp_cos();
    case TAN:
        return simp_tan();
    case ABS:
        return simp_betrag();
    case SIGN:
        return simp_signum();
    default:
        return this;
    }
}

HlExprList *HlExprList::simp_plus()
{
    HlExprList *a, *b;

    if (First()->NumberQ())
    {

        // 2+3+u -> 5+u
        while (Length() > 1 && Second()->NumberQ())
        {
            changed = true;
            First()->setDoubleVal(First()->getDoubleVal() + Second()->getDoubleVal());
            delarg(Second());
        }

        // 0+u -> u
        if (Length() > 1 && First()->Is(0.0))
        {
            changed = true;
            delarg(First());
        }
    }

    int found = false;

    do
    {
        found = false;
        for (int i = 0; i < Length(); i++)
        {

            for (int k = 0; k < Length(); k++)
            {

                if (i != k)
                {

                    switch (arg(i)->is_times_of(arg(k)))
                    {

                    case 0: // Nichts dergleichen
                        break;

                    case 1: // a + n*a -> (n+1)*a
                        arg(k)->First()->setDoubleVal(arg(k)->First()->getDoubleVal() + 1);
                        delarg(arg(i));
                        found = true;
                        changed = true;
                        break;

                    case 2: // m*a + n*a -> (m+n)*a
                        arg(k)->First()->setDoubleVal(
                            arg(k)->First()->getDoubleVal() + arg(i)->First()->getDoubleVal());
                        delarg(arg(i));
                        found = true;
                        changed = true;
                        break;
                    }

                    if (!found)
                    {
                        if (arg(i)->SameQ(arg(k))) // a + a -> 2*a
                        {
                            a = arg(i);
                            b = arg(k);
                            apparg(C(N(TIMES), N(2.0), N(a)));
                            delarg(a);
                            delarg(b);
                            found = true;
                            changed = true;
                        }
                    }
                }
                if (found)
                    break;
            }
            if (found)
                break;
        }
    } while (found);

    // Plus(u) -> u
    if (Length() == 1)
    {
        changed = true;
        return (HlExprList *)onlyFirstArg();
    }

    return this;
}

HlExprList *HlExprList::simp_times()
{
    HlExprList *a, *b;

    if (First()->NumberQ())
    {

        //  2*3*u -> 6*u
        while (Length() > 1 && Second()->NumberQ())
        {
            changed = true;
            First()->setDoubleVal(First()->getDoubleVal() * Second()->getDoubleVal());
            delarg(Second());
        }

        // 0*u -> 0
        if (Length() > 0 && First()->Is(0.0))
        {
            changed = true;
            delete this;
            return N(0.0);
        }

        // 1*u -> u
        if (Length() > 1 && First()->Is(1.0))
        {
            changed = true;
            delarg(First());
        }
    }

    int found = false;

    do
    {
        found = false;
        for (int i = 0; i < Length(); i++)
        {
            for (int k = 0; k < Length(); k++)
            {
                if (i != k)
                {
                    switch (arg(i)->is_power_of(arg(k)))
                    {

                    case 0: // Nichts dergleichen
                        break;

                    case 1: // a * a^n -> a^(n+1)
                        arg(k)->Second()->setDoubleVal(
                            arg(k)->Second()->getDoubleVal() + 1);
                        delarg(arg(i));
                        found = true;
                        changed = true;
                        break;

                    case 2: // a^m * a^n -> a^(m+n)
                        arg(k)->Second()->setDoubleVal(
                            arg(k)->Second()->getDoubleVal() + arg(i)->Second()->getDoubleVal());
                        delarg(arg(i));
                        found = true;
                        changed = true;
                        break;
                    }

                    if (!found)
                    {
                        if (arg(i)->SameQ(arg(k))) // a * a -> a^2
                        {
                            a = arg(i);
                            b = arg(k);
                            apparg(C(N(POWER), N(a), N(2.0)));
                            delarg(a);
                            delarg(b);
                            found = true;
                            changed = true;
                        }
                    }
                }
                if (found)
                    break;
            }
            if (found)
                break;
        }
    } while (found);

    // Times(u) -> u
    if (Length() == 1)
    {
        changed = true;
        return (HlExprList *)onlyFirstArg();
    }

    return this;
}

HlExprList *HlExprList::simp_power()
{
    HlExprList *q;

    switch (First()->typeOfHead())
    {

    case NUMBER:

        // 2^3 -> 8
        if (Second()->NumberQ() && !Second()->Is(0.0))
        {
            changed = true;
            double h = pow(First()->getDoubleVal(), Second()->getDoubleVal());
            delete this;
            return N(h);
        }

        // 0^u -> 0 (fuer u<>0)
        if (First()->Is(0.0))
        {
            if (!(Second()->Is(0.0)))
            {
                changed = true;
                delete this;
                return N(0.0);
            }
        }

        // 1^u -> 1
        if (First()->Is(1.0))
        {
            changed = true;
            delete this;
            return N(1.0);
        }
        break;

    case TIMES:

        if (Second()->IntegerQ())
        {
            if (First()->Is(TIMES))
            {

                // (a*b)^n -> a^n * b^n
                changed = true;
                q = N(TIMES);
                for (int i = 0; i < First()->Length(); i++)
                {
                    q->apparg(C(N(POWER), N(First()->arg(i)), N(Second())));
                }
                delete this;
                return q;
            }
        }
        break;

    // (a^b)^c -> a^(b*c)
    case POWER:
    {
        changed = true;
        HlExprList *q = C(N(POWER), N(First()->First()),
                          C(N(TIMES), N(First()->Second()), N(Second())));
        delete this;
        return q;
    }

    default:
        break;
    }

    // a^0 -> 1, fuer a<>0
    if (Second()->Is(0.0) && !(First()->Is(0.0)))
    {
        changed = true;
        delete this;
        return N(1.0);
    }

    // a^1 -> a
    if (Second()->Is(1.0))
    {
        changed = true;
        return (HlExprList *)onlyFirstArg();
    }

    return this;
}

HlExprList *HlExprList::simp_sqrt()
{
    HlExprList *q;

    switch (First()->typeOfHead())
    {
    case NUMBER:
        if (eval_functions)
        {
            changed = true;
            double v = evalF();
            delete this;
            return N(v);
        }
    default:
        changed = true;
        q = C(N(POWER), N(First()), N(0.5));
        delete this;
        return q;
        break;
    }
    return this;
}

HlExprList *HlExprList::simp_betrag()
{
    HlExprList *q;
    int n = int(Second()->evalF());
    if (n > 1)
    {
        delete this;
        return N(0.0);
    }
    if (n == 1)
    {
        q = C(N(SIGN), N(First()), N(0.0));
        delete this;
        return q;
    }
    else
    {
        if (First()->typeOfHead() == NUMBER)
        {
            q = N(fabs(First()->evalF()));
            delete this;
            return q;
        }
    }
    return this;
}

HlExprList *HlExprList::simp_signum()
{
    HlExprList *q;
    int n = int(Second()->evalF());
    if (n > 0)
    {
        delete this;
        return N(0.0);
    }
    else
    {
        if (First()->typeOfHead() == NUMBER)
        {
            q = N(HlSignum(First()->evalF(), 0));
            delete this;
            return q;
        }
    }

    return this;
}

HlExprList *HlExprList::simp_sin()
{

    switch (First()->typeOfHead())
    {
    case NUMBER:
        if (eval_functions)
        {
            changed = true;
            double v = evalF();
            delete this;
            return N(v);
        }
        break;

    default:
        break;
    }

    return this;
}

HlExprList *HlExprList::simp_cos()
{
    switch (First()->typeOfHead())
    {
    case NUMBER:
        if (eval_functions)
        {
            changed = true;
            double v = evalF();
            delete this;
            return N(v);
        }
        break;

    default:
        break;
    }

    return this;
}

HlExprList *HlExprList::simp_tan()
{
    switch (First()->typeOfHead())
    {
    case NUMBER:
        if (eval_functions)
        {
            changed = true;
            double v = evalF();
            delete this;
            return N(v);
        }
        break;

    default:
        break;
    }

    return this;
}
