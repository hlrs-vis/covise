/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <string.h>
#include <stdio.h>

#include "ELIST.H"

/*--------------------------------------------------------------*/
/* depend: Stellt rekursiv fest, ob this oder seine Argumente   */
/*         von v abhaengig sind, wenn ja -> true sonst -> false */
/*--------------------------------------------------------------*/

bool HlExprList::Depend(HlExprList *v)
{
    if (getsym() == v->getsym())
        return true;

    if (Length() == 0)
        return false;

    for (int i = 0; i < Length(); i++)
    {
        if (arg(i)->getsym() == v->getsym())
            return true;

        if (arg(i)->Length())
        {
            if (arg(i)->Depend(v))
                return true;
        }
    }
    return false;
}

/*------------------------------------------------------------------------*/
/* HlExprList::compare( e ): Vergleicht rekursiv e mit this.                   */
/*                      Rueckgabewert: this = e ->  0                     */
/*                                     this < e -> -1                     */
/*                                     this > e ->  1                     */
/*------------------------------------------------------------------------*/

int HlExprList::compare(HlExprList *e)
{
    size_t i, n, h;

    if (Is(e->typeOfHead()))
    {

        switch (typeOfHead())
        {

        case IDENT:
            return strcmp(getsym()->getLexemName().c_str(), e->getsym()->getLexemName().c_str());

        case NUMBER:
            return (getDoubleVal() == e->getDoubleVal()) ? 0 : (getDoubleVal() < e->getDoubleVal()) ? -1 : 1;

        default:
            n = (Length() < e->Length()) ? Length() : e->Length();
            for (i = 0; i < n; i++)
            {
                if ((h = arg(i)->compare(e->arg(i))) != 0)
                {
                    return h;
                }
            }
            return (Length() == e->Length()) ? 0 : (Length() < e->Length()) ? -1 : 1;
        }
    }
    else
    {
        return (typeOfHead() < e->typeOfHead()) ? -1 : 1;
    }
}

/*---------------------------------------------------------------*/
/* is_times_of(e):  Stellt fest, ob e ein Vielfaches von this    */
/*                  ist. Dabei werden drei Faelle unterschieden: */
/*                  this + n*e    -> return 1                    */
/*                  m this + n*e  -> return 2                    */
/*                  sonst         -> return 0                    */
/*---------------------------------------------------------------*/

int HlExprList::is_times_of(HlExprList *e)
{
    int i;

    if (e->Is(TIMES))
    {
        if (e->First()->NumberQ())
        {

            if (Is(TIMES))
            {
                if (First()->NumberQ())
                {
                    if (Length() == e->Length())
                    {
                        for (i = 1; i < Length(); i++)
                        {
                            if (arg(i)->UnsameQ(e->arg(i)))
                            {
                                return 0;
                            }
                        }
                        return 2;
                    }
                }
            }

            if (e->Length() == 2)
            {
                return SameQ(e->Second());
            }

            else if (e->Length() > 2)
            {
                if (Is(TIMES))
                {
                    if (Length() == e->Length() - 1)
                    {
                        for (i = 0; i < Length(); i++)
                        {
                            if (arg(i)->UnsameQ(e->arg(i + 1)))
                            {
                                return 0;
                            }
                        }
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

/*---------------------------------------------------------------*/
/* is_power_of(e):  Stellt fest, ob e eine Potenz von this       */
/*                  ist. Dabei werden drei Faelle unterschieden: */
/*                  this * e^n    -> return 1                    */
/*                  this^n * e^m  -> return 2                    */
/*                  sonst         -> return 0                    */
/*---------------------------------------------------------------*/

int HlExprList::is_power_of(HlExprList *e)
{

    if (e->Is(POWER))
    {
        if (e->Second()->NumberQ())
        {

            if (Is(POWER))
            {
                if (Second()->NumberQ())
                {

                    if (First()->SameQ(e->First()))
                    {
                        return 2;
                    }
                }
                return 0;
            }

            if (SameQ(e->First()))
            {
                return 1;
            }
        }
    }

    return 0;
}

/*------------------------------------------------------------------*/
/* order: wendet rekursiv das Kommutativgesetz auf this und seine   */
/*       Argumente an, um bei kommutativen Vernuepfungen (Addition, */
/*       Multiplikation) die Reihenfolge der Operanden zu sortieren */
/*       c+a+b -> a+b+c, 1+x+5 -> 1+5+x                             */
/*------------------------------------------------------------------*/

HlExprList *HlExprList::Order()
{

    for (int i = 0; i < Length(); i++)
        setarg(i, arg(i)->Order());

    if (getsym()->getProp(KOMMUTATIV))
    {

        for (int i = 0; i < Length() - 1; i++)
        {
            for (int k = i + 1; k < Length(); k++)
            {
                if (arg(i)->compare(arg(k)) > 0)
                {
                    changearg(i, k);
                }
            }
        }
    }

    return this;
}

/*-----------------------------------------------------------------*/
/* flat: Wendet rekursiv das Assoziativgesetz auf this und seine   */
/*       Argumente an, um assoziative Verknuepfungen (Addition,    */
/*       Multiplikation) aufzuloesen.                              */
/*       a*(b*c) -> a*b*c                                          */
/*-----------------------------------------------------------------*/

HlExprList *HlExprList::Flat()
{
    for (int i = 0; i < Length(); i++)
        setarg(i, arg(i)->Flat());

    if (getsym()->getProp(ASSOZIATIV))
    {

        for (int i = 0; i < Length(); i++)
        {
            HlExprList *h = arg(i);
            if (Is(h->typeOfHead()))
            {
                for (int k = 0; k < h->Length(); k++)
                {
                    apparg(N(h->arg(k)));
                }
                delarg(h);
                i--;
            }
        }
    }

    return this;
}

/*-----------------------------------------------------------------------*/
/* ExpandTimes: Wendet rekursiv das Distributivgesetz auf this und seine */
/*          Argumente an.                                                */
/*          (a+b)(c+d) -> ac + ad + bc + bd                              */
/*-----------------------------------------------------------------------*/

HlExprList *HlExprList::ExpandTimes()
{
    int found = false;
    HlExprList *ak, *ai;

    for (int i = 0; i < Length(); i++)
        setarg(i, arg(i)->ExpandTimes());

    if (Is(TIMES))
    {
        do
        {
            found = false;
            for (int i = 0; i < Length(); i++)
            {
                for (int k = 0; k < Length(); k++)
                {
                    if (i != k)
                    {
                        if (arg(k)->Is(PLUS))
                        {
                            ak = arg(k);
                            ai = arg(i);
                            found = true;

                            // (a+b)(c+d) -> ac + ad + bc + bd
                            // a(b+c)     -> ab + ac
                            apparg(Distribute(TIMES, ai, ak));
                            delarg(ai);
                            delarg(ak);
                        }
                    }
                    if (found)
                        break;
                }
                if (found)
                    break;
            }
        } while (found);
    }

    return this;
}

/*-----------------------------------------------------------------------*/
/* Distribute: Wendet t bezueglich der Summen distributiv an             */
/*             (t,a)   -> t(a)                                           */
/*             (t,a+b) -> t(a) + t(b)                                    */
/*-----------------------------------------------------------------------*/

HlExprList *HlExprList::Distribute(types t, HlExprList *e1)
{

    if (e1->Is(PLUS))
    {

        HlExprList *h = N(PLUS);

        for (int i = 0; i < e1->Length(); i++)
            h->apparg(C(N(t), N(e1->arg(i))));

        return h;
    }

    else
        return C(N(t), N(e1));
}

/*-----------------------------------------------------------------------*/
/* Distribute: Wendet t bezueglich der Summen distributiv an             */
/*             (t,a,b)      ->  t(a,b)                                   */
/*             (t,a+b,c)    ->  t(a,c) + t(b,c)                          */
/*             (t,a,b+c)    ->  t(a,b) + t(a,c)                          */
/*             (t,a+b,c+d)  ->  t(a,c) + t(a,d) + t(b,c) + t(b,d)        */
/*-----------------------------------------------------------------------*/

HlExprList *HlExprList::Distribute(types t, HlExprList *e1, HlExprList *e2)
{
    HlExprList *h;

    if (e1->Is(PLUS))
    {

        if (e2->Is(PLUS))
        {
            h = N(PLUS);
            for (int i = 0; i < e1->Length(); i++)
            {
                for (int k = 0; k < e2->Length(); k++)
                {
                    h->apparg(C(N(t), N(e1->arg(i)), N(e2->arg(k))));
                }
            }
            return h;
        }

        else
        {
            h = N(PLUS);
            for (int i = 0; i < e1->Length(); i++)
            {
                h->apparg(C(N(t), N(e1->arg(i)), N(e2)));
            }
            return h;
        }
    }

    else if (e2->Is(PLUS))
    {
        h = N(PLUS);
        for (int i = 0; i < e2->Length(); i++)
        {
            h->apparg(C(N(t), N(e1), N(e2->arg(i))));
        }
        return h;
    }

    else
        return C(N(t), N(e1), N(e2));
}
