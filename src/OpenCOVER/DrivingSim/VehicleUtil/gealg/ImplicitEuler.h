/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------
//Author: Florian Seybold, 2009
//www.hlrs.de
//----------------------------------

#ifndef __ImplicitEuler_h
#define __ImplicitEuler_h

#include "GeometricAlgebra.h"

namespace gealg
{

template <class ET, class ST>
class ImplicitEuler
{
public:
    ImplicitEuler(const ET &et, ST &st)
        : exprTuple(et)
        , stateTuple(st)
    {
    }

    ST &integrate(const double &h)
    {
        ST k1;
        evaluateElement(k1, stateTuple);
        ST k2;
        evaluateElement(k2, addAndMultiply(stateTuple, k1, 0.5 * h));
        ST k3;
        evaluateElement(k3, addAndMultiply(stateTuple, k2, 0.5 * h));
        ST k4;
        evaluateElement(k4, addAndMultiply(stateTuple, k3, h));

        addAndMultiplyToStateTuple(k1, 1.0 / 6.0 * h, k2, 2.0 / 6.0 * h, k3, 2.0 / 6.0 * h, k4, 1.0 / 6.0 * h);

        return stateTuple;
    }

protected:
    static const int exprTupleSize = std::tr1::tuple_size<ET>::value;

    void evaluateElement(ST &result, const ST &argument)
    {
        EvaluateElement<>::operate(result, argument, exprTuple);
    }
    template <int I = exprTupleSize, int dummy = 0>
    struct EvaluateElement
    {
        static void operate(ST &result, const ST &argument, const ET &exprTuple)
        {
            std::tr1::get<exprTupleSize - I>(result).evaluateLazy(std::tr1::get<exprTupleSize - I>(exprTuple), argument);
            EvaluateElement<I - 1>::operate(result, argument, exprTuple);
        }
    };
    template <int dummy>
    struct EvaluateElement<0, dummy>
    {
        static void operate(ST &, const ST &, const ET &)
        {
        }
    };

    ST addAndMultiply(const ST &l, const ST &h, const double &f)
    {
        ST result;
        TupleAddAndMultiply<>::operate(result, l, h, f);
        return result;
    }
    template <int I = exprTupleSize, int dummy = 0>
    struct TupleAddAndMultiply
    {
        static void operate(ST &r, const ST &l, const ST &h, const double &f)
        {
            std::tr1::get<exprTupleSize - I>(r).evaluateLazy(std::tr1::get<exprTupleSize - I>(l) + std::tr1::get<exprTupleSize - I>(h) * f);
            TupleAddAndMultiply<I - 1>::operate(r, l, h, f);
        }
    };
    template <int dummy>
    struct TupleAddAndMultiply<0, dummy>
    {
        static void operate(ST &, const ST &, const ST &, const double &)
        {
        }
    };

    void addAndMultiplyToStateTuple(const ST &l1, const double &f1,
                                    const ST &l2, const double &f2,
                                    const ST &l3, const double &f3,
                                    const ST &l4, const double &f4)
    {
        TupleAddAndMultiplyToStateTuple<>::operate(stateTuple, l1, f1, l2, f2, l3, f3, l4, f4);
    }
    template <int I = exprTupleSize, int dummy = 0>
    struct TupleAddAndMultiplyToStateTuple
    {
        static void operate(ST &stateTuple, const ST &l1, const double &f1,
                            const ST &l2, const double &f2,
                            const ST &l3, const double &f3,
                            const ST &l4, const double &f4)
        {
            std::tr1::get<exprTupleSize - I>(stateTuple).evaluateLazy(std::tr1::get<exprTupleSize - I>(stateTuple) + std::tr1::get<exprTupleSize - I>(l1) * f1 + std::tr1::get<exprTupleSize - I>(l2) * f2 + std::tr1::get<exprTupleSize - I>(l3) * f3 + std::tr1::get<exprTupleSize - I>(l4) * f4);
            TupleAddAndMultiplyToStateTuple<I - 1>::operate(stateTuple, l1, f1, l2, f2, l3, f3, l4, f4);
        }
    };
    template <int dummy>
    struct TupleAddAndMultiplyToStateTuple<0, dummy>
    {
        static void operate(ST &, const ST &, const double &,
                            const ST &, const double &,
                            const ST &, const double &,
                            const ST &, const double &)
        {
        }
    };

    ET exprTuple;
    ST &stateTuple;
};
}

#endif
