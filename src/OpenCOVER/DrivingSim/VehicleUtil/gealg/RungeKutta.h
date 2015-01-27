/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------
//Author: Florian Seybold, 2009
//www.hlrs.de
//----------------------------------

#ifndef __RungeKutta_h
#define __RungeKutta_h

#include "GeometricAlgebra.h"

namespace gealg
{

template <class ET, class ST, uint8_t NST = tuple_size<ET>::value> //NST: Number of integratable states, rest are helper evaluations und input elements
class RungeKutta
{
public:
    RungeKutta(const ET &et, ST &st)
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
    static const int exprTupleSize = tuple_size<ET>::value;
    static const int stateTupleSize = tuple_size<ST>::value;

    //void evaluateElement(ST& result, const ST& argument) {
    void evaluateElement(ST &result, ST argument)
    {
        CopyInputElement<>::operate(argument, stateTuple);
        EvaluateAndCopyHelperElement<>::operate(result, argument, exprTuple);
        EvaluateStateElement<>::operate(result, argument, exprTuple);
    }

    template <int I = (stateTupleSize - exprTupleSize), int dummy = 0>
    struct CopyInputElement
    {
        static void operate(ST &argument, const ST &stateTuple)
        {
            get<stateTupleSize - I>(argument) = get<stateTupleSize - I>(stateTuple);
            //if(stateTupleSize-I == 20) std::cerr << "Copied input element " << stateTupleSize-I << ": " << get<stateTupleSize-I>(argument) << std::endl;
            CopyInputElement<I - 1>::operate(argument, stateTuple);
        }
    };
    template <int dummy>
    struct CopyInputElement<0, dummy>
    {
        static void operate(ST &, const ST &)
        {
        }
    };

    template <int I = (exprTupleSize - NST), int dummy = 0>
    struct EvaluateAndCopyHelperElement
    {
        static void operate(ST &result, ST &argument, ET &exprTuple)
        {
            //get<exprTupleSize-I>(argument).evaluateLazy(get<exprTupleSize-I>(exprTuple), argument);
            get<exprTupleSize - I>(argument) = get<exprTupleSize - I>(exprTuple)(argument);
            get<exprTupleSize - I>(result) = get<exprTupleSize - I>(argument);
            //if(exprTupleSize-I == 16) std::cerr << "Eval helper element " << exprTupleSize-I << ": " << get<exprTupleSize-I>(argument) << std::endl;
            EvaluateAndCopyHelperElement<I - 1>::operate(result, argument, exprTuple);
        }
    };
    template <int dummy>
    struct EvaluateAndCopyHelperElement<0, dummy>
    {
        static void operate(ST &, ST &, const ET &)
        {
        }
    };

    template <int I = NST, int dummy = 0>
    struct EvaluateStateElement
    {
        static void operate(ST &result, const ST &argument, ET &exprTuple)
        {
            //get<NST-I>(result).evaluateLazy(get<NST-I>(exprTuple), argument);
            get<NST - I>(result) = get<NST - I>(exprTuple)(argument);
            EvaluateStateElement<I - 1>::operate(result, argument, exprTuple);
        }
    };
    template <int dummy>
    struct EvaluateStateElement<0, dummy>
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
    template <int I = NST, int dummy = 0>
    struct TupleAddAndMultiply
    {
        static void operate(ST &r, const ST &l, const ST &h, const double &f)
        {
            //get<NST-I>(r).evaluateLazy(get<NST-I>(l) + get<NST-I>(h) * f);
            get<NST - I>(r) = (get<NST - I>(l) + get<NST - I>(h) * f);
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
        TupleAddAndMultiplyHelperIntoStateTuple<>::operate(stateTuple, l1, f1, l2, f2, l3, f3, l4, f4);
    }
    template <int I = NST, int dummy = 0>
    struct TupleAddAndMultiplyToStateTuple
    {
        static void operate(ST &stateTuple, const ST &l1, const double &f1,
                            const ST &l2, const double &f2,
                            const ST &l3, const double &f3,
                            const ST &l4, const double &f4)
        {
            //get<NST-I>(stateTuple).evaluateLazy(get<NST-I>(stateTuple)
            get<NST - I>(stateTuple) = (get<NST - I>(stateTuple)
                                        + get<NST - I>(l1) * f1
                                        + get<NST - I>(l2) * f2
                                        + get<NST - I>(l3) * f3
                                        + get<NST - I>(l4) * f4);
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
    template <int I = (exprTupleSize - NST), int dummy = 0>
    struct TupleAddAndMultiplyHelperIntoStateTuple
    {
        static void operate(ST &stateTuple, const ST &l1, const double &f1,
                            const ST &l2, const double &f2,
                            const ST &l3, const double &f3,
                            const ST &l4, const double &f4)
        {
            get<exprTupleSize - I>(stateTuple) = (get<exprTupleSize - I>(l1) * f1
                                                  + get<exprTupleSize - I>(l2) * f2
                                                  + get<exprTupleSize - I>(l3) * f3
                                                  + get<exprTupleSize - I>(l4) * f4);
            TupleAddAndMultiplyHelperIntoStateTuple<I - 1>::operate(stateTuple, l1, f1, l2, f2, l3, f3, l4, f4);
        }
    };
    template <int dummy>
    struct TupleAddAndMultiplyHelperIntoStateTuple<0, dummy>
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
